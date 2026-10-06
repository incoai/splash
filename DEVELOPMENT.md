# Development

This document runs from using Splash to working on it: running and
configuring the server, the [API](#api), the [models](#models) it loads, its
[internals](#internals), then [validation](#validate) and
[packaging](#package).

Building from source needs Apple Silicon with macOS 26.4+, Xcode 26 or newer,
Python 3.12–3.14, and a Metal 4 compiler with `uint4b_format` tensor support.
The macOS 26.2 SDK can compile the host code, but Xcode 26.2's default Metal
component cannot compile the kernels; select a newer Metal toolchain when
using that SDK. Packaged users need none of these development tools.

## Build and run

```sh
git clone https://github.com/incoai/splash.git
cd splash
make -j4
./splash serve --model mlx-community/Qwen3.8-27B-4bit
```

`--model` names an upstream Hugging Face model: an MLX affine 4-bit, group-64
repository such as `mlx-community/Qwen3.8-27B-4bit`, or a GGUF repository and
variant, `OWNER/REPO:VARIANT`, such as `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`.
Splash identifies the model from its own metadata and pairs the DFlash2 draft
trained for it. The first serve sets up Python dependencies and downloads the
model and its draft; each start loads the weights into memory
([Weight loading](#weight-loading)) and follows the model's revision
([Revisions](#revisions)). Legacy Splash packages remain loadable
([Legacy Splash packages](#legacy-splash-packages)). Public repositories need
no login; private or gated ones need `HF_TOKEN` or `hf auth login`. Ctrl+C
stops serving, and a second Ctrl+C stops the engine at once; stop before
upgrading.

### Engine restarts and sleep

An engine that fails is restarted at once. If it fails again within 60 s of
starting, the next restart waits 5 s; after a third such failure (failed
restarts count) Splash stops restarting it, and requests get 500 `engine_failed`
until the server is restarted; its message names the crash trace when
`SPLASH_CRASH_TRACE=1` recorded one
([crash traces](#failures-and-crash-traces)). Meanwhile generation requests get
503 `engine_recovering`, whose message names the last failure. An engine whose
loop leaves a status request unanswered for 30 seconds, while requests are
pending or during a background status refresh, is failed and restarted by the
same rules.

Timeouts and keep-alives count only time the Mac is awake: a request in flight
when the Mac sleeps continues when it wakes. While requests run, Splash keeps
the Mac from sleeping automatically, as `caffeinate -i` does; the display may
still sleep, and closing the lid or choosing Sleep still sleeps the Mac.
`serve --allow-idle-sleep` lets it sleep automatically.

### Shell completion

A Homebrew install links the completions into Homebrew's directories for them
under `brew --prefix`: `share/zsh/site-functions`, `etc/bash_completion.d` and
`share/fish/vendor_completions.d`, where a shell set up for Homebrew's
completions finds them. In a source checkout, source
`install/completions/splash.bash` for Bash, `install/completions/_splash` for
Zsh after `compinit`, or `install/completions/splash.fish` for fish.

Completion suggests commands, the official model IDs (bundled, and as
`splash serve` last refreshed them), the upstream models the README starts with
and installed models, a GGUF's `OWNER/REPO:VARIANT` included, without network
access.

## Server configuration

### Server options

`splash serve --help` lists every option with examples:

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | Required | The upstream model, `OWNER/REPO`, or a GGUF's `OWNER/REPO:VARIANT`. See [upstream model loading](#upstream-model-loading). |
| `--revision` | Default branch | Select an upstream target branch, tag, or commit. See [revisions](#revisions). |
| `--draft-model` | Matching DFlash2 checkpoint | Override the draft with a compatible repository or local directory. See [drafts](#drafts). |
| `--language-only` | Off | Skip vision loading; image and PDF input is rejected. See [vision](#vision). |
| `--offline` | Off | Start the installed model without asking the Hub, as `HF_HUB_OFFLINE=1` does. See [revisions](#revisions). |
| `--port` | `SPLASH_PORT` or `8000` | HTTP port. |
| `--host` | `127.0.0.1` | HTTP bind address. See [network and authentication](#network-and-authentication). |
| `--allowed-host` | No extra names | Additional HTTP Host name, e.g. `mymac.local`; repeatable. |
| `--allowed-origin` | No other origin | Origin whose pages may call the API from a browser or webview, e.g. `tauri://localhost`; `'*'` for any; repeatable. |
| `--api-key` | `SPLASH_API_KEY` or none | Require this key, as a bearer token or `x-api-key`, on every route but the chat page, `/health` and `/ready`. |
| `--no-webui` | Off | Disable the chat page. |
| `--max-memory` | Auto | Lower the ceiling on Metal allocations, e.g. `28G`; not combined process RSS. See [memory and context](#memory-and-context). |
| `--max-context` | Auto | Set the context limit, e.g. `100K`. See [memory and context](#memory-and-context). |
| `--kv-format` | `int8` | Target KV storage: `int8` or `bf16`. See [KV cache precision](#kv-cache-precision). |
| `--idle-release` | `10m` | Time without a request before the engine unwires its memory and frees the weights; the next request restores them. `off` keeps both. See [weight loading](#weight-loading). |
| `--max-cache-disk` | `0` (off) | SSD quota for cached KV pages and states, e.g. `16G`; kept for the session, or across restarts with `--persistent-cache`. See [SSD cache](#ssd-cache). |
| `--persistent-cache` | Off | Keep the SSD cache across restarts; needs `--max-cache-disk`. See [persistent cache](#persistent-cache). |
| `--cache-dir` | `~/Library/Caches/Splash/prefix-cache` | Where `--persistent-cache` keeps its files; needs `--persistent-cache`. |
| `--served-model-name` | None | Additional model name the API accepts; repeatable. See [API model aliases](#api-model-aliases). |
| `--announce-served-name` | Off | Report the first `--served-model-name` in responses; needs `--served-model-name`. |
| `--default-reasoning-effort` | `SPLASH_DEFAULT_REASONING_EFFORT` or the template's | Effort of Chat and Responses requests that set none. See [default reasoning effort](#default-reasoning-effort). |
| `--max-request-size` | `128M` | Largest HTTP request body. See [request limits](#request-limits-and-timeouts). |
| `--max-image-pixels` | `4194304` | Maximum resized pixels per image. An image's vision scratch grows with its patches (pixels / 256), to about 600 MiB at the default. |
| `--request-timeout` | None | Time a request may take from its arrival; a request's own `timeout` can only shorten it. |
| `--queue-size` | `32` | Requests admitted at once, running or waiting; more get 503 with `Retry-After`. |
| `--decode-share` | `0.5` | Decode time owed per unit of prefill time while other requests generate. Higher keeps their output faster during a long prompt and slows that prompt; `0` alternates one command each. |
| `--disable-ane` | Off | Prefill on the GPU alone. By default a dense model's long prompts also use the Neural Engine when that is faster. See [Neural Engine prefill](#neural-engine-prefill). |
| `--allow-idle-sleep` | Off | Let the Mac sleep automatically while requests run; by default it stays awake until they finish (the display may still sleep). |

Sizes (`--max-memory`, `--max-cache-disk`, `--max-request-size`) are a whole
number of bytes, or one with a `K`, `M` or `G` suffix in binary units: `28G`
is 28 GiB, and `28GB`, `28GiB` and `28g` say the same. Token counts
(`--max-context`) are a whole number, or one with a `K` suffix of 1024 tokens:
`100K` is 102,400 tokens. `--max-memory` and `--max-context` also take `auto`,
their default, and `--max-cache-disk 0` turns the SSD cache off. Durations
(`--idle-release`, `--request-timeout`) are seconds, or a number with an `s`,
`m` or `h` suffix, such as `90s`, `30m` or `1.5h`; `--idle-release off` never
releases.

### Network and authentication

The default listener is `127.0.0.1:8000`. To accept LAN connections:

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --host 0.0.0.0 --api-key YOUR_KEY
```

Connect to the server's LAN IP. `--host` selects the IPv4 bind address;
`--allowed-host NAME` accepts an additional HTTP Host name, such as the Mac's
`.local` name, a custom DNS name or a proxy hostname. It does not change the
listener or allowlist client IPs. A request that names the server any other way
gets 403 with the `--allowed-host` flag that would accept it; the check keeps out
web pages that rebind a DNS name to this address. The chat page works over plain
HTTP from another machine too.

Set `SPLASH_API_KEY` in the server and agent shells to require authentication;
`serve --api-key KEY` overrides the server's environment value. Every request
then needs `Authorization: Bearer KEY` or `x-api-key: KEY` (one that sends both
needs the key in both), `/status` and `/metrics` included; only the chat page,
`/health`, `/ready` and browsers' `OPTIONS` preflights stay public. Enter the
key in the chat page to send requests; the page does not persist it. Use
`serve --no-webui` to disable the page. Authentication is off by default, and
the server then ignores any key a client sends, so a client that insists on
one can send any value.

A page served elsewhere, or an application's webview such as Jan's and other
Tauri apps', calls the API from another origin, which its browser names in
`Origin`. `--allowed-origin ORIGIN` admits one, such as `tauri://localhost` or
`http://localhost:3000`; repeat it for more, or pass `'*'` for every origin. The
server then answers the browser's preflight `OPTIONS` request and names the
origin in `Access-Control-Allow-Origin` on its responses, which let the page
read `Retry-After` and `WWW-Authenticate` too. A page of any other origin gets
403. Its browser hides that response from the page, which sees a network error,
so the server prints the refused origin and the `--allowed-origin` flag that
would accept it, once per origin. A `file://` page or a sandboxed frame sends
`Origin: null` instead, which only `'*'` admits; the server does not print its
refusal. Origins match exactly: a pattern such as `tauri://*` or
`http://*.example.com` is refused at startup; only a bare `'*'` admits every
origin. With `'*'` every page open in a browser that reaches the server can use
it, so set `--api-key` too; the server warns at startup without one.

Use `--port 8001` or set `SPLASH_PORT=8001` to select another port. The agent
launchers take the same `--port`, as in `splash pi --port 8001`, or read the
same `SPLASH_PORT`; `--port` overrides it. They pass every other argument,
`--help` included, to the agent; to pass an agent's own `--port`, start its
arguments with `--`, as in `splash opencode --port 8001 -- --port 4096`.
Separate ports allow separate servers. Each plans its memory from the whole Mac,
as if it ran alone, so give each a `--max-memory` that leaves room for the
others. The packaged agent launchers connect to loopback, so use a listener
that includes loopback when launching agents locally.

### Memory and context

Splash sizes memory and context when it starts. Its Metal budget is the GPU's
recommended working set less a small margin; `--max-memory` can only lower it,
such as `--max-memory 28G` to leave room for other applications. The budget
caps Metal allocations, not combined process RSS. The context limit is the
model's native window, 256K tokens for both supported families, when the budget
holds it beside the weights and, unless `--disable-ane`, the
[Neural Engine split](#neural-engine-prefill) this Mac's calibration of a dense
model takes, and otherwise as much as the budget holds beside them.
`--max-context` sets it, such as `--max-context 100K`, up to what the budget
holds without the split: beyond what the split leaves, the split takes fewer
channels or the GPU runs alone, with a warning. A larger value stops startup
with the most the model and memory allow.

The startup summary and `maximum_context_tokens` in `/status` show the effective
context limit. `/v1/models` and `/v1/models/{id}` report the same limit as
`max_model_len` and its compatibility alias `context_length`, including model
aliases. Clients can impose a smaller limit. This is a capacity limit, not a
guarantee of a fast first token for a long uncached prompt. If the model cannot
fit, startup prints a memory budget breakdown and stops.

### KV cache precision

Select the target KV format when starting the server:

```bash
splash serve --model mlx-community/Qwen3.8-27B-4bit --kv-format bf16
```

BF16 avoids target KV quantization, uses approximately twice the target KV
memory, and can be slower at long contexts. Model weights are unchanged. Restart
to switch formats. Omit `--kv-format` or use `--kv-format int8` for the default.
The [SSD cache](#ssd-cache) supports both formats, preserving their stored
bytes without further quantization.

### SSD cache

`--max-cache-disk SIZE` keeps cached prompts on the SSD when memory runs short.
Default: `0` (off). Evicted KV pages and request states that a later request
could resume from are written to the SSD within that quota, and a request that
matches them reads them back instead of computing that part of its prompt
again.

Without it, a request that runs out of memory cannot keep its progress
checkpoints and replays its prompt after each suspension. With it off, startup
suggests it in one line when the memory available at startup may not hold the
advertised context. It does not raise the context limit. A quota smaller than
the working set can cause repeated reads and writes; it is not a write-rate
limit. Its state writes stage through a buffer of one state (109 MiB for 35B,
187 MiB for 27B), which the memory plan sets aside within `--max-memory`; a
quota too small for one state leaves it off.

Restarts recompute everything unless `--persistent-cache` is on (see
[persistent cache](#persistent-cache)); `--max-cache-disk` alone keeps evicted
KV pages and states for the session only.

### Persistent cache

`--persistent-cache` keeps the SSD cache across restarts of the same model, so a
restart, an upgrade or a crash keeps conversations' prefixes. It needs
`--max-cache-disk` and shares its quota.

Files live in `~/Library/Caches/Splash/prefix-cache/<namespace>/` (mode 0700;
`--cache-dir` sets the root). The namespace digests the model manifests, the KV
and state layouts and the cache format, so no model reads another's cache. One
engine holds a directory at a time; another waits 20 s, then uses temporary
files. A namespace unused for 14 days is removed.

Each conversation's newest restore point at least 2048 tokens deep is written
10 s after publication, unless the conversation has gone past it; its RAM copy
stays. These writes pause while the cache has written 128 GiB in the last hour.
A full quota keeps restore points whole: it drops copies no point needs first,
then the oldest point.

A start takes back every recorded point whose chain is whole. A clean stop
writes the waiting points (newest first, up to 6 s) and syncs the files. After
a crash the next start takes back what reached the disk and runs on probation
for a minute; if it fails then, the start after it begins empty. `/status`
reports all this under `disk` (`persistent`, `kv_copies`, `taken_back`,
`write_behind`).

## Model storage

Allow space for the target and BF16 draft downloads: up to about 21 GB for the
Qwen3.8-27B 4-bit examples and 24 GB for Qwen3.6-35B-A3B. Other variants have
different sizes. Downloads use the Hugging Face cache, and `brew upgrade splash`
keeps the installed models. Splash keeps no other copy of the weights on disk.
Agents keep their sessions in their own homes
([files Splash writes](#files-splash-writes)).

### Revisions

Each start follows the model's default branch, or the branch or tag
`--revision` names: one Hub request of at most 5 seconds asks for its current
commit, and when the Hub answered, another asks for the commit of the draft
repository's default branch ([Drafts](#drafts)). A new commit downloads only
the files that changed and replaces the installed model once it is installed.
There is no update flag; to stay on one commit, pass it as `--revision`. A
40-hex `--revision` never moves, nor does its draft, and `--offline` (or
`HF_HUB_OFFLINE=1`) forbids the Hub: both start the installed model without a
request.

When the Hub does not answer, or a new commit cannot be installed, the
installed model starts, with a line naming the reason. Without the Hub, Splash
rebuilds a missing or damaged installation from the files it downloaded
before, for the same revision. It pins the commits it installs in the Hugging
Face cache (`refs/splash`), so pruning the cache keeps an installed model's
files. [Installation](#installation) gives the mechanics.

### Model cache

To download new models to another disk, set the cache location before serving:

```sh
HF_HUB_CACHE=/Volumes/Models/huggingface splash serve --model mlx-community/Qwen3.8-27B-4bit
```

`HF_HUB_CACHE` selects the Hugging Face download cache. Alternatively, set
`HF_HOME` to relocate the Hugging Face home directory, including its default
`hub` cache. Model links stay in Splash's data directory; existing downloads
are not moved.

### Files Splash writes

| Path | Holds |
| --- | --- |
| `~/Library/Application Support/Splash` | The data directory of a packaged install: `models/`, the installed models' links into the Hugging Face cache and the metadata derived from GGUFs; `runtime/`, the locks of running servers; `catalog/`, the official model list `splash serve` last fetched; and `thinking.key`, which encrypts the Messages reasoning a response hides. A source checkout keeps the first three in `install/models` and `build/runtime`. |
| Hugging Face cache (`HF_HUB_CACHE`, by default `~/.cache/huggingface/hub`) | Model and draft downloads, with Splash's pins under `refs/splash` ([revisions](#revisions)). |
| `~/Library/Caches/Splash/prefix-cache` (`--cache-dir`) | The [persistent cache](#persistent-cache), with `--persistent-cache` only. |
| `$TMPDIR/splash-cache-*` | The [SSD cache](#ssd-cache)'s files without `--persistent-cache`, or when another process holds the persistent cache's directory or it cannot be used: one of KV pages and one of states, together at most about `--max-cache-disk`. Each is unlinked as it is created, so no directory lists it, and its space returns when the engine exits, even after a crash. `--cache-dir` does not move them; they follow `TMPDIR`. |
| `~/Library/Logs/Splash/crash` | [Crash traces](#failures-and-crash-traces), with `SPLASH_CRASH_TRACE=1` only. |
| Pi's `models.json` (`~/.pi/agent`, or `PI_CODING_AGENT_DIR`) and the Hermes profile (`~/.hermes/profiles/splash`, or `splash-<port>`) | The settings `splash pi` and `splash hermes` write ([agent launchers](#agent-launchers)). Every agent keeps its sessions in its own home; `splash claude`, `codex` and `opencode` pass their settings at launch and write none. |

## API model aliases

Repeat `--served-model-name NAME` to accept additional API model IDs. The full
`--model` ID still selects the model. By default `/v1/models` lists that ID
first, followed by unique aliases; each alias's `root` identifies the loaded
model, and generation and scoring responses report the real model ID even when
requested through an alias. The model list and lookup support both names.

Some clients check a response's `model` against the name they requested and
reject the real ID. For them, add `--announce-served-name`, which requires
`--served-model-name`: responses then report the first alias, whichever
accepted name the request used, and `/v1/models` lists that alias first and
gives every other entry, the real ID included, that alias as its `root`, so
`splash <client>` configures clients with it. `/status` keeps reporting the
loaded model.

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --served-model-name local-qwen
```

Aliases cannot contain whitespace, control characters, `\`, `%`, `?`, `#`,
or empty, `.` or `..` path segments. This keeps model discovery URLs unambiguous.

## Default reasoning effort

`--default-reasoning-effort` (or `SPLASH_DEFAULT_REASONING_EFFORT`) sets the
fallback for Chat `reasoning_effort` and Responses `reasoning.effort` when absent
or null. Accepted values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`,
`max`. An explicit request value wins; the CLI flag takes precedence over the
environment. Unset, the model's template default is unchanged. An effort, the
default or a request's own, reaches the template as `enable_thinking`, false
for `none`, and as `reasoning_effort`, not as a token budget, so a template that
switches reasoning by either follows it. A template that rejects an effort
renders an alias: `xhigh` for `high` and `max`, `low` for `minimal`; one that
rejects `none` renders by `enable_thinking` alone while thinking is off.

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --default-reasoning-effort none
```

`/apply-template` uses the same default. Anthropic `thinking` keeps its protocol
semantics (off when omitted); judgment endpoints always disable thinking. The
built-in chat page sends no effort unless the user picks one.

A Chat request's `chat_template_kwargs`, as vLLM and SGLang accept them, are
passed to the template as variables and outrank the effort, so
`{"enable_thinking": false}` turns reasoning off and reaches the template as
`reasoning_effort` `none` too, as effort `none` does, and `true` turns it on,
at the template's default effort where the effort is `none`. `enable_thinking`
must be a boolean; null leaves it to the effort. They cannot set what Splash
passes itself, such as `tools` or `add_generation_prompt`.

## Agent launchers

Agents must already be installed; `splash claude|opencode|codex|hermes|pi`
connects to the running server. Arguments pass through, for example
`splash codex resume --last` or `splash hermes chat -q "Hello"`.

VS Code's local agent connects through its custom endpoint provider; see
[VS Code setup](docs/vscode.md) for reasoning controls and output limits.

`splash pi` adds a `splash` provider to Pi's `models.json` (`splash-<port>` for
a server on another port), preserving other providers, settings and sessions.
The chat page names no model, and the launchers configure each client with the
model ID the running server reports, so a client need not list a model in its
own catalog for Splash to serve it under its full repository ID.
`splash opencode`, `pi` and `hermes` configure an output limit per response of
32K tokens (`CLIENT_RESPONSE_TOKENS` in `install/clients.py`); OpenCode and
Hermes, which reserve it out of the context they compact at, get a quarter of a
context under 128K instead. Hermes 2026.9.7 and later ignore it and, like Codex,
leave the limit to the server; Claude Code keeps its own, which
`CLAUDE_CODE_MAX_OUTPUT_TOKENS` raises.

`splash hermes` runs Hermes in the `splash` profile (`splash-<port>`) of the
user's Hermes root, `~/.hermes` or the root `HERMES_HOME` belongs to, and
creates it with `hermes profile create` on first use. It writes only the
profile's model settings; Hermes's tools and the root's `config.yaml` remain
the user's.

With `SPLASH_API_KEY` set, Pi's `models.json` and the Hermes profile name the
variable instead of holding the key: Pi and Hermes read the key from their
environment, so one started without `splash` needs the variable in its shell.
A file these launchers rewrite keeps its mode; one they create is private.

## Upgrading from earlier versions

Before 1.2.0, Splash gave Hermes a home of its own: `hermes` in
`~/Library/Application Support/Splash/runtime` in a release, or in a checkout's
`install/agents` (earlier `install/profiles` or `build/runtime`), and
`ports/<port>/hermes` there for a server on another port. Hermes took each home
for its root, except `install/profiles/hermes`, whose root was the checkout's
`install/`: a Hermes that manages its own runtime installed its tools in its
root and pointed the `hermes` command at them. An upgrade leaves these homes and
their sessions in place, and Splash no longer uses them. While one still exists,
run `hermes pm install` once in a normal shell; it installs Hermes's tools under
`~/.hermes/tools` and points the `hermes` and `hermes-acp` launchers in
`~/.hermes/hermes-agent/.hermes/bin` at them. Once those launchers name no old
root, or do not exist, as with Hermes before its managed runtime, delete the old
homes with the sessions Splash started there, and the files Hermes added to a
checkout's `install/`, which `git status` lists as untracked.

Splash 1.1 kept prepared copies of the weights in
`~/Library/Caches/Splash/weights` (or the directory `SPLASH_WEIGHT_CACHE`
named), which nothing reads now and which can be deleted.

## API

Splash serves OpenAI's Chat Completions, Completions and Responses APIs and
Anthropic's Messages API. OpenAI SDKs take the base URL
`http://127.0.0.1:8000/v1` and Anthropic SDKs `http://127.0.0.1:8000`, on the
port `--port` sets. With [authentication](#network-and-authentication) off any
API key works; with it on, clients send the server's key. `model` names the
served model ID or one of its [aliases](#api-model-aliases), as `/v1/models`
lists them; another name returns 404 `model_not_found`, which Messages, whose
errors carry no code, answers as a `not_found_error`. Chat, text completion and
Responses requests that omit it get the served model; Messages requires it.

### Endpoints

| Method and path | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/completions` | OpenAI text completions |
| `POST /v1/responses` | OpenAI Responses |
| `GET`, `DELETE /v1/responses/{id}` | A stored response ([Responses state](#responses-state)) |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/messages/count_tokens` | A Messages request's prompt tokens, images included, as `{"input_tokens": N}`, without inference |
| `GET /v1/models`, `/v1/models/{id}` | The served model and its aliases, with the context limit |
| `POST /tokenize`, `/apply-template` | A text's tokens, or a conversation's rendered prompt, without inference |
| `POST /v1/judgments`, `/v1/systemone` | Scores over fixed answers ([judgment contracts](#judgment-contracts)) |
| `GET /status`, `/metrics` | JSON status and Prometheus metrics ([status](#status-readiness-and-metrics)) |
| `GET /health` | 200 while the server answers; public |
| `GET /ready` | 200 while the engine is ready, else 503; public |
| `GET /` | The chat page; public, and gone with `--no-webui` |

### Request fields

Chat Completions, text completions, Responses and Messages share these fields:

- `stream`: stream server-sent events. In Chat and text completions,
  `stream_options.include_usage` adds usage to the stream; their non-streaming
  responses always include it.
- `priority`: `foreground`, `normal` (the default) or `background`. A request
  waits while one of a higher priority runs, and when lanes or memory run short
  a higher priority suspends a running request of a lower one
  ([admission and eviction](#admission-and-eviction)).
- `timeout`: the seconds a request may take from its arrival; it can only
  shorten `--request-timeout`. A request that runs out returns 504
  `request_timeout`.
- `seed`: an unsigned 64-bit integer, random when omitted
  ([sampling](#sampling)).
- `stop`, in Messages `stop_sequences`: a string or up to four strings. It
  cannot be combined with tools a call may name or with structured output,
  since it could cut either short.
- `return_progress`, with `stream` true: prompt progress before the output
  ([output limits and streaming](#output-limits-and-streaming)).

Chat Completions also takes `messages`, `tools`, `tool_choice` and
`parallel_tool_calls` ([tool calls](#tool-calls)), `response_format`
([structured output](#structured-output)), `reasoning_effort` and
`chat_template_kwargs` ([default reasoning effort](#default-reasoning-effort)),
`preserve_thinking`, which reaches the template (the Qwen templates keep earlier
turns' reasoning with it), `max_completion_tokens` or `max_tokens`,
`ignore_eos` and the [sampling](#sampling) fields. The reply's reasoning comes
in `reasoning_content`, beside `content`, in the message and in each stream
delta, and an assistant message sent back may carry its `reasoning_content`.
`n` other than 1 and `logprobs: true` return 400.

`POST /v1/completions` serves OpenAI's legacy text completions. `prompt` is one
string, which the loaded tokenizer encodes as a raw prompt, adding its own
special tokens such as a BOS and recognizing special-token strings, or one array
of token IDs from its vocabulary. No chat template, reasoning split, tools or
images apply: `text` is the generated text, decoded without special tokens.
`max_tokens` defaults to 16, OpenAI's default for the endpoint as in vLLM and
SGLang, or to what the context leaves when that is less. The sampling fields
below, `seed`, `stop`, `priority`, `timeout` and `stream` with
`stream_options.include_usage` work as in Chat. Batched prompts, `suffix`,
`echo`, `logprobs`, `best_of` and `n` other than 1 are rejected.

Responses takes `input` and `instructions`, function `tools` and `namespace`
groups of them, `tool_choice`, `parallel_tool_calls`, `reasoning.effort`,
`text.format` for structured output, `max_output_tokens`, the sampling fields,
and `store` and `previous_response_id` ([Responses state](#responses-state)).
Reasoning comes back as `reasoning` output items. `conversation`,
`background: true`, `truncation` other than `disabled` and `context_management`
edits return 400.

Messages takes `max_tokens`, `system`, `messages`, custom `tools`, `tool_choice`
(`auto`, `any`, `tool` or `none`, with `disable_parallel_tool_use`),
`stop_sequences`, `temperature`, `top_p`, `top_k`, `thinking` and
`output_config`. A request reasons only when `thinking` sets it:
`{"type": "enabled"}` and `{"type": "adaptive"}` reason at
`output_config.effort`, one of `low`, `medium`, `high` (the default), `xhigh`
and `max`, while `{"type": "disabled"}` and an omitted `thinking` do not;
`budget_tokens` is not read. Reasoning comes back in `thinking` blocks. With
`thinking.display` `omitted` (or `updates`) a block carries no text, and its
reasoning travels encrypted in its `signature` under the key in `thinking.key`
([files Splash writes](#files-splash-writes)); a later request that sends the
block back has the reasoning restored. A signature Splash cannot read, such as
another provider's, keeps the block's visible text without recovering the
private reasoning. Structured output is `output_config.format` (or
`output_format`) with a JSON schema. `context_management` accepts only
`clear_thinking_20251015` with `keep: "all"`, which sets `preserve_thinking`.

`POST /tokenize` accepts `{"content":"hello","add_special":false}` and returns
`{"tokens":[...]}` using the loaded tokenizer. Special-token strings are recognized;
`parse_special:false` and `with_pieces:true` are unsupported.
`POST /apply-template` accepts Chat-style `messages`, `tools` and reasoning options,
and returns `{"prompt":"..."}` using the same template as generation.
`add_generation_prompt` defaults to true. Image prompts retain textual placeholders;
raw tokenization does not account for image embeddings (use `count_tokens` for that).
Both endpoints run without inference and share bounded preparation capacity with
`count_tokens`; they can inspect prompts larger than the serving context limit.

### Request limits and timeouts

HTTP request bodies need a `Content-Length` and are limited to 128 MiB;
`serve --max-request-size 256M` overrides this. Concurrent input bytes share a
budget of at least 512 MiB (or twice the request limit), including retained
generation inputs. This is an input-byte budget, not a process RSS limit: large
ASCII/base64 strings can use roughly twice their encoded size during JSON
parsing alone. Decoded images and object-heavy JSON need additional memory.
Oversized requests return 413; exhausted ingress capacity returns 503. Image and
model context limits apply independently. Stored Responses history is charged
before decoding. Uploads allow 30 seconds of inactivity; total upload time is
limited to 30 seconds plus the body size at 512 KiB/s (286 seconds for 128 MiB),
capped by `--request-timeout` when set. Timed-out uploads return 408 and release
their input reservation. An upload refused before it is read, such as one over
the shared budget, is still received on these terms, so a client that sends its
whole body before reading the response gets the refusal. An inference request
ends when its client disconnects, and a client that shuts down its sending side
after the request, as `nc` does at the end of its input, counts as disconnected
and gets no response. A response write waits up to 30 seconds for a client that
has stopped reading, past `--request-timeout` too, and the request counts
against `--queue-size` until then. `/status` reports `http.request_body_bytes`
and `http.max_request_bytes`. A body's JSON, and JSON inside it such as a
history tool call's arguments, nests at most 128 levels of arrays and objects;
deeper is refused as invalid JSON.

The server holds at most `--queue-size` + 64 connections at once (96 by
default), each in a connection slot. A connection waits for its request without
a thread until the request's head, its request line and headers, has arrived, or
its client has closed; one of the server's worker threads, one per slot and all
started with the server, then reads and serves the request, and the connection
keeps its slot until it has been answered. The server takes every connection the
kernel holds at each wakeup and starts no thread while connections arrive: macOS
queues at most 128 connections the server has yet to take and resets those
beyond, so the accept loop must keep up with a burst. The head is due 30 seconds
after the server accepted the connection, which is closed without a response if
the head has not arrived by then; a head that passes 64 KiB goes to its thread
unfinished, to be read with 30 seconds allowed between reads. When every slot is
taken, a connection still waiting for its head, or receiving an upload refused
unread, gives way to the new one, the longest waiting first, so stalled clients
cannot lock others out; when every slot holds a request whose head has arrived,
the new connection is refused. A connection refused, or giving way, before its
request is read gets a fixed 503 with code `frontend_overloaded` in OpenAI's
shape whatever its path ([errors](#errors)). The server then reads and drops
what its client still sends until the client closes or 2 seconds pass, so that
the client reads the 503 rather than a reset, even one that sends its request
only after the refusal. At most 1024 such connections linger at once; one more
takes the place of the one refused longest ago. The server raises its soft limit
of open files to 10240 at start (macOS starts a process at 256), so the
connections it holds cannot exhaust it.

### Sampling

Chat, text completions and Responses sample with `temperature` (default 1.0,
in [0, 2]), `top_p` (0.95), `top_k` (20, a positive integer, 0 or -1), `min_p`
(0, in [0, 1]), `presence_penalty` and `frequency_penalty` (0, in [-2, 2]) and
`repetition_penalty` (1, positive); the defaults are Qwen's generation config,
which sets no penalties, so its recommended `presence_penalty` of 1.5 for
non-thinking use must be sent explicitly. Each field that is out of range
returns 400 naming it. As in vLLM, a nonzero temperature below 0.01 samples at
0.01, and the penalties rewrite the raw logits before temperature, for greedy
requests too: repetition divides a positive logit and multiplies a negative
one for every token of the prompt or of the output so far, and presence and
frequency lower the logit of every output token by `presence_penalty` plus
`frequency_penalty` times its count. Speculative decoding stays exact: each
verified draft position counts the draft tokens before it, so a penalized
request samples as it would without a draft. Splash does not implement
`logit_bias`: a non-empty one returns 400; `null` and `{}` are accepted. Like
vLLM and HF, repetition counts every prompt token: with the Qwen templates
that includes the tool-call syntax every tool-enabled system prompt carries,
earlier tool calls and reasoning, and the `<think>` markers of the generation
prompt, so a `repetition_penalty` above 1 can delay tool calls and the end of
reasoning and change names copied from the context. Anthropic Messages defines
no penalties and no `min_p`.

In vLLM's order, after temperature `min_p` first drops every token less
likely than `min_p` times the most likely one, `top_k` then keeps the most
likely of the rest, every one when it is 0 or -1 or past the vocabulary, and
`top_p` then the fewest of those whose probabilities, renormalized over them,
sum past it. All three apply to the whole vocabulary, exactly, for the first
token, the verified draft positions and their corrections alike: the sampler
sums each row's softmax denominator, finds where its distribution ends
without sorting the vocabulary, and draws over every token the distribution
keeps. vLLM itself refuses a nonzero `min_p` with speculative decoding; here
it is one more cut of the target distribution, which acceptance and
correction read as they read the others, so sampling stays exact.

A lane's arithmetic can depend on the batch it decodes in and on how its prompt
is chunked. Concurrent requests change both, and chunk boundaries also come from
the cached boundary a request resumes from and from the junctions and
checkpoints earlier requests left. So a greedy or seeded request repeats its
output when it runs alone with the same cached prefixes; alongside other
requests, or with other prefixes cached, it can differ.

### Output limits and streaming

Chat's `max_completion_tokens` or `max_tokens` and Responses'
`max_output_tokens` bound a response's output. Omitted, the output may use
all the context the prompt leaves, as in vLLM and SGLang: the context limit,
and `--request-timeout` when set, are the only server bounds on a request that
names no limit. A value larger than what the context leaves returns 400
`context_length_exceeded` with the prompt's tokens, the value and the window,
as in vLLM and SGLang; that error and an invalid value's name the field the
request sent. Messages requires `max_tokens`; a larger value than the context
leaves generates up to the context limit, since Claude Code asks for the same
limit on every turn and does not compact for it. A response the context limit
ends then has the `stop_reason` `model_context_window_exceeded`, as in
Anthropic's API, not `max_tokens`. Messages refuses a final assistant message
(prefill) with 400; `count_tokens` still counts it.

Chat and text completions accept `"ignore_eos":true` (default false), as vLLM
and llama.cpp do: the model never selects its own stop tokens, and a draft
proposal of one is rejected, so generation runs to its output budget and
finishes with `length` unless a `stop` string ends it first. Benchmarks use it
to generate a fixed number of tokens. Constrained tool calls, output under
`tool_choice: "none"` and structured output generate under a grammar, which
decides where the output ends, so combining them with `ignore_eos` returns 400.

Streaming requests accept `"return_progress":true` (default false). Before output,
`prompt_progress` reports `{total, cache, processed, time_ms}`: prompt tokens,
initial cached tokens, completed tokens including cache, and elapsed milliseconds
since prefill admission. Updates follow completed chunks and never regress during
recovery; they are not a time estimate. Chat uses empty-delta chunks, text
completions empty-text chunks, Responses `response.in_progress`, and Messages
`ping`. Queueing and prompt preparation do not advance this counter.
Non-streaming requests cannot enable it.

Chat and text completions include a llama-server-style `timings` object, both in
non-streaming responses and in the final finish-reason chunk of a stream,
even without `include_usage`. `prompt_n` and `predicted_n` are the full prompt
and output counts; `cache_n` is the cached prompt count. `prompt_ms` measures
native start to first emission, and `predicted_ms` measures first emission to
completion. These elapsed intervals exclude the initial admission queue and
are not isolated GPU timings. `prompt_per_second` uses only uncached prompt
tokens; `predicted_per_second` excludes the entire first emission (which can
contain multiple speculative tokens). Thus rates use tokens processed in the
measured interval, not the full counts. An unavailable rate is zero, including
responses completed in one emission. Per-request draft counters are omitted
because the native runtime only reports them at batch level.

### Tool calls

With `tool_choice` `auto`, parallel calls allowed and no tool marked
`"strict": true`, nothing constrains the output: the model writes its calls in
the chat template's layout as it would without a grammar, and each call reads
back as written. A grammar constrains the output when any of these holds:

- `tool_choice` is `required` or names a tool;
- `"parallel_tool_calls": false`;
- a tool is strict;
- the request also sets a response format;
- `tool_choice` is `"none"`, which renders the tools like any other choice and
  only prevents calls.

The grammar holds each call to the template's layout and an offered tool's
name, and a strict tool's arguments to its schema; other tools' arguments stay
free. A strict tool's parameters come in schema order, as OpenAI's strict mode
writes them, which requires every field; an optional field written after a
later one cannot be placed. A Responses tool that omits `strict` is not
strict.

The model writes each argument as text, which converts by the JSON types the
tool declares for that parameter:

- A parameter that cannot be a string takes the JSON its text spells, of
  whatever type; text that is not JSON stays text. Where the tool declares a
  boolean or null, `True`, `False` and `None` also read as `true`, `false` and
  `null`, as templates that write values through Jinja's `string` filter spell
  them.
- A parameter that may be a string takes the JSON its text spells only when
  that is of another declared type (an integer counts as a number); otherwise
  it stays the text as written.
- A parameter whose type is left open, or that the tool does not declare,
  takes the JSON its text spells unless that is a string, which stays as
  written.

| Declared type | Text | Argument |
| --- | --- | --- |
| `integer` | `42` | `42` |
| `integer` | `4.5` | `4.5` |
| `boolean` | `True` | `true` |
| `["string", "number"]` | `12` | `12` |
| `["string", "number"]` | `12:00` | `"12:00"` |
| `["string", "integer"]` | `0800` | `"0800"` |
| not declared | `7` | `7` |
| not declared | `seven` | `"seven"` |

Calls are not validated against their schemas; a client reports what its tool
rejects. A `stop` string cannot be combined with tools a call may name
([request fields](#request-fields)).

The output is read as the chat template lays calls out: a call opens at
`<tool_call>` and the newline before `<function=`, and a value ends at
`\n</parameter>\n` where the next parameter, `</function>` or `</tool_call>`
follows, so a value may hold the tags in any other order, as a file an agent
writes may. Tags written otherwise are text. Besides:

- names lose the space around them, and a call that names no function is
  dropped, as is a `</think>` in text;
- a repeated parameter keeps its first value, which has streamed;
- text after a call stays content, with or without its `</tool_call>`;
- a call the token limit cuts keeps the unfinished arguments it has, which
  Chat and Responses (with status `incomplete`) return and a Messages stream
  has streamed, while a complete Messages response leaves the call out;
- JSON that would decode to a lone surrogate or nest past 122 levels keeps its
  text, which writes back as valid JSON, so that the call's arguments read back
  within the 128 levels the server reads where a request carries them deepest,
  as a `tool_use` block's `input` in a Messages history, five levels into the
  body;
- a call's opening also ends the reasoning where a call may follow.

`dev/tests/server/test_tool_call_reading.py` holds model outputs with the calls
and content they read as.

Every tool's schema is checked against its JSON Schema dialect as the request
arrives, and its arguments are framed through schema composition and local
references, which give each parameter its types. A strict tool's framed
arguments are closed, so that an object or an array takes nothing it does not
declare. Its values are written in Qwen XML: raw text where any string may be
the value, whose pattern, format and length go unenforced; otherwise one of the
values each member of the value's union lists, as its text, or JSON of a member
that lists none, leaving out array item bounds and `multipleOf` above 64 and
patterns the grammar cannot compile (look-around, word boundaries,
backreferences). The framed schemas of one request are limited to 16 MiB, and a
strict tool's parameter names cannot contain XML delimiters, start or end with
whitespace, or run past 256 characters. What a grammar writes reads back as
written: raw text ends at its first `\n</parameter>\n`, and an enumerated value
that holds one returns 400.

A tool's schema nests at most 64 levels of JSON objects and arrays, in which an
object's `properties` map is a level of its own, and cannot reference a remote
document. Hosted search is unsupported; configure client-owned tools such as
MCP.

### Structured output

Chat's `response_format` takes `{"type": "json_object"}` or
`{"type": "json_schema", "json_schema": {"schema": ...}}`. Responses takes them
in `text.format`, the second without the `json_schema` object around its
fields: `{"type": "json_schema", "name": ..., "schema": ..., "strict": ...}`.
Messages takes a schema in `output_config.format` (or
`output_format`) as `{"type": "json_schema", "schema": ...}`. A schema nests at
most 64 levels of JSON objects and arrays, as a tool's does, cannot reference a
remote document, and cannot combine `unevaluatedProperties` with
`patternProperties`. `response_format` constrains generation and validates
final output; it does not inject formatting instructions into the prompt.
Clients should describe their output requirements in their own messages.

### Images and PDFs

Images arrive inline as base64 `data:` URLs: Chat `image_url` parts, Responses
`input_image` items and Messages `image` blocks with a `base64` source; an image
URL to fetch returns 400. JPEG, PNG, WebP and GIF decode. An image file may
hold up to 32 MiB and 33,554,432 pixels, with an aspect ratio up to 200:1, and
is resized to at most `--max-image-pixels` (4,194,304 by default). A request
carries at most 64 images, PDF pages included.

PDFs arrive inline too: Chat `file` parts with `file_data`, Responses
`input_file` items and Messages `document` blocks with a base64
`application/pdf` source. Each page becomes its text and an image of at most
1,048,576 pixels. PDF input supports base64 documents within a shared 64 MiB
source/rendering budget and the native 64-image limit (one image per page).
Model context and isolated rendering limits also apply. URL inputs, `file_id`,
opening passwords and citations are unsupported.

A model served with `--language-only` refuses both with 400 ([vision](#vision)).

### Responses state

With `store` true, the default, a completed or incomplete response is kept with
its input history in a store of the server process, at most 64 MiB, which
drops the least recently used responses first and is lost when the server
restarts. A response too large for it reports `"store": false`.
`previous_response_id` continues a stored response: its history comes before
the request's `input`. An ID the store no longer holds returns 404
`previous_response_not_found`, on which clients resend the full history.
`GET /v1/responses/{id}` returns a stored response and
`DELETE /v1/responses/{id}` removes it; an unknown ID returns 404.
`/status.response_store` reports the store's entries, bytes, evictions, hits and
misses. Responses automatic truncation and unsupported history edits return
errors.

### Judgment contracts

`POST /v1/systemone` accepts the [TypeSafe System One](https://docs.typesafe.ai/)
request and response shapes: `noul`, `choice` and `score` questions over a shared
state. It works with the official `typesafe-sdk` (verified with 0.7.0). Use the
actual served model ID, not a hosted Jev model name; `/v1/models` answers both
OpenAI model discovery and the SDK's `models.list()`.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(
    base_url="http://127.0.0.1:8000",
    api_key="local",  # Use SPLASH_API_KEY's value if server authentication is on.
    model="mlx-community/Qwen3.8-27B-4bit",
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

`POST /v1/judgments` scores one [SemIf](https://github.com/TheoLeeCJ/SemIf) row of
2–16 options and returns raw option logits:

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

`POST /v1/judgments` preserves SemIf's `direct-options-v1` JSON serialization,
system prompt and A–P option order. It returns the exact rendered prompt's SHA-256,
answer token IDs, raw option logits, normalized probabilities and zero completion
tokens. Every answer label must round-trip as one token, including at the actual
assistant prompt boundary. Unsupported generation controls return errors rather
than silently changing the scoring protocol. SemIf-derived code retains its MIT
notice in `server/judgments.py`.

`POST /v1/systemone` requires the served `model`, a string/object/array `state`,
and a nonempty `questions` map. Instructions may be omitted, null or structured;
criteria descriptions may also be structured. Noul criteria may be omitted.
Choice and score domains contain 1–255 entries. Singletons return their sole
answer without inference. Other domains use deterministic, distinct single-token
slots selected from the tokenizer. All questions are validated before any inference.
A request holds at most 64 questions and 1M total prepared prompt tokens;
larger batches are rejected before any inference.
Questions run sequentially within a request under one shared deadline, allowing
prefix reuse without filling the admission queue; independent HTTP requests still
share the scheduler. Disconnects and timeouts cancel the current question.

Preparation renders each prompt once, then enforces the context limit and the
batch token budget before the per-slot boundary checks. A label can change only
the tokens after the prompt's last `<|im_end|>`, which the tokenizer encodes on
its own, so each check encodes that end with the option appended rather than
the whole prompt. The checks observe the request deadline and stop when the
client disconnects.
Prompts that exceed the context limit are rejected, not truncated.

System One validation uses 422 `detail` arrays; successful responses contain
`model`, `answers`, and `usage`, plus an `x-typesafe-request-id` header. SDK model
discovery reports an empty `release_date` because Splash records none for a model.
The official SDK is a client only, not a server dependency. API compatibility does
not imply Jev weights, accuracy, proprietary confidence semantics or calibration.

These are local model scores, not calibrated confidence. Probabilities are a
softmax over the declared answer slots. Choice/score `confidence` is normalized
entropy concentration, `1 - H(p) / log(K)`, not an estimate of correctness.
Score answers are probability-weighted level indices. Measure accuracy and
calibrate on representative held-out data before using decision thresholds.

### Status, readiness and metrics

`GET /status` returns instance identity and the effective context limit as JSON.
Proxy consumers can use these fields; additional fields may be added:

| Field | Meaning |
| --- | --- |
| `requests.submitted`, `completed`, `cancelled`, `failed` | Native request counters since engine start |
| `memory_actual.current_bytes`, `peak_bytes` | Metal allocations, not process RSS |
| `weights.idle_release_seconds`, `released`, `restores` | `--idle-release` in seconds (`null` for `off`), whether the weights' memory is released now, and the times it was restored for a request since engine start |
| `ane_ffn.state`, `share`, `minimum_rows`, `reason` | The prefill FFN's [Neural Engine split](#neural-engine-prefill): `split` while it serves, at `share` of the FFN's channels over chunks of `minimum_rows` rows or more; `off` when the start left the GPU alone (`share` and `minimum_rows` 0); `stopped` once it stopped while serving, until the engine restarts. `reason` is the start's outcome, or why the split stopped |
| `ane_ffn.split_commands`, `evaluations`, `ane_ms`, `reruns` | Prefill commands the split ran since engine start, the Neural Engine's evaluations in them and its milliseconds over them (each evaluation's from the GPU's signal to start it to its report); and the chunks the GPU ran again alone after the split's work for them failed |
| `model_timing.prefill.last_gpu_ms`, `total_gpu_ms` | GPU time of the last prefill command, and of all since engine start, warmup included: each from its first Metal command buffer's start to its last one's end, so with the Neural Engine split it spans the GPU's waits on the Neural Engine; a chunk run again adds the rerun's |
| `metrics.decode_tokens_per_second` | Aggregate native decode throughput, not a request's end-to-end rate |
| `metrics.decode_wall_ms` | Total wall time of the decode commands on the GPU, from submission to completion |
| `metrics.decode_cycle_ms` | Total engine time of the decode commands, each from the previous command's completion (or its plan after idleness) to its own: the GPU command plus the host work between commands |
| `loop.max_tick_ms` | Longest control pass and engine step of the native loop. A reader thread keeps reading requests meanwhile, so a request frame whose write stalls 5 s after it started fails the engine only when the process stopped reading; while requests are pending the server asks for status every 10 s and fails an engine whose loop does not answer within 30 s. |
| `maximum_context_tokens` | Declared context limit; available memory may limit admission |
| `vision`, `input_modalities` | Whether image and PDF input is accepted; `false` and `["text"]` after `--language-only` |
| `chat_template.later_system` | `native`, `patched` or `unsupported`: how system messages after the first render (per name for named templates) |
| `transport.recovering`, `transport.stopped`, `transport.error` | The engine is restarting, or Splash stopped restarting it after repeated failures; `error` names its failure, the last failed restart, or why restarts stopped |

`GET /ready` is 200 while the engine's own `ready` is true. While the engine
loop is busy, `/ready` keeps its last answer until the loop has left status
requests unanswered for 30 s; a loop that leaves a status refresh unanswered
that long fails the engine, which then restarts. `/ready` stays healthy under
warning pressure, and reports 503 while macOS reports critical pressure;
requests already running continue, and one that needs more memory is suspended
until the pressure lifts.

`GET /metrics` exports in Prometheus text format the `requests`,
`memory_actual` and `metrics` fields above, the status's scheduler, KV, cache,
admission and image counts, and readiness and memory pressure; the table's
other fields are in `/status` alone.
`splash_kv_free_allocated_pages` counts free pages of allocated extents, not
remaining capacity; memory headroom is `splash_memory_headroom_bytes`. Consumers
should tolerate missing native fields while the engine is unavailable, and
counter resets after an engine restart. A proxy that displays per-request token
counts must read them from the responses' usage
([request fields](#request-fields)).

`/metrics` also exports fixed latency histograms in seconds, with a bounded
set of stages in `/status.latency`. HTTP duration includes body upload and
response writing for admitted API requests. Preparation, queue, template,
tokenization, output grammar preparation and image preparation are measured
separately; preparation includes its nested stages. Tokenization covers the
encoding call, including reuse when available. Histogram buckets are cumulative
and labeled by upper bound. HTTP TTFT (`http_ttft`) starts before upload and
ends at the first native token event; native TTFT (`metrics.ttft_ms`,
per-request `request_latency.ttft_ms`) starts when the engine receives the
request. Output intervals are between native token events, which can contain
multiple speculative tokens; they are not per-token latency. Native queue
timing is recorded from successful completions. These histograms live with the
HTTP process and survive a native engine restart.

For slow tool-bearing requests, the `latency` section of `/status` separates
preparation, tokenization, grammar preparation, native queueing and HTTP TTFT.
Grammar preparation includes construction, compilation/cache lookup and
per-request cloning; it does not include generation-time masks. The
`grammar_cache` counters show whether compiled output grammars are reused. Tool
definitions still contribute tokens to the prompt; saving their JSON alone
cannot avoid model prefill.

Exact-prefix caching reuses model work while the server runs, and across
restarts with `--persistent-cache` ([persistent cache](#persistent-cache)).
Text requests also reuse tokenized history at literal message-end boundaries
when the tokenizer supports independent encoding there. This process-local cache
retains at most four prefixes and 8 MiB of text/token storage; it falls back to
full encoding for other tokenizer pipelines. `/status.tokenizer_cache` reports
its usage. It does not alter prompt text, token IDs or the GPU KV cache.

### Errors

An error is answered as the API of its path answers it, whatever the method
and whether the request was read past its request line. OpenAI's paths, and
every path no other API has, answer `{"error": {"message", "type", "code"}}`,
with `type` `server_error` for a 5xx status and `invalid_request_error`
otherwise.
`/v1/messages` and the paths under it answer as Anthropic's API does,
`{"type": "error", "error": {"type", "message"}}`, with Anthropic's type for the
status, such as `overloaded_error` for 503. `/v1/systemone` answers invalid
requests with 422 `detail` arrays and every 503 with 529. Every 503, and every
529, carries `Retry-After: 1`. A connection the server refuses before reading
any of its request, as when every connection slot is taken
([request limits](#request-limits-and-timeouts)), gets one fixed 503 whatever
its path, `/v1/systemone`'s included: OpenAI's `server_error` with code
`frontend_overloaded`. The codes a client may act on:

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request_error` | A malformed request, or a field Splash cannot honor; the message names it. |
| 400 | `context_length_exceeded` | The prompt, or the prompt and the requested output, exceed the context limit. |
| 400 | `capacity_exhausted` | The request cannot fit in memory even alone; retrying fails the same way. |
| 401 | `authentication_error` | The API key is missing or wrong. |
| 403 | `forbidden` | A Host or Origin the server does not serve; the message names the flag that admits it. |
| 404 | `model_not_found` | `model` is neither the served ID nor an alias. |
| 404 | `previous_response_not_found` | The store no longer holds a Responses `previous_response_id`. |
| 408 | `request_timeout` | The upload stalled or ran past its time. |
| 413 | `request_too_large` | The body exceeds `--max-request-size`. |
| 415 | `invalid_request_error` | The body is not `application/json`, or is compressed. |
| 500 | `engine_failed` | The engine failed repeatedly and Splash stopped restarting it ([engine restarts](#engine-restarts-and-sleep)). |
| 500 | `model_result_invalid` | The model produced a non-finite result for this request; others continue ([failures](#failures-and-crash-traces)). |
| 500 | `internal_server_error` | A fault in the server, not in the request; the console names the error's type and the server line it arose at. |
| 501 | `not_implemented` | The request's method is not one the server has. |
| 503 | `engine_recovering`, `runtime_unavailable` | The engine is restarting; retry. |
| 503 | `server_shutdown` | The server is stopping: requests that arrive or are still running get it. |
| 503 | `frontend_overloaded` | The server holds `--queue-size` requests, every connection slot holds a request ([request limits](#request-limits-and-timeouts)), or a shared input budget or request preparation is full; retry. |
| 503 | `resource_timeout` | The request waited for memory and got none in time; retry, or free memory. |
| 503 | `mask_timeout` | The server left the request's token mask unanswered for 5 s; retry. |
| 503 | `runtime_busy` | The server could not hand the request's token mask to the engine: its mask queue was full or the engine connection failed; retry. |
| 504 | `request_timeout` | The request's `timeout`, or `--request-timeout`, ran out. |
| 505 | `http_version_not_supported` | The request is not HTTP/1.0 or HTTP/1.1. |

## Models

### Upstream model loading

A target is identified by its own configuration: an MLX `config.json`, or the
one `gguf.model_config` derives from the selected GGUF's header, read with a
few HTTP range requests. Before any weight file is downloaded, the installer
runs the engine's own check on it, `model-check` (`build/splash model-check`
in a checkout, `engine/splash model-check` in a packaged installation), which
holds it to the rules every start applies to the installation
(`inspectSourceConfiguration` in `ModelDescriptor.mm`) and names the family
it describes. For a GGUF the check also takes the scalar metadata of its
header, which the installer copies (`gguf.scalar_metadata`), and holds it to
the rules every start's planner holds the file to (`gguf::requireMetadata`):
its sizes, RoPE base, dimensions and scaling, RMS epsilon, GDN sizes, FFN
widths and experts. The target's text configuration is checked before its
vision tower (a GGUF's projector, an MLX target's processor and the shards
holding its tower) and the tower's configuration, so a model of no family
Splash serves is told so whatever its tower, and the check runs again with the
draft's `config.json` once the family's draft is chosen.
A model the engine refuses is refused with its reason, and so is a GGUF with
tensors of types the loader cannot read (`gguf.require_loadable`, [GGUF
targets](#gguf-targets)); no weight file is downloaded for either. Tensor
shapes, and an MLX vision tower's weights, which must not be quantized
(`VisionLoader.cpp`), are checked only when the model loads.
`families.FAMILIES` names the draft trained for each family; repository names
and model-card `base_model` fields play no part. An MLX target must declare
affine 4-bit, group-64 `quantization` in `config.json`, a MoE's router and
shared-expert gate 8-bit. Every start checks the configuration again. Remote
Python code is not loaded.

```bash
splash serve --model mlx-community/Qwen3.6-35B-A3B-4bit
splash serve --model unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M
splash serve --model mlx-community/Qwen3.8-27B-4bit --language-only
```

A model ID with `--revision`, `--language-only` or `--draft-model` is a
separate installation from the same ID without them.

### GGUF targets

`--model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` selects the repository's
root-level `.gguf` file (a lower-case extension, as the native loader requires)
for that variant: the file named with the model name its GGUFs share, then
`-UD-Q4_K_M`, or else the only one whose name ends in `-UD-Q4_K_M`
(`upstream.select_gguf`). Before any weight download, its header must list
every tensor the loader reads with a type it accepts for that tensor
(`gguf.loaded_tensors`, checked by `gguf.require_loadable`; a test holds its
quantized types to `runtime/metal/abi/QuantFormat.h`). The native loader checks again
and lists every unsupported tensor in one error:

- linears and experts: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q4_1,
  IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL,
  MXFP4 or PQ2_0;
- token embeddings: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q4_1, IQ3_S,
  IQ4_NL or IQ4_XS, every type llama-quantize gives a token table by default
  in a file whose linears load, and Prism's PQ2_0;
- norms, the MoE router and shared-expert scalar gate, and the GDN
  convolution, decay and time-step bias: F32;
- GDN alpha and beta: both of one type, any of the linears' formats (one
  segment of the GDN input projection), F32 or BF16, which loading widens to
  the F32 values it equals.

Of Unsloth's files in September 2026 that covers every file of Qwen3.8-27B and
Qwen3.6-35B-A3B, from UD-IQ1_S up, but UD-Q8_K_XL and BF16, whose BF16 tensors
need kernels that do not exist yet. llama-quantize's default type selection
stores alpha and beta in the file type's format (Q4_K in a Q4_K_M), as in
lmstudio-community's files, so those load too. A format's image takes the bits
per weight of its GGUF blocks, but for Q3_K's and Q6_K's padded meta units (1/16
bit more) and IQ3_S's chunk words (4.06 bits for its 3.44).

### PQ2_0

PQ2_0 is Prism ML's type 142, `block_pq2_0` of PrismML-Eng/llama.cpp, which
upstream GGML does not define: 2-bit codes q worth d (q - 1) with one half d per
128 weights.

Prism ML's GGUFs, such as `prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0`, store
every projection for rotated inputs: the `prism.hadamard.*` metadata names the
tensors whose weights multiply H (D x), H the normalized Walsh-Hadamard
transform of each block of 1024 inputs and D an explicit sign per input, and the
token table, whose rows are stored as H (D e). The engine runs that one form, on
dense targets whose rotation names exactly the tensors the planner repacks
(every quantized projection and the head, and alpha/beta when quantized), a
PQ2_0 token table, and GDN value heads in grouped order (the installer screens
the parameters, `GgufFile` and the planner check the rest).

### Drafts

Each family names the repository of the DFlash2 checkpoint trained for it
(`families.FAMILIES`), which holds it as the release publishes it:
`config.json` and BF16 `model.safetensors` (or the shards its index names).
Installation downloads only those files and follows the repository's default
branch as it follows the target's ([Revisions](#revisions)); `--draft-model`
accepts another repository, followed the same way, or a local directory that
holds them. A checkpoint is installed only when the engine accepts its
configuration with the target's (its `model-check`), so a draft of another
architecture never replaces one that loads. Native loading validates the
configuration against the target again and loads the draft like a target
([Weight loading](#weight-loading)):
`DraftCheckpointLoader` (`DraftCheckpoint.cpp`) plans the images of a Splash
package's draft files, `layer-<N>.bin` and `model.bin`, and
`AffinePreparation` quantizes each projection to 4 bits in groups of 64 as
MLX's affine quantization rounds it and copies every other tensor as stored.
For both families the images are byte for byte the Q4 drafts of the Splash
packages.

### Vision

Vision comes from the target repository: MLX's `vision_tower.*` tensors,
linking only `config.json` and the shards holding them, or the GGUF
repository's root projector, a GGUF whose name holds `mmproj` (as
`mmproj-BF16.gguf` or `MODEL-mmproj-BF16.gguf`), chosen by its header: a `clip`
projector whose weights are BF16, or F32; BF16 is preferred. F16 has a narrower
exponent than BF16, so an F16 projector has already rounded small weights and
is not used. The processor configuration (MLX `preprocessor_config.json`, the
GGUF's `clip.vision` metadata) must describe the one preprocessing Splash
implements (`server/images.py`); it is checked before any weight download and
not installed.

Both sources are written into an image laid out as a package's
`vision/model.bin`, which the one BF16 vision operator reads: BF16 tensors are
copied, and F32 or F16 tensors are converted under the exact-BF16 rule of
[weight loading](#weight-loading).
Unsloth's mmproj stores its 1-D tensors, patch embedding and position table as
F32, all of them BF16-exact, and loads byte-identical to the package's file.
Quantized MLX towers, deepstack projectors and mmproj tensors the tower does not
use are rejected.

`--language-only` links and loads no vision weights and removes them from
memory accounting. It skips a GGUF's mmproj download; MLX vision tensors share
shards with the language model, which download in full. The native Ready event
announces vision only when the model loaded it. Without it, image and PDF input
fails with a 400 naming the modality. Every API shape converts its media to
image and file parts, and message normalization, the one place that accepts or
rejects them, checks before any image is decoded or PDF rendered, in user
turns, tool results and stored Responses history alike. `/status` and
`/v1/models` report `vision: false` and `input_modalities: ["text"]`, and the
launchers configure OpenCode, Hermes and Pi without attachments.

### Tokenizer and chat templates

An MLX target's configuration, tokenizer files and chat template come from its
resolved snapshot. For GGUF, `install/gguf.py` reads the selected file's
metadata without mapping or decoding weight tensors. Vocabulary IDs, BPE merge
ranks, control/user-defined token types, BOS/EOS/padding IDs and template text
come from that file; the supported `gpt2/qwen35` profile supplies the NFC and
byte-level pre-tokenization algorithms. Unknown profiles, malformed metadata
and automatic BOS/EOS insertion are rejected, with no cross-repository
fallback; GGUF sidecar tokenizer/config files do not override embedded metadata.
Model geometry is translated from the same metadata, subtracting any declared
MTP layers from the layer count. Only the header is read before the download.
The tokenizer and configuration are derived from the downloaded file once and
cached under `models/.metadata`, keyed by the size and digest of each source
GGUF, the SHA-256 of `gguf.py` and the `tokenizers` version; publication is
atomic and entries are hash-checked on use.

Request preparation merges the leading system and developer messages into one
system message, joined by a blank line: Responses instructions and developer
items, or an Anthropic `system` and a leading system message. A system message
after that, as agent clients send when they change instructions during a
conversation, renders where it occurs as a system turn in the template's own
markup.

Responses and Messages history replay an assistant turn as one message whose
text precedes its calls, the order the chat template renders, even where the
response wrote text after a call.

`server/chat_templates.py` probes each of the tokenizer's templates, including
each named variant such as `tool_use`, once at startup, right after the
tokenizer is validated and before the native runtime starts, so a tokenizer
without a template Splash can serve stops startup before any weights load. The
probe renders a canary conversation whose later system message carries a
marker. Startup logs the outcome (`Chat template · ...`), and `/status` reports
it as `chat_template.later_system`:

- `native`: the marker renders in place; the template is used unchanged.
- `patched`: the template rejects the message (the official Qwen templates
  raise) or drops it (Unsloth's Qwen3.6 GGUF template skips it). Jinja's own
  parser finds the construct responsible: the `raise_exception` in the message
  loop's system branch, or the loop condition that excludes system messages.
  The patch renders the message there with the block the template gives a
  leading system message, and is kept only if ordinary conversations (with and
  without tools, every reasoning effort from `none` to `max`, preserved
  thinking, tool calls and results, images), each rendered as requests render
  it, are byte-identical and the canary renders in place.
- `unsupported`: the template renders the message out of place, has no single
  such construct, renders something before its system block (such as a BOS
  token), or its patch failed a probe. A request with a later system message
  fails with a 400 instead of losing it.

Every request, including image placeholder, token-count and judgment
(`/v1/judgments`, `/v1/systemone`) rendering, uses the template chosen at
startup; tokenizer files and the tokenizer object are unchanged. The probe's
upstream fixtures are in `dev/tests/fixtures/chat_templates/`.

### Legacy Splash packages

Splash packages, such as `incoai/Qwen3.8-27B-Splash`, are the prebuilt format
that predates upstream loading, and `--model` still accepts them. They contain
`manifest.json`, the `target/`, `draft/` and `vision/` weight files and
`tokenizer/`; the manifest lists artifact paths, sizes and SHA-256 hashes.
Qwen3.8-27B packages use schema 3 / `splash-packed-q4`, Qwen3.6-35B-A3B
packages schema 4 / `splash-packed-q4-moe`. Compatible community fine-tunes may
use any nonempty manifest model name. Native loading validates geometry, tensor
sizes, binary headers, tokenizer and target/draft compatibility, and reads the
weight files into memory as they are. `install/legacy.py` installs a package
as a selection link to its verified Hub snapshot, pinned like an assembly's
sources. An installed package starts without a Hub request. A package has no
variants, so a `:VARIANT` suffix is rejected, and `--revision`,
`--language-only` and `--draft-model` require an upstream model ID.

## Internals

### Code and API boundaries

- `server/`: OpenAI Chat/Completions/Responses, Anthropic Messages/count_tokens,
  typed judgments, templates, streaming and input processing. No client-version
  branches.
- `runtime/engine/`: scheduling, memory admission and reusable request state.
- `runtime/model/`: target/draft execution and vision.
- `runtime/ops/` and `runtime/metal/`: operators and Metal kernels.
- `runtime/ane/`: the Neural Engine client, on the private AppleNeuralEngine
  framework, and the handoff of a Metal command's work to it.
- `install/`: launcher, client configuration and model installation.
- `dev/`: maintained tests, benchmarks and build/release tools.

Within `server/`, `server.py` owns HTTP routing and startup,
`connections.py` the HTTP ingress bounds and `errors.py` each API's error
shape; `frontend.py` prepares
requests and history; `backend.py` owns native request lifecycles. `judgments.py`
owns finite-choice prompts, validation and typed answer math. `output.py` parses
generated text as it arrives, one parser serving streamed and complete
responses alike, and `constraints.py` compiles token constraints. Messages and
Responses build streamed and complete responses from the same block sequence;
the one difference is a call cut by the token limit, which a complete Messages
response leaves out. `make architecture-check` prevents lower layers from
importing the HTTP entry module, and keeps one import style in `server/` and
`install/`: a module imports its package's modules relatively. The server
runs as `python -m server.server`; `install/launcher.py`, `install/models.py`
and `install/catalog.py`, which run as scripts, import their siblings through
a PEP 366 header. `serve_options.py` defines the options
`splash serve` shares with the server once, each with its check, default,
help and help group, and how the launcher passes it on; it imports only the
standard library, since the launcher parses them before `.venv` exists.

The values the native runtime holds constant (the Metal command timeout, the
KV tier's transfers, the input queue's bound, the latency window, the
write-behind budget, the prefill checkpoint interval and the resource wait),
its clocks and its live host-memory estimate are parameters of the component
that holds them: production leaves them at their defaults and tests set them.
`make architecture-check` keeps production from measuring durations on the
standard library's steady clock or its timed waits, which count sleep: the
runtime measures them on `AwakeClock` (`runtime/AwakeClock.hpp`), and
wall-clock instants on `system_clock`.

### Installation

`install/upstream.py` installs a model from its upstream repository: it
inspects the target, pairs the draft trained for it and decides when to follow
the Hub. The result is an assembly, a local directory of links to the sources'
Hub snapshots, published atomically. The other installer modules each own one
part: `hub.py` the sources, the Hub cache and its pins; `assembly.py` the
assembly layout, its build, verification and garbage collection, and the
metadata derived from a GGUF; `families.py` each family's draft repository;
`legacy.py` Splash packages; and `models.py` model IDs, selections, the
installation lock, the command line (`install/models.py --model ID
prepare|verify|link`, where `link` prints the selection link) and running the
engine's checks. Since the installer checks a model with the engine, a source
checkout builds the engine before it installs, as `make install` and the first
`./splash serve` do, and a packaged install ships it. The assembly's
`model.json` records the resolved sources and selected formats. The native
loader reads it, and `splash serve`, `test-http-real` and the HTTP regression
benchmark hold it while they run, so a concurrent installation cannot collect
the assembly they serve. It is local installation metadata, not a file model
publishers supply.

Installation resolves each source's revision to a commit once, downloads by
that commit and records it in `model.json`, so a repository update cannot mix
files from different revisions. It pins those snapshots in the Hub cache
(`refs/splash/<installation>/<commit>`), so pruning the cache cannot remove files
an installed model links. Publishing a new assembly retires the installation's
other pins, in the repositories it links and in those its predecessor linked,
so pruning can free what no installation links any more.

`hub.Repository.resolve` alone decides whether a start asks the Hub
([Revisions](#revisions)), each request within `hub.HUB_TIMEOUT`:

- The installed commits: the assembly's links, sizes and times are checked and
  it starts. It is re-assembled first when the draft's repository moved or
  this release changed the GGUF metadata adapter; if the new draft cannot be
  fetched or is not the family's ([Drafts](#drafts)), the installed one is
  kept.
- A new commit: only changed files are downloaded, and the new assembly
  replaces the installed one atomically once published.
- No answer, or a new commit that cannot be installed: the installed assembly
  starts, with one line naming the Hub's reason or, on stderr, the
  installation attempt that failed.
- A 40-hex `--revision`, or `--offline`: a verified installation starts
  without a request.

A missing assembly, or one that no longer verifies, is built again. Without the
Hub it is built from a cached snapshot, of the commit the `--revision` names,
or else the one the installation recorded or pinned, or one the Hub cache
records for the branch, never of another revision. Only files downloaded before
are available, which is enough to rebuild a damaged or deleted assembly; a new
selection needs the Hub once for its draft's default branch. The installer
never rewrites upstream files.

### Weight loading

Every start writes a model's target, draft and vision tensors into weight
images in memory, in the layouts the kernels read: an MLX target, the DFlash2
draft and any vision tower in the layouts of a Splash package's files, which run
the same kernels, and a GGUF target in the `MDGG0001` layout of the GGUF
kernels. Each source adapter is a loader, which validates the source's metadata
and plans its images, and a writer: `AffineTargetLoader` (`AffineTarget.cpp`)
and `AffinePreparation` for an MLX target, `DraftCheckpointLoader`
(`DraftCheckpoint.cpp`) and `AffinePreparation` for the draft,
`GgufTargetLoader` (`GgufTarget.cpp`, planned by `GgufImage.cpp`) and
`GgufPreparation` for a GGUF target, `VisionLoader` and `VisionPreparation` for
an MLX or GGUF vision tower. A package's files are read as they are
(`packageImage`). `AffinePreparation` reorders an MLX target's codes, scales and
biases into 256-row tiles without requantization, quantizes the draft's BF16
projections into the same tiles ([Drafts](#drafts)) and computes GDN decay as
`float(-exp(double(A_log)))`, which may differ by one float ULP in this small
vector from packages produced with MLX's float exponential. `GgufPreparation`
repacks GGUF blocks ([GGUF targets](#gguf-targets)).

Loading never rounds a target or vision weight, and rounds the draft's
projections only as the packages' drafts are rounded. A tensor it converts to
BF16 (vision tensors stored as F32 or F16, a GGUF's convolution taps and
time-step bias) must be exactly representable in BF16; otherwise loading fails,
naming the tensor and, for a vision tensor, its file. The hashes of the images
written from the test fixtures, in
`dev/tests/fixtures/weight-goldens/goldens.json`, fail the tests on any change
of their bytes; the README beside it gives the procedure for an intended
change.

`WeightImages` (`WeightImages.hpp`) holds a model's images, each in a Metal
buffer of its own that the weights are views of. Before any image is written,
the factory (`ModelFactory.cpp`) constructs the vision tower's loader
(`planVisionLoader`), the draft's and the target's, so every source's metadata
is checked first. A writer writes every byte of its image, zeroing alignment
and padding, as a new buffer's contents are undefined. Writers read their
sources uncached (`F_NOCACHE`), so the page cache keeps no second copy of the
model, on a thread per core (`parallelFor`); a thread stages at most 4 MiB of a
source at a time, and a GGUF image the rows of one repack, at most 32 MiB
(`kGgufRepackStagingBytes`), for the `gguf_repack` kernel, which writes the
planes into the image. Once an image is written, its writer checks that no
source file it read was written since it was opened
(`WeightSource::checkUnchanged`), so a source changed in place fails the load
instead of mixing two versions. A start logs `Weights loaded in N s.`; loading
reads the sources at about the speed of the disk that holds them
([upstream loading](dev/benchmarks/upstream-loading.md#loading-time)).

At load time the engine validates the GGUF metadata, including the rotary
embedding and norm epsilon the kernels assume (`rope.freq_base`,
`rope.dimension_count`, `attention.layer_norm_rms_epsilon`, and no
`rope.scaling.type` but `none`), and plans the `MDGG0001` layout.
`GgufPreparation` stages rows in image order on the CPU within the
staging bound, splitting rows wider than it into column chunks, and runs the
`gguf_repack` kernel, which writes their planes into the image. Embeddings and
F32 sections are copied in bounded steps.

Each image's record (`WeightFileRecord`) names what it was written from: the
SHA-256 of the record of every source file's digest, an assembly's `model.json`
or a package's `manifest.json`, which the installer verifies at every start
(`ModelDescriptor::sourceIdentity`). The weight manifest fingerprints `/status`
reports (`loaded_model_layout_sha256`, `target_model_sha256`) cover it, so
models of one layout from different sources never share a fingerprint.

Runtime admission counts the images exactly once (`modelWeightBytes`, which
`tune-kernels` and the runtime oracle use too). Before loading, startup refuses
a model whose images, with the pipeline and runtime reserves, one lane's state,
the KV runway and any disk tier state staging, exceed the hard budget, so a
model that can never fit is not loaded.

When the idle release passes without a request, the memory control between
commands releases the images' memory (`WeightImages::release`): their buffers
are freed, the weights' views stay the same handles, and a command that binds
released memory fails (`MetalBackend::releaseMemory`). The next request waits
while the same writers write the images again (`WeightImages::restore`), an
image per tick so that the loop keeps answering status and cancellations; this
takes about as long as the load at startup and logs `Weights restored in N s`. A
restore is not admitted again: the memory plan counted the images at startup,
and nothing else allocates while the engine holds no request. A restore that
fails (an allocation the driver refuses, a read error, a source written in
place) stops the engine, which the server starts again. With the
[Neural Engine split](#neural-engine-prefill), the release unloads its program
after the images, and the restore loads it again in one tick more, after the
last image (`ReleasableMemory`).

`loadQwenTarget` (`QwenTargetLoader.hpp`) reads a target's images
(`QwenTargetFiles`: a package's files, or the images
`AffineTargetLoader` or `GgufTargetLoader` plans) through the format that
stores them. `AffineTargetFormat`, for package and MLX images, reads every
projection, a fused one too, as one affine Q4 tensor and the norms as bf16.
`BlockTargetFormat`, for GGUF images, reads each GGUF tensor as one
block-quantized `QuantizedSegment` (a fused projection's tensors in output
column order), the norms as F32, and keeps the GDN output projection's input
in llama.cpp's tiled value-head order. Both Qwen families share one layout
(`QwenHybridLayout`) and its validator.

Operator plans use each projection's physical layout, `Affine64` or `Block32`,
independently of the source container. `Projection`, `MoeWeights` and
`EmbeddingWeights` (`runtime/ops/Weights.hpp`, `MoE.hpp`) hold either layout
and represent different operator contracts. Arena sizing collects each
projection's actual layout (a GGUF target's block projections beside its
affine draft's) and reserves the vocabulary head only for decode.

### Kernels

Affine Q4 decode (`runtime/ops/Linear.cpp`) runs MPP tiles on Apple10 and later
(the M6 reports family 11 and runs the same rules) and bf16 simdgroup matrix
tiles on Apple9. On Apple10 a projection with at most two N128 tiles per core
runs that tile over two, four or eight K partitions (`LinearTile::Split128`,
`kernels/decode/linear_q4_grid_split.metal`), the most whose split grid still
fits four 256-thread threadgroups per core; the last partition of each tile
adds the fp32 partials in split order, so the sums do not depend on
scheduling. The split count depends on the grid per core, never on the batch
width, so a request's sums are the same alone and batched. Every other
projection keeps the sequential tiles (`dev/benchmarks/device-policy.md`).

Every tensor keeps its stored format: the F32 norm multipliers, GDN decay, the
MoE router and shared-expert scalar gate, and GDN alpha/beta when a file
stores them as F32 stay F32 and run in fp32, as llama.cpp keeps them (Apple10
prefill chunks multiply the router and alpha/beta on the neural accelerator as
three bf16 parts per weight that sum to it exactly, so only fp32 accumulation
rounds). The GDN convolution and time-step bias become bf16 under the exact
rule.

Decode runs one of two kernel families, chosen by GPU family in `runtime/ops/LinearGguf.cpp`. On
Apple9 (M3, M4) the register tile (`LinearTile::GgufRegister`) runs the kernels of
`runtime/metal/kernels/decode/linear_gguf_sgmatrix.metal`, which feed the codes themselves to
bf16 matrix operations with one fp32 epilogue per coefficient group, so every output is the bf16
rounding of its fp32-accumulated sum. They read their activations as the Table16 table
(`kernels/common/gguf_sgmatrix.h`) that the input's producer writes, or
`decode_linear_gguf_prepare` when none did. The formats whose operands those kernels build from
grid lookups, IQ3_XXS, the IQ2 formats and IQ1 (`apple9StagesFormat`), and Q2_K from two lanes
decode faster on the staged tile there, which a projection all of whose segments are in them
takes wherever the tile holds its lanes' rows unpadded. On Apple10 (M5) the staged tile
(`LinearTile::GgufStaged`) runs the kernels of `runtime/metal/kernels/shared/gguf_linear.metal`,
which dequantize each weight once to half in threadgroup memory (`kernels/common/gguf_staged.h`)
for MPP `matmul2d`, the neural accelerator's path, on bf16 activations; a step of three request
lanes runs the 32-row tile over four lanes of storage. Prefill runs the staged kernels on both
families: the 128-row prefill tile (`LinearTile::GgufPrefill`), and the staged tile for chunks of
up to 32 rows. Every projection splits its K across threadgroups by one rule (`decodeSplits`: each
tile's tiers of threadgroups per core and inputs per partition, from measured occupancy, Apple9's
staged tile taking the register tile's) that does not depend on the batch width. The MoE experts
(`runtime/ops/MoE.cpp`) run the same numerics per family over the grouped rows: the register form
in `linear_gguf_sgmatrix.metal`, the staged one in `kernels/shared/moe_gguf.metal`, which Apple9
takes for experts mostly in the formats it stages (`MoeShape::expertFormat`). The float router and
alpha/beta projections run in `kernels/shared/gguf_float.metal`, and the token rows are gathered
by one template in `kernels/shared/embedding.metal`. These plans are fixed rules of GPU family,
core count, shape and format.

A rotated projection rotates its input once into `LinearScratch::rotated`
(`gguf_rotate`, in fp32 and rounded once to bf16) before its quantized segments,
whose kernels are the format's, while float segments read the input as it is;
the register tile prepares its table from the rotated rows, so the input's
producer writes plain rows. The token table gathers each row through the inverse
(`gguf_embed_rotated_pq20`).

A GGUF kernel of one quantized tensor names its epilogue last: `a` none, `r` residual, `g` the
up pass with the silu gate. The staged ones are `gguf_decode_<format>_m<rows>_<e>` and
`gguf_prefill_<format>_<e>` (and `gguf_prefill_<format>_r_leading_inputs` over a view of the
leading inputs of wider rows), the register ones `gguf_decode_sg_<format>_l<lanes>_<e>`, and the
experts `moe_expert_gguf_m<rows>_<e>` and `moe_expert_gguf_sg_<e>`; the fused projections run
`gguf_decode_fused_m<rows>` and `gguf_decode_sg_fused_l<lanes>`. The norm, GDN and
attention-gate variants that also write a register kernel's input table carry `table64` (the
affine Q4 kernel's) or `table16` (the GGUF one's) in their names. The epilogue kinds of both GGUF
families are in `kernels/common/gguf_tile.h`, the SiLU and sigmoid every kernel shares in
`kernels/common/activation.h`, and the MMA helpers every register kernel uses, affine, GGUF or
fp32, in `kernels/common/sgmatrix.h`.

The ABIs are in `runtime/metal/abi/Gguf.h`, which also defines the tile geometry the kernels
and `LinearGguf.cpp` share, and `MoE.h`; the image formats in
`runtime/metal/abi/QuantFormat.h`, their decoding in `runtime/metal/kernels/common/quant_formats.h`
and the decode-only value tables in `runtime/metal/abi/QuantTables.h`, which no image byte
depends on; weight loading's repack ABI is `runtime/metal/abi/GgufRepack.h`. The tables are
llama.cpp's and the decoding follows its Metal kernels: both keep llama.cpp's MIT notice in
`THIRD_PARTY_NOTICES`, which the release archive ships.

The tests' CPU reference (`dev/tests/engine/GgufFormatReference.hpp`) must reproduce the golden
hashes of upstream GGML's dequantization (llama.cpp 7ab4ee7; for PQ2_0, which upstream lacks,
PrismML-Eng/llama.cpp 01ae597) in `gguf-reference`, and `gguf-planner` checks the planner's
plans; both run in `make test-engine-cpu`.
`make test-engine-metal` runs `gguf-preparation`, which checks every format's planes, as the
production executor and its `gguf_repack` kernel write them, bitwise against the reference,
the alpha/beta, norm, convolution and router bytes they write and the golden images; then
`gguf-dequant`, the staged tile's dequantizer, built with the production Metal flags, against
the half rounding of every reference weight; `gguf-rotation`, `gguf_rotate` and the rotated
PQ2_0 token gather bitwise against the fp32 butterflies and within one bf16 step of fp64;
`gguf-projection`, every GGUF projection through
`ops::Linear` with each tile forced, so both decode tiles run on every GPU, at one to four
lanes, every K split and epilogue, fused segments, every gate/up format pair and the prefill
tiles, each output inside the fp64 bound of `GgufFormatReference.hpp`; and `gguf-moe`: the float
projections on both float tiles and the MoE layer on every GGUF plan, the staged 8- and 32-row
tiles and the Apple9 register tile whatever GPU runs it, in every format, against fp64. The
goldens and how to regenerate them are in `dev/tests/fixtures/weight-goldens/`; with
`SPLASH_GGML_ORACLE=<libggml-base.dylib>`, `gguf-reference` also compares the reference with
GGML directly and prints GGML's hashes.

Two benchmark tools repeat the measurements behind the GGUF split tiers and MoE plans, with the
weights DRAM-cold. `make benchmark-gguf-projection GGUF_PROJECTION_ARGS='q4k 5120 8192'` times one
projection (up to three fused formats and widths, then `K` and an optional epilogue) on both
decode tiles at one to four lanes and every K split, and marks the device policy's pick, or, with
a trailing `prefill=R[,R...]` after the epilogue and the round count, one format's 128-row
prefill tile at each chunk of `R` rows (more than 32);
`make benchmark-gguf-moe` times one MoE layer at the 35B shape, GGUF against affine Q4, on the
device's plans and the other GGUF tile.

Two more time the affine Q4 kernels on synthetic weights, with no model. `make benchmark-decode`
times the N128, N256 and paired N128 decode tiles on eight simdgroups, plain and with the residual
and gate/up epilogues each takes, at 8 to 32 rows, on the 27B's projection shapes, its draft's
context projection and the LM head, over a range of threadgroup counts, and prints each one's time
and effective bandwidth. It leaves out the other decode tiles the device policy chooses: Apple9's
register tile (`LinearTile::Q4Register`), Apple10's `Split128` and paired N256 tiles, and N128 on
four simdgroups at 24 rows. `make benchmark-prefill` times the prefill tiles the device policy
chooses among (N128 and N256 on eight simdgroups, N128 on four), with their residual and gated
epilogues, on the 27B's and 35B's projection shapes at 17 to 2,048 rows: the measurements behind
the Apple9 prefill rule in `runtime/ops/Linear.cpp`.

### Neural Engine prefill

Prefill chunks of a dense target split each layer's FFN by intermediate
channel (`runtime/ops/AneFfn.cpp`), from the least rows startup finds the split
faster at (below). The GPU runs the leading
channels on its prefill kernels, its down projection reading a view of the
leading inputs of down's rows (`Projection::leadingInputs`) through instances
of the residual kernels of their own (`<kernel>_leading_inputs`), which run
the residual kernel's tile on the weight planes advanced past the inputs the
view leaves unread. The Neural Engine runs the rest as one W8A8 program,
Hadamard-rotated int8 activations and per-row int8 weights
(`runtime/ane/Program.mm`): a chunk takes the smallest of
its functions that holds it, one every 128 rows from 512 to 2048, all reading
and writing one set of surfaces sized for 2048 rows. The ANE service holds
memory for a loaded program's intermediate values, which its functions share:
13 programs of one function each held it 13 times (3.25 GB at the M6's share,
against 0.67 GB for the one program). The GPU requantizes the
ANE's weights from the Q4 or GGUF planes one layer ahead into double-buffered
IOSurfaces and adds the ANE's partial down projection to its own. Shared
events order each evaluation between the GPU's packing and that join inside
the one prefill command (`metal::EventStep`); each signal ends a Metal command
buffer, so the queue holds 512. MoE targets, the mixers and decode stay on the
GPU.

Startup decides the split on the loaded model before it allocates the KV cache
(`runtime/engine/AneFfnStartup.cpp`), once it has built the memory governor,
which is the same with the split or without, every step arithmetic until the
memory plan holds a split. `--disable-ane`, a target the split does not take
and a Mac whose private ANE interface does not resolve (`ane::unavailable`,
which compiles nothing) leave the GPU alone; a dense target whose layers it
does not take, or a Mac without the interface, says why, a MoE target says
nothing. The split's buffers are a memory category of their own (`Neural Engine
split`), which comes out of the KV cache, the more of it the more channels the
ANE takes: with `--max-context` the split takes at most the channels whose plan
still holds that context, and if even the fewest do not, the GPU runs alone and
the start warns how much context any split leaves. Without it the engine serves
the automatic context, which while the split may run is what the plan holds
with the split this Mac's calibration of the model takes (below), the GPU
alone's if it found no gain, whatever the start then makes of it: a split that
failed leaves the GPU alone at the same context, and the memory the split would
hold stays with the KV cache. Every start of a model on a Mac, the server's
relaunch of the engine after a failure among them, so serves the same context.
Until a calibration is remembered, a start that fails assumes the largest split
the model could take (all its channel units but one, or the most whose plan
holds any context). The log line names it beside the context without the split
(`context 61,433 tokens (73,721 with --disable-ane)`); `--disable-ane`, a
target the split does not take and a Mac without the interface serve the
latter. The start then logs `Setting up the Neural Engine FFN split (splash
serve --disable-ane keeps the FFN on the GPU)`, so that a fault in the private
client that ended the process would leave the way around it above, and, when it
calibrates (below), times the split as a prefill runs it
(`runtime/ops/AneFfnMeasurement.cpp`): commands of layers through `begin()`,
`add()`, `commit()` and the handoff, each split layer after the GPU alone's FFN
of the same layer, a stand-in for the mixer, which the GPU's layer alone timed
alike takes away again, the median of three. On five FFN layers spread over the
model's depth, with the ANE taking the 512-channel units nearest 0.4 and 0.8 of
them in programs of one 2048-row function, which the ANE service keeps once
compiled, it times the split layer and its GPU part alone, and the GPU's layer
alone, each over commands of the first four full-chunk layers, the fifth
staged. It fits `T(s) = max(A(s), G(s) + u·A(s))`
(`runtime/ops/AneFfnCalibration.cpp`): G the line through the GPU part's
timings; A, the ANE's part, proportional to its channels through the split
layer at 0.8, where the ANE's part takes the longer on every Mac measured; and
u the bandwidth the ANE's part takes from the GPU's beside it, the split
layer's excess over G at 0.4 where G takes the longer there (on the M6 the
GPU's part runs a tenth to a third longer beside the ANE's). It takes the units
whose T, or T a unit either way, is least at its longest, since the timings can
put T's least a unit off: a unit past where the ANE's part takes the longer
costs the ANE's time per unit, a unit short of it the GPU's, and on an M5 Max
and an M5 Pro the ANE's is the longer (1.7 ms a unit of the full-chunk layer
against 0.6 and 1.1 ms), on an M6 the GPU's (1.9 against 1.2 ms). On
Qwen3.8-27B that takes 8 of the 34 units on an M5 Max, 14-15 on an M5 Pro and
25-26 on an M6, the best each Mac's sweeps measured. The split runs if T there
gains 5% on the GPU alone. A start calibrates the split once for a Mac, a
model, a macOS version, an engine build and the most units its memory plan
holds, and remembers what it found (`remembered-*` in the cache directory
below): the units, the least chunk and the GPU alone's layer at 512 and 2048
rows, or that no split gains enough. Later starts take it untimed: they build
the split, check it as below and serve, in 0.6 s on an M5 Max, 0.9 s on an M5
Pro and 1.4 s on an M6. Across many starts over hours of prefill on those three
Macs the timings chose the same units within their noise, while timing every
start cost 3-8 s, and a start whose timings strayed could change the share,
each share compiling a program of its own (half a minute on an M6).

The start then builds the split that serves and checks every function of its
program against the GPU alone (`AneFfn::verify`): at the function's own rows
over two layers, which stage the ANE's weights through both sets, with the
ANE's output filled with NaN first so that rows it leaves unwritten show,
against the leading rows of a full chunk on the GPU alone, which computes every
row alike whatever the chunk's rows. The outputs must be finite and differ from
the GPU's by at most 10% of what the ANE adds over the GPU's part; int8 leaves
2.5-2.8% on Qwen3.8-27B. A start that calibrates then times the GPU's FFN layer
alone at 512 and 2048 rows and the split layer of every function as above, and
splits the chunks from the least rows from which every larger chunk's split
layer takes at most 2% longer than the GPU's at the chunk's rows. Every row
count is checked, since a batch's chunk sums its lanes' rows: an evaluation
costs the ANE its function's rows, which a chunk pads up to, and the functions
do not cost it in proportion to their rows (on an M5 Max 640 rows take 93% of
768's). On an M5 Max with Qwen3.8-27B at 0.24 that is 540-850 rows over
calibrations.

The GPU runs the FFN alone if no share or chunk gains enough, or the ANE or one
of its functions is unavailable or fails the check, as under Metal's validation
layer, whose wrapped shared events the ANE cannot share. The start warns of a
failure, as of a refused `--max-context`, and `--disable-ane` silences both;
only cancellation, an interrupted wait for the ANE's service or a backend that
stopped serving ends the start. Startup logs which (`Neural Engine FFN split at
share ...`). The split's program also holds buffers in the ANE's service,
outside Metal, which only the host's free memory shows: on Qwen3.8-27B about
0.7 GB at the M6's share and 0.2 GB at the M5 Pro's, against 0.5 and 0.26 GB of
the split's own; the governor meets them while serving as it meets other
applications' memory. The first start compiles the ANE programs, about 9 s on
an M5 Max (27 s for calibration's share 0.8) and half a minute on an M6; the
ANE service keeps them under a hash of their source, which stays in
`splash-ane` of the per-user cache directory (`getconf DARWIN_USER_CACHE_DIR`),
so later starts load them, in about 0.1 s, until macOS clears that directory.
Startup gives the service 120 s to compile and load a program, after which the
GPU runs alone: the private client's work for every program runs in turn on one
queue (`runtime/ane/Program.mm`), so work queued behind a service that stopped
answering runs out of its time too. The ANE computes in fp16, whose range
(±65504) bounds a split layer's gate pre-activations and its intermediate
values over each token's input peak; it leaves its output's token scales to the
GPU's join, in fp32. While serving, any failure of the ANE's work stops the
split for the life of the process (`runtime/ane/Handoff.mm`): an evaluation
that fails, cannot start or has not completed 2 s after the GPU raised its
event (Metal fails a command buffer whose wait stays unmet for 5 s), or a value
the join reads that is not finite. The CPU then raises the event to the GPU's
last wait, the chunk runs again on the GPU alone, which computes what it would
have, as does every chunk after it, and the engine logs `Warning · Neural
Engine FFN split stopped (...)`. A split that loses to the GPU alone stops the
same way, after a chunk whose results it keeps (`ops::ane_ffn::Breaker`): over
each 8 commands it ran, the ANE's evaluations, each from the GPU's signal to
start it to its report, took as long as the GPU alone's FFN layers of their
functions' rows, on the line through the timings calibration took; the start's
calibration is then forgotten, so that the next start calibrates again. The GPU
then waits on the ANE at every layer, so the split is slower than the GPU
alone; another process keeping the ANE busy costs the split 2-8% and a minute
of prefill slows the M6's ANE by a third, far short of that. A given share
(`--ane-ffn-share`) takes no timings and never stops for it. The idle release
(`--idle-release`) unloads the program, so that the ANE's service gives its
buffers back while the engine is idle, and keeps the split's own, which the
residency keep-alive unwires (`AneFfn::release`); the next request loads the
program again after the weights (`AneFfn::restore`), in about 0.1 s, and logs
`Neural Engine FFN program reloaded in N s`. A program that does not unload or
load again within 10 s stops the split like a failure while serving, and the
request runs on the GPU alone. A split that stopped unloads its program at the
next idle release once the ANE has reported every evaluation, and keeps it
unloaded. The split's logits differ from the GPU's alone (KL about 1e-4 to 7e-4
on Qwen3.8-27B); `splash serve --disable-ane` keeps the FFN on the GPU, and
`backend-benchmark --ane-ffn-share` runs a given share over chunks of
`--ane-ffn-minimum-rows` rows or more (512 by default), which
`backend_regression` pins to its first round's. `make test-engine-cpu` runs
`ane-ffn-calibration`, the fit, the choice, the least chunk, a calibration's
line of text and the breaker on timings it is given, also under the sanitizers,
and `ane-ffn-startup`, each outcome of a start on a model it plays, a
remembered calibration among them, and the automatic context, the same in each
where the split may run. `make test-engine-metal` runs `ane-ffn`: its kernels
against CPU references for affine Q4 and every GGUF format under shader
validation, then, without it, the split's memory and its output against the GPU
alone, the same output once its program is unloaded and loaded again, a chunk's
rows on the GPU alone as a full chunk's, `verify()` of every function and of a
function the client's instrumented build
(`runtime/ane/ProgramInstrumentation.hpp`) binds to another's procedure, each
fault of an evaluation it injects at each layer, a program that does not load
again and evaluations slower than the GPU alone, which the breaker stops after
8 commands, telling the start to forget its calibration, and the private
client's failures, limits, unload and load, and cache files on a small program.
`handoff` runs the event handoff against an agent the CPU plays in each way it
can fail, and `ane-program-faults` checks that an Objective-C exception and a
changed method signature fail as `std::runtime_error`. `make test-real` runs
the model runtime oracle with the GPU alone and with the split as a start makes
it, which must not be unavailable, and whose automatic context must be what the
plan holds with the split it runs.

### Residency and KV extents

Every buffer the backend allocates belongs to one residency set attached to its
command queue (`MetalBackend::allocateBuffer`): weights, KV extents, state cells
and draft rings, and scratch alike stay wired between requests until the idle
release (`--idle-release`, 10 minutes by default) passes without a command, and
the next command wires them again. `--idle-release off` keeps every buffer
wired, and the weight images allocated, while the engine runs; KV extents return
only through the reclaims below, memory pressure among them. Memory returns to
macOS when the engine releases it, never because macOS compressed or dropped an
idle buffer. macOS page cache, driver allocations and other applications still
affect memory pressure and swap.

KV pages live in extents: ordinary shared Metal buffers of one size per pool,
between half and one and a half times 128 MiB, in which every tensor region of
each attention layer starts 64 KiB-aligned, sized to leave the fewest of the
budget's pages unused (`Layout::extentPagesFor`). The pool allocates an extent
when it needs one of its pages. When it is built it allocates the runway, the
extents of the first 64 pages, which startup warmup runs on; nothing but the
pool allocates or releases an extent. An extent whose last page is free stays
allocated until a reclaim releases it, at once and only between commands: memory
pressure, an admission the budget denies, the publication of a replay point in
use or a junction, or startup cleanup. Kernels reach a page through the GPU
address in its request's page table, so no command binds KV; the residency set
makes extents resident for every command. The host reaches the same memory
(`PageStorage::spans`), which is how the disk tier moves pages. A reclaim
returns free pages before it evicts anything: an empty extent as it is, and the
free pages scattered over the others as soon as they cover the extent that holds
the fewest pages, whose pages the pool copies to them (`KvPool::compactExtent`).
The blocks and requests on those pages follow them, a page a disk transfer reads
or writes stays where it is, and a pass that evicts everything copies only what
is left afterwards. Every extent a pass empties is released, except that a
warning pass, like startup cleanup, keeps one empty extent as the runway the next
request starts from.
`/status` reports under `kv` the pages of allocated extents (`pages_allocated`),
those requests and the cache hold (`pages_active`, `pages_cache`) and those
nothing holds (`pages_free`), the bytes allocated and the bytes of empty extents
(`allocated_bytes`, `reclaimable_bytes`), the extents allocated and released
(`extent_allocations`, `extent_releases`) and those emptied by moving pages
(`extent_compactions`, `pages_moved`), and the longest allocation, release and
emptying of one (`extent_allocate_max_ms`, `extent_release_max_ms`,
`extent_compact_max_ms`); the last includes re-pointing the cached blocks and
requests on the moved pages. The counts and the longest allocation include the
runway allocated at startup, before serving begins; how long a whole pass holds
the loop shows in `loop.max_tick_ms`.

### Admission and eviction

`/status.admission` distinguishes memory and concurrency waits, counts the
requests held back behind one refused memory (`held_behind_refusal`; during
recovery only the suspended ones and those of a strictly higher priority, and
the refused request itself while a pass defers it, even one that recovery does
not try, which still waits for memory) and those waiting for a disk restore
(`restoring`), reports suspended requests, recovery draining and the oldest
current wait age, which for a request holding admission closed runs from when
its wait began. Memory transitions also appear in the console. When macOS runs
short of memory, growth that no request in service needs pauses and the cache
gives memory back, a paced pass at a time, down to one lane's state buffers and
one KV extent. A pass counts only memory that leaves the engine; a cached state
it evicts first refills the lane's state buffers it keeps. A request in service
keeps growing within `--max-memory`, first into cached pages no request holds:
it takes cached KV, and a cached state only where it sits on the KV that goes
next, since a state's buffers give it no page. A new request waits while another
is in service unless it can start from what the engine already holds, and with
none in service it starts. A request whose start was refused memory, under host
pressure or at `--max-memory`, holds back the requests that arrived after it
until it starts, fails or is cancelled, so the lanes that finish leave their
memory to it; a higher priority is not held back. It keeps holding them back
while it waits for a prefix another lane is computing. After a suspension,
suspended requests resume before new work of their priority or below, one at a
time, higher priority first and then in the order they arrived; one refused
memory holds back the suspended requests after it, and their resource waits do
not run out while it does. When every lane is taken, or a new request's start
is refused memory that no copy being written to disk is about to free, the
resident of the lowest priority below it is suspended for it
(`cache.priority_suspensions` in `/status`) and resumes once a lane and its
memory are free. A new request below the priority of a running lane neither
starts nor takes a lane until no lane of that priority runs, since it could not
run before. Critical pressure evicts every unpinned cache entry and stops all
growth.

A request keeps its reusable model state at the last whole 32-token page before
its generation prompt, the text a chat template appends to open the reply: the
next turn may render it differently, so a follow-up resumes from there.

Until the request ends, suspended or not, that replay point is in use, and so is
the KV it restores through. When the last request using it ends, the point
becomes the newest ordinary state; the older points on its chain keep their age.
Cache victims come in three classes: checkpoints, then ordinary states and KV
(first the KV no state restores through, which saves no prefill), then what is
in use. No work displaces anything of a class above its own. Memory for running
requests takes what is in use after everything else. Two kinds of work take
nothing in use. A start that a resident lane holds back waits for that lane. And
when no lane's growth fits, the lane that yields first, such as one just started
beside a decoding lane, pays for the memory with its own suspension; while other
lanes fit, it waits for them, resident. A publication in use takes cached KV and
states in the same order, then the oldest state in use; of the KV it takes only
leaves whose page frees at once, and only while an extent can be emptied; the
extent is released at once, and the snapshot follows. An ordinary publication (a
junction) makes room the same way short of what is in use, and an optional
checkpoint takes only checkpoints; neither drops a state whose write must wait
for the write in flight. A disk copy in use may displace the oldest copy in use,
and ordinary or optional work never displaces anything in use. Nothing in use is
pinned, so running work that needs the memory still takes it once nothing else
is left. A resumed lane that lost its prompt's replay point rebuilds it on the
way; the point its generated history reaches is a disposable checkpoint.
`/status` reports under `state` the replay points unfinished requests hold
(`in_use`, zero when idle) and those evicted all the same (`in_use_evictions`).

Requests sharing a cold prefix can wait for a resident request's planned recovery
point, including one whose prefix is still being restored from disk, then enter
through the ordinary cache restore path. Waiting requests hold no lane or KV
pages and return to ordinary admission when no useful producer remains. Late
arrivals can extend the plan at complete state boundaries.
Higher-priority work does not wait for a lower-priority producer. `/status` exposes
`scheduler.waiting_prefix` separately from resource waits.

Greedy and sampled requests share unconstrained decode batches; each lane keeps
its own sampling policy and RNG, and a greedy lane takes the argmax path in any
batch. Constrained requests use a separate batch for the host mask exchange. A
mask request the server leaves unanswered for 5 s fails that request with the
retryable `mask_timeout` (HTTP 503), so a stalled grammar cannot hold the
batch's command.

Long prefill uses disposable rolling checkpoints every 4096 tokens; none is
planned within one prefill chunk (2048 tokens) of where the request resumes or
of its replay boundary. Each retires when the next one or a junction lands, but
the last stays cached once the replay point is published: a later request that
shares the prompt up to it, such as another conversation with the same long
system prompt, resumes there instead of from the start. It stays only if the
replay state fits beside it; refused memory, the replay state takes its buffers
rather than another lane's checkpoint. With the SSD cache it costs a write as
well: reclaim takes checkpoints first and writes the states it evicts to the
SSD, this one too (187 MiB for 27B), and one published straight to the SSD,
when no RAM slot took it, stays there. The persistent cache's hourly write limit
does not pause these writes, and a full quota makes room for one by dropping the
oldest copy, perhaps an older conversation's restore point.
Contended prefill adapts toward a 500 ms slice, keeping
2048-token chunks for long unopposed work. Without contention, a prompt that can
finish within one 2048-token chunk ends its chunk at its last row instead of
sharing it with a longer prompt. While requests of the same or a higher priority
decode (a lane waiting for its token mask is owed nothing), each slice owes them
decode time, `--decode-share` times its own, before the next slice runs. These
policies do not extend client deadlines. Memory recovery waits are bounded:
after a suspension, new work waits for resident requests only while memory is
still short, and at most for the 30 s resource wait; suspended requests then
resume first, each within its own resource wait. A request of a strictly higher
priority than every suspended request waits for neither. A resource wait's limit
restarts whenever an earlier lane has work in flight: one submitted before the
waiting request, or admitted before the request was first refused memory or
suspended. Such a lane holds memory the request waits for until it finishes.
Other lanes do not extend the wait. Readiness does not guarantee that a
request-sized allocation fits. A request that cannot fit even alone, after every
cached prefix was evicted, fails with 400 `capacity_exhausted`, naming
`--max-memory` and `--max-context`; retrying it fails the same way.

### Disk tier

The tier holds cached request states (GDN cell plus draft ring) and KV pages.
RAM and disk copies share the same block tree and recency order. Restoring a
prefix keeps its disk copy, so its next eviction needs no write while that copy
remains cached. The KV tier takes no Metal memory. The state staging buffer,
one state, is a buffer of the backend's like every other: resident, and set
aside by the memory plan within `--max-memory` when the tier starts. A quota too
small for one state leaves the tier disabled and sets nothing aside. With the
tier off, startup suggests it when the memory plan, within what the host had
available at startup beyond its reserve and the warning margin, holds less than
the context limit (`EngineMemoryPlan::contextTokensWithin`). The estimate is
conservative, since macOS compresses other applications further once the
engine loads.

Writes happen when RAM reclamation selects a victim. States copy through one
staging buffer, freeing their RAM immediately. KV leaves needed by a state
on them or below them are written straight from their extents and released
after the write succeeds. Unneeded tails are dropped without writing, before any
state, together with any disk copies below them. A restored page is read
straight into its extent. Either way the transfer runs on the file's IO worker
beside whatever command the model runs: a cached page is never written by a
command, and no command uses a page before its read has landed. At most 128 KV
pages are in transfer at a time, demotions at most half of them and restores at
most three quarters, since one worker serves both in order and a burst of either
kind must leave the other its share. When the tier takes no more, leaves that
need a write stay until a transfer lands, while leaves whose page frees without
one still go.

A state with no available RAM cache slot can be written directly from its lane.
When every state in RAM is in use and no cached KV is left to take, a replay
point takes the slot of the oldest by writing that one out, and goes
unpublished while the staging buffer is busy.
Rolling checkpoints replace the least recently used copies like any state, so
a suspended request keeps its progress when the quota is full; they retire when
replaced; a prompt's last stays. Matched KV restores start from the root toward
the selected state, in that order as the tier takes them, with the state read
alongside. Cancellation drops unsubmitted, unshared reads; submitted transfers
drain before their buffers can be reused. Restored states remain usable even
when there is no room to promote them into RAM cache. Promotion takes only the
RAM of a state that keeps a disk copy.

Two files share one quota for live slots, one of KV pages and one of states.
With `--persistent-cache` they are `kv.slots` and `state.slots` in the cache's
directory, each beside a file of its slots' records, and stay for the next
start. Otherwise, or when another process holds that directory or it cannot be
used, they are temporary (`$TMPDIR/splash-cache-*`), unlinked as they are
created and gone when the engine exits. A full quota of temporary files
replaces the oldest redundant copy first, then the oldest sole copy, across both
KV and states; sole copies of states in use, and the KV they restore through,
make room only for a copy that is itself in use, and last. A persistent tier
keeps restore points whole instead, since its redundant copies are what the next
start takes back: a full quota drops a KV copy that no state needs first, then
the oldest state copy, redundant or not, after which the copies of its chain
that no other state needs go first in turn; the only copy of a state in use
makes room only for a copy in use, and last. Either way an ordinary state that
finds no other room is dropped. A freed slot returns its quota at once and its
blocks to the volume (`F_PUNCHHOLE`) once its file's IO worker finishes the
transfer it is in, so the two files together occupy about the live slots instead
of each keeping its high-water mark; `/status` reports that as `disk.file_bytes`
beside `disk.used_bytes`. Until the punch the other file can take that quota, so
allocated blocks can briefly exceed `--max-cache-disk` by the slots freed but
not yet punched, bounded by the transfer each file's worker is in: about 2% of a
10G tier in a 25-minute churn run on an M5 Max. A volume sized exactly to the
quota can therefore fill; the write that finds it full stops that file's writes
for the rest of the process, as below, while serving continues. On a volume that
cannot punch holes the files keep the blocks of freed slots, which the runtime
reports once.

Transfers use `pread`/`pwrite` with `F_NOCACHE`, every one an aligned range:
whole 1 MiB chunks of 16 KiB-aligned memory that start at an aligned offset of
the slot (a state's staging buffer and lane buffers) move directly, everything
else through the file's own 1 MiB buffer. A failed write disables further writes
to that file. Failed KV writes retain RAM pages; failed state writes invalidate
the disk copy. A failed read invalidates its cached data, allowing lookup to
fall back to the surviving prefix.

With `--persistent-cache`, each slot's record carries a CRC-32C of its payload;
a copy that fails it on first read is a miss.

`/status` reports the shared quota and KV transfers under `disk`. Its cumulative
`read_bytes` and `written_bytes` count bytes transferred by file IO across KV
and state files, including partial or cancelled transfers. They exclude
filesystem metadata and physical SSD write amplification. State transfers appear
under `state` (`disk_bytes`, `offloads`, `disk_hits`, `disk_promotions`). Cache
counters include:

- `kv_disk_hit_tokens`: tokens restored by completed KV transfers. Shared
  transfers count once, including those completed before cancellation or a
  resource retry.
- `lost_state_misses`: lookups that matched KV where a reusable state used to be.
- `probe_hashed_blocks`: KV pages hashed to rank waiting requests between
  commands. A waiting request's match is kept across passes and extended only
  past what the cache changed.

### Native protocol

The engine receives `ignore_eos` as bit 0 of the request frame's flags word and
`return_progress` as bit 1; the word rejects undefined bits.

Requests carry the score-token IDs and Done events the selected f32 logits; the
server and the engine must speak the same native wire version. Scoring requires
2–255 distinct, in-vocabulary tokens, no images or generation constraints, and a
zero output budget. It may use the full context window because no generated
token needs a reserved position. The final prefill chunk runs the target head
but no sampling policy or DFlash decode. Successful scoring emits no Tokens
event, finishes with Stop, and reports zero decode time. Cancelled requests
carry no logits.

### Failures and crash traces

A non-finite logit row is a per-request failure, not an engine fault: a score
logit, or a token the sampling kernels could only select outside the
vocabulary. The engine reports `model_result_invalid` for that request alone,
before it publishes the failing step's cache state or any output, and the rest
of the batch finishes normally. Prompt chunks that already succeeded keep the
blocks they committed, exactly as they do for a cancelled request. GPU faults
and broken engine invariants stay fatal and still mark the runtime unhealthy.

Request logs omit bodies; full crash traces require explicit
`SPLASH_CRASH_TRACE=1` and can contain private conversation data. With it, a
failed engine's last frames, at most 512 and 16 MiB, are written to
`~/Library/Logs/Splash/crash/`, which keeps the newest four traces within
64 MiB. A frame over 16 MiB, such as a request with large images, is kept only
as a marker with its size and SHA-256 (`omitted_frames` counts them), and a
trace missing engine input that way cannot be replayed.
`python -m server.crash_trace <trace>`, run from the checkout root, replays a
trace.

## Validate

```sh
make check
make install test-real test-http-real MODEL=mlx-community/Qwen3.8-27B-4bit
```

`make check` needs no model weights, but it runs the Metal tests, so it needs a
Mac that Splash supports: it runs the source checks, `check-native-cpu`,
`check-native-metal`, `check-native-build` and the Python checks.
`make check-native-cpu` builds production and runs native CPU tests without a
GPU; `make check-native-metal` requires a supported Metal device and runs the
kernel tests under shader validation, and the Linear pipeline resource check
without it. They include loading small synthetic MLX, GGUF and vision sources
and the GGUF kernels on synthetic tensors. `make check-native-build` builds
every native test, benchmark and tool and runs none. Hosted CI runs the CPU
checks, `check-native-build` and the sanitizers. `make architecture-check`, one
of the source checks, also fails when a workflow under `.github/workflows`
names a Hugging Face token (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or a
`secrets.HF_*` secret): CI installs public models alone and never receives one.

The native tests are in `dev/tests/engine`; each Python test is in the
directory of what it tests: `dev/tests/server`, `dev/tests/install` (launcher,
client configuration and model installation) or `dev/tests/tools`
(`dev/tools`, `dev/benchmarks`, the Makefiles and the real-model harnesses).

The real-model targets run against an installation in this checkout's
`install/models`: run `make install` with the same options first, or in the
same command, as above. They take `MODEL` exactly as `splash serve --model`
does, and `REVISION`, `DRAFT_MODEL` and `LANGUAGE_ONLY=1` as its `--revision`,
`--draft-model` and `--language-only`:

| Target | Runs |
| --- | --- |
| `verify-models` | the installer's restarts without the Hub, `verify --full`, and the record of the weight images (`dev/tools/installer_restarts.py`, `weight-digests`, [Release check](#release-check)) |
| `test-real` | vision parity with the family's fixture in `dev/tests/fixtures/vision-parity/` when the installation serves vision, and the native model runtime oracle |
| `test-http-real` | the HTTP frontend on an isolated server (`dev/tests/smoke_real.py`); with `HTTP_SMOKE_ARGS="--persistent-cache --max-cache-disk 8G"`, instead of that smoke, the [persistent cache](#persistent-cache) across a restart in a fresh cache directory: a clean stop, a restart that takes the restore point back, and a next turn that restores the prompt from disk; `release-check` runs both |
| `test-agent-real` | the five official clients through `splash serve` (`dev/tests/agent_real.py`), in `AGENT_SCENARIO` `complete` (the default) or `smoke` |
| `test-release-real` | the HTTP smoke and all five clients on one `splash serve` |
| `test-performance-real` | the native decode, partial-prefix and short-prompt benchmark, or with `BASELINE` its ABBA comparison with that build (`dev/benchmarks/backend_regression.py`) |
| `release-check` | one model on this Mac ([Release check](#release-check)) |

`test-agent-real` and `test-release-real` first run the tests
`dev/tests/install/test_clients.py` has of the installed OpenCode, Codex and Pi
among the selected clients, which need no model.

`test-agent-real` runs Hermes in a profile of its own in the developer's Hermes
root, `splash-test-<id>`, which moves into the run's folder under
`build/release` when Hermes finishes.

Each phase's record keeps what its requests reused of the cache (`reuse`).
Every request after a phase's first resends the conversation, so a phase that
completed two or more requests and reused no cached prompt token fails. Two
phases are exempt: the cancellation phase (`cancel`), and the reference wave
that compacts the conversation, whose summary request and the request after it
share only the system prompt and tools. Replay points of unfinished requests
evicted during a phase only print a warning.

`benchmark-backend`, `benchmark-decode-profile` and `tune-kernels` take `MODEL`
the same way. The models they are run with, one per family and source format:

| Family | MLX | GGUF | Splash package |
| --- | --- | --- | --- |
| Qwen3.8-27B | `mlx-community/Qwen3.8-27B-4bit` | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | `incoai/Qwen3.8-27B-Splash` |
| Qwen3.6-35B-A3B | `mlx-community/Qwen3.6-35B-A3B-4bit` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` | `incoai/Qwen3.6-35B-A3B-Splash` |

The source formats load differently: an MLX target is written into a package's
layout, a GGUF target into its own layout for the GGUF projection and MoE
kernels, and a package's files are read as they are.

On a 24 GB Mac, `test-agent-real` stops a client's workflow at macOS's warning
memory pressure, which the smaller GGUF variants such a Mac uses can reach under
an agent's load; `SPLASH_TEST_PRESSURE_STOP=4` stops only at critical pressure,
to observe how the engine sheds its cache. The runtime oracle in `test-real` has
no production memory guard: the weights, and then what the runtime
allocates as it runs, must fit in what macOS has available above its reserve,
so it stops, naming what it needs, while other programs hold that memory. With
only desktop applications open, a 24 GB Mac runs it for those variants.

`make check-native-build` builds the affine source oracle and `weight-digests` so
they cannot break unnoticed. No target runs the oracle, as it needs real
models: after `make all build/engine-tests/affine-source-oracle`, pass it
`build/splash.metallib`, an installed MLX model's `target` directory and the
matching installed package to compare every byte of their images.

Compare performance on the same idle Mac with the same model and workload.
`make tune-kernels MODEL=...` measures each projection key of the installed
model on this Mac: the policy default in `runtime/ops` against the tile
configurations that won an earlier run (`dev/tuning/LinearTuning.hpp`). It
prints, per key, the winning configuration with its paired GPU and wall-time
gain, or that the default is kept; it changes no default and saves no profile.
For a GGUF model it measures only the draft's projections, and says so in its
header, since the device policy alone plans block projections
([GGUF targets](#gguf-targets)). Keep generated reports, profiles, local paths
and experiment notes out of the source tree and commits.

### Release check

A release is checked once per source identity, and then on each Apple GPU
family (an Apple9 M3 and an Apple10 M5) against a retained baseline build,
`BASELINE`: a checkout whose `build/` holds `splash`, `splash.metallib`,
`engine-tests/backend-benchmark` and, for a build that loads the weights into
memory (1.2.0 and later), `engine-tests/weight-digests`. `release-check` fails
without it. The baseline must load the model: it is the previous release's
build when that loads the model. The legacy package can always be compared with
1.0.2, as below; Splash 1.0.x loads only Splash packages and has no
`weight-digests`, so the package's weight images are not compared with it. From
a clean checkout:

```sh
make check test-sanitizers                      # once, model-free, on one Mac
make check-native-metal                         # on the other Mac
make install release-check MODEL=mlx-community/Qwen3.8-27B-4bit REVISION=<commit> BASELINE=../splash-baseline
make install release-check MODEL=unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M LANGUAGE_ONLY=1 REVISION=<commit> BASELINE=...
make install release-check MODEL=mlx-community/Qwen3.6-35B-A3B-4bit REVISION=<commit> BASELINE=...
make install release-check MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M REVISION=<commit> BASELINE=...
make install release-check MODEL=incoai/Qwen3.8-27B-Splash BASELINE=../splash-1.0.2
make install verify-models MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
make test-agent-real MODEL=mlx-community/Qwen3.6-35B-A3B-4bit REVISION=<commit> AGENT_SCENARIO=smoke AGENT_CLIENTS=...
```

Pin each upstream model to one commit, the same on both Macs. Without a Hub
token for a private draft repository, set `DRAFT_MODEL` to a local copy of the
draft's checkpoint ([Drafts](#drafts)). The Metal suite depends on the GPU
family, so it runs once on each Mac: within `make check` on the first, alone on
the other. Per model, `release-check`:

- runs the runtime oracle and, when the installation serves vision, vision
  parity (`test-real`);
- restarts the installer offline, with the Hub unreachable
  (`HF_ENDPOINT=http://127.0.0.1:9`) and with an empty `HF_HUB_CACHE`: each
  restart must start the same assembly within 10 seconds, name the Hub's
  reason on one line when it asked the Hub ([Revisions](#revisions)), and
  download nothing; a legacy package is not restarted. It then hashes the
  sources and records in `weights.json` the component, size and SHA-256 of
  every weight image the installation loads (`verify-models`);
- runs the HTTP smoke, which for a text-only installation checks the 400s
  instead of images (`test-http-real`);
- stops and restarts a server with a [persistent cache](#persistent-cache),
  and requires the conversation's next turn to restore its prompt from disk
  (`test-http-real` with `HTTP_SMOKE_ARGS="--persistent-cache --max-cache-disk 8G"`);
- compares this build with `BASELINE`, which must have another build
  identity, in ABBA order (`test-performance-real`), every round after the
  first to report a [Neural Engine share](#neural-engine-prefill) running that
  share when its build takes one: output tokens and acceptance must be
  identical (`EXPECT_OUTPUT_CHANGE=1` allows changed outputs with acceptance
  within 0.02), and so must the bytes of the weight images both load
  (`dev/benchmarks/weights.py`); decode and prefill GPU time, with the cold
  prefills of short prompts (481 to 2017 tokens, a first chunk of 480 to 2016
  rows) when both builds' benchmarks
  run them, may regress by at most the larger of 2% and twice the run's own
  ABBA spread, a spread above 5% fails as inconclusive, and this build's
  memory plan must hold, without the Neural Engine split, at least the
  context the baseline's holds without it. `ANE_FFN_SHARE=SHARE` runs
  every round at that share instead, in each build whose benchmark takes one
  (0: the GPU alone); a baseline whose benchmark takes none must report none.

Results go to `build/release/<owner>--<repo>[--VARIANT]/`. The weight images
do not depend on the GPU, so each model's `weights.json` must be identical on
the two Macs. The unpinned `verify-models`, run after the pinned ones while the
default branch still names the pinned commit, resolves the branch online, and
its unreachable-Hub restart must fall back with the Hub's reason. The agent
clients depend on neither the model's format nor the GPU: run the smoke
scenario once per Mac, with the five clients split between the Macs, and
`AGENT_SCENARIO=complete` for one model when the client integration changed.
Expect about 1.5 hours on an M5 Pro and 2.5 hours on an M3 Max, most of it in
the three 27B comparisons. A laptop can cap its GPU power during a long
comparison and so make it inconclusive; rerun `make test-performance-real` for
that model alone once the Mac has cooled.

`.venv/bin/python dev/tools/kernel_identity.py BASELINE/build build` compares
the AIR of each kernel the decode path may run in the two builds'
`splash.metallib` (`--match` for others) and fails if any differs.

Two pairs of constants in `runtime/engine/Protocol.hpp` and `server/protocol.py`
version what the server and the engine exchange. When the native wire layout
changed since the last release, bump `kProtocolVersion` and `PROTOCOL_VERSION`
together; each side refuses frames of another version. When the status
document the engine writes changed, bump `kStatusSchemaVersion` and
`STATUS_SCHEMA_VERSION` together; the server refuses a status document of
another schema. Builds between releases share a version while its layout
changes.

### Local benchmarks

From a source checkout with the model installed, use the native benchmark for
prefill, decode and batch measurements:

```sh
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit BASELINE=/path/to/baseline
```

The first characterizes this build: the decode widths B1-B4, a 14,096-token
partial-prefix request and the cold prefills of short prompts, three samples
each, in
`build/release/<owner>--<repo>[--VARIANT]/backend-benchmark.json`;
`make benchmark-backend MODEL=...` adds the 2K to 128K contexts the memory
plan holds. The second compares this build with a retained checkout's in ABBA
order, as the release check does ([Release check](#release-check)), and writes
`backend-regression.json` there. Neither is a comparison with another engine
or a test of agent task quality.

`benchmark-backend` lists the contexts the memory plan cannot hold in its
report. Its cache checks reuse each context's cached prefix, so they need that
memory free: when other programs leave too little, the engine evicts cached
prefixes and the checks fail, naming what each lookup found.

For a same-machine HTTP regression check, retain a `splash` binary **and its
adjacent `splash.metallib`** built from a checkout with the same native wire
version and status schema as this one (the server refuses any other), with
`engine-tests/weight-digests` beside them, then run from the candidate
checkout, after
`make build/engine-tests/weight-digests`:

```sh
.venv/bin/python -m dev.benchmarks.http_regression \
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M \
  --baseline-binary /path/to/baseline/build/splash \
  --contexts 2048,10000 --samples 5
```

It takes any installed model and, for an upstream one, holds its assembly for
the whole run, so every round serves the same model. It starts isolated servers
in ABBA order, compares matched cold, exact-prefix and decode requests by the
release check's speed rule and weight bytes (decode by `metrics.decode_cycle_ms`
per output token, so host work between commands counts), and saves
`build/release/http-regression.json`. With `--burst N` it sends N requests of
each context at once instead, 16 output tokens each, and reports each build's
replay points lost to the burst (`replay_state_publication_failures`), its
decode rate and advertised context. `--follow-up`, which needs `--burst`, then
sends each conversation's next turn and compares their time to first token and
how many resumed at their replay point; a burst run passes only with it. It does
not contact your running server. Use the same power mode and charger, stop other
GPU workloads, and report chip/GPU cores, memory, Splash version, model
revision, actual input/output token counts, and cache hits with results. Keep
cold prefill, cached TTFT and sustained decode separate; a UI token rate alone
does not measure end-to-end agent performance.

## Package

Release archives contain no Hugging Face credentials and use the official model
list committed with the source. `dev/tools/update_model_catalog.py` regenerates
that list from the official collection, independently of packaging; the
model-catalog workflow runs it and opens a pull request while the workflow is
enabled.
Users accessing private models supply their own `HF_TOKEN` or Hugging Face login.

Release versions are three-part, `x.y.z`, with no `v` prefix: after `1.2.1`
comes `1.2.2` for a fix or `1.3.0` for a feature. Use the same version in all
three commands:

```sh
VERSION=1.2.2  # the release being built
make package RELEASE_VERSION=$VERSION
make package-bottle RELEASE_VERSION=$VERSION
make package-check RELEASE_VERSION=$VERSION
```

`make package` writes the runtime archive,
`dist/splash-<version>-arm64-macos26.tar.gz`, its checksum beside it
(`<archive>.sha256`) and the formula, `dist/splash.rb`; `make package-bottle`
adds the bottle to `dist/` and its block to the formula. These commands do not
publish. Build bottles on the oldest supported macOS. Bottle/check commands use
a temporary tap and remove their installation; they refuse to replace an
existing Splash installation. The install check requires a poured bottle and
runs the bundled launcher without a compiler or separate Python installation.

To publish, tag the verified release commit in `incoai/splash` with the version
(no `v` prefix), preserving existing history and tags. Create a GitHub Release
with the runtime archive, its `.sha256` file and the bottle. Open a pull request
in `incoai/homebrew-tap` replacing `Formula/splash.rb` with `dist/splash.rb`;
its URLs must point to the published release assets. After the tap update
merges, verify a fresh install and an upgrade from the previous release through
the public tap, including a real model request and preservation of user data.
Run `brew audit --strict --online incoai/tap/splash`.

Before publishing, verify that the models the README names and the draft
repositories of `families.FAMILIES` are publicly accessible. Check the installed
bottle on a supported Mac without developer tools; building from the source tree
is not an installation check.

The runtime package allowlists engine, Python, server and launcher files; tests,
benchmarks and developer documents are excluded. User model links survive
upgrades; downloads remain in the Hugging Face cache.

### Private test releases

Testers can install a build before it is public, from a private Hugging Face
repository:

```sh
make package RELEASE_VERSION=$VERSION
make publish-test RELEASE_VERSION=$VERSION TEST_RELEASE_REPO=owner/splash-releases
```

`make publish-test` (`dev/tools/publish_test.py`) uploads the archive, its
checksum, the tester installer `dev/tools/install.sh` and `latest`, which
names this version, to `TEST_RELEASE_REPO` with the developer's own
`hf auth login`, creating the repository as a private one if it does not
exist; `publish_test.py --no-latest` uploads without moving `latest`. It
prints the command testers run with a read token for the repository in
`SPLASH_TOKEN`: curl reads the token's header from its standard input, so the
token is in no command line or URL. The installer requires an Apple Silicon
Mac on macOS 26.4 or newer. It downloads the version `latest` names, or
`SPLASH_VERSION`, checks its SHA-256, unpacks it under
`~/Library/Application Support/Splash/app`, points `current` at it, writes a
`splash` command into `/opt/homebrew/bin` or `/usr/local/bin` when writable,
else `~/.local/bin` (`SPLASH_BIN_DIR` chooses), and removes older versions. It
refuses to replace a `splash` command it did not write, and to switch versions
while a server runs. Running it again upgrades; installed models stay. To try
an archive before uploading it, serve `dist/` (`python3 -m http.server`) and
run the installer with `SPLASH_BASE_URL` naming that server, `SPLASH_VERSION`
and any `SPLASH_TOKEN`.
