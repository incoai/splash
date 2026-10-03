# Development

Use Apple Silicon with macOS 26.4+, Xcode 26 or newer, Python 3.12–3.14,
and a Metal 4 compiler with `uint4b_format` tensor support.
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
trained for it. The first serve sets up Python dependencies, downloads the
model and its draft, and prepares the weights once
([Weight preparation](#weight-preparation)); each start follows the model's
revision ([Revisions](#revisions)). Legacy Splash packages remain loadable
([Legacy Splash packages](#legacy-splash-packages)). Public repositories need
no login; private or gated ones need `HF_TOKEN` or `hf auth login`. Ctrl+C
stops serving, and a second Ctrl+C stops the engine at once; stop before
upgrading.

An engine that fails is restarted at once. If it fails again within 60 s of
starting, the next restart waits 5 s; after a third such failure (failed
restarts count) Splash stops restarting it, and requests get 500
`engine_failed` naming the crash trace until the server is restarted.
Meanwhile generation requests get 503 `engine_recovering`, whose message names
the last failure. An engine whose loop leaves a status request unanswered for
30 seconds, while requests are pending or during a background status refresh,
is failed and restarted by the same rules.

Use `--max-context 100K` or `--max-memory 28G` to set optional limits. Memory
limits cap Metal allocations, not combined process RSS. Agents must already be
installed; `./splash claude|opencode|codex|hermes|pi` connects to the running server.
Arguments pass through, for example `./splash codex resume --last` or
`./splash hermes chat -q "Hello"`.

Set `SPLASH_API_KEY` in the server and agent shells to require authentication;
`serve --api-key KEY` overrides the server's environment value. API requests
then require `Authorization: Bearer KEY` (or Anthropic's `x-api-key`). Health
and readiness probes and the chat page remain public; enter the key in the
chat page to send requests. The page does not persist the key. Use
`serve --no-webui` to disable the page. Authentication is off by default.

HTTP request bodies are limited to 128 MiB; `serve --max-request-size 256M`
overrides this. Concurrent input bytes share a budget of at least 512 MiB
(or twice the request limit), including retained generation inputs. This is
an input-byte budget, not a process RSS limit: large ASCII/base64 strings can
use roughly twice their encoded size during JSON parsing alone. Decoded images
and object-heavy JSON need additional memory. Oversized requests return 413;
exhausted ingress capacity returns 503. A connection that has sent no request
yet, or is receiving an upload refused unread, gives way to a new one when
every connection slot is taken, the first with that 503, so stalled clients
cannot lock others out. Image and model context limits apply independently.
Stored Responses history is charged before decoding. Uploads allow 30 seconds
of inactivity; total upload time is limited to 30 seconds plus the body size
at 512 KiB/s (286 seconds for 128 MiB), capped by `--request-timeout` when set.
Timed-out uploads return 408 and release their input reservation. An upload
refused before it is read, such as one over the shared budget, is still
received on these terms, so a client that sends its whole body before reading
the response gets the refusal.
`/status` reports `http.request_body_bytes` and `http.max_request_bytes`.

Source `install/completions/splash.bash` for Bash or
`install/completions/_splash` for Zsh after `compinit`. Completion suggests
commands, the official model IDs (bundled, and as `splash serve` last refreshed
them), the upstream models the README starts with and installed models, a
GGUF's `OWNER/REPO:VARIANT` included, without network access.

## Server configuration

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

A page served elsewhere, or an application's webview such as Jan's and other
Tauri apps', calls the API from another origin, which its browser names in
`Origin`. `--allowed-origin ORIGIN` admits one, such as `tauri://localhost` or
`http://localhost:3000`; repeat it for more, or pass `'*'` for every origin,
which is vLLM's default. The server then answers the browser's preflight
`OPTIONS` request and names the origin in `Access-Control-Allow-Origin` on its
responses, which let the page read `Retry-After` and `WWW-Authenticate` too. A
page of any other origin gets 403. Its browser hides that response from the
page, which sees a network error, so the server prints the refused origin and
the `--allowed-origin` flag that would accept it, once per origin. A `file://`
page or a sandboxed frame sends `Origin: null` instead, which only `'*'`
admits; the server does not print its refusal. Origins match exactly: a pattern
such as `tauri://*` or `http://*.example.com` is refused at startup; only a
bare `'*'` admits every origin. With `'*'` every page open in a browser that
reaches the server can use it, so set `--api-key` too; the server warns at
startup without one.

Use `--port 8001` or set `SPLASH_PORT=8001` to select another port. Set the same
`SPLASH_PORT` in the local agent shell. Separate ports allow separate servers;
their memory limits are independent. The packaged agent launchers connect to
loopback, so use a listener that includes loopback when launching agents locally.

### Server options

`splash serve --help` lists all options and examples. Common options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--revision` | Default branch | Select an upstream target branch, tag, or commit. See [revisions](#revisions). |
| `--draft-model` | Matching DFlash2 checkpoint | Override the draft with a compatible repository or local directory. See [drafts](#drafts). |
| `--language-only` | Off | Skip vision loading; image and PDF input is rejected. See [vision](#vision). |
| `--host` | `127.0.0.1` | HTTP bind address. |
| `--port` | `SPLASH_PORT` or `8000` | HTTP port. |
| `--max-memory` | Auto | Ceiling on Metal allocations, e.g. `28G`; not combined process RSS. |
| `--max-context` | Auto | Context limit, up to `256K`, e.g. `100K`. |
| `--max-cache-disk` | `0` (off) | Session-local SSD cache, e.g. `16G`. See [disk cache](#disk-cache). |
| `--kv-format` | `int8` | Target KV storage: `int8` or `bf16`. |
| `--decode-share` | `0.5` | Decode time owed per unit of prefill time while other requests generate. Higher keeps their output faster during a long prompt and slows that prompt; `0` alternates one command each. |
| `--max-image-pixels` | `4194304` | Maximum resized pixels per image. An image's vision scratch grows with its patches (pixels / 256), to about 600 MiB at the default. |
| `--request-timeout` | None | Seconds a request may take from its arrival; a request's own `timeout` can only shorten it. |
| `--queue-size` | `32` | Requests admitted at once, running or waiting; more get 503 with `Retry-After`. |
| `--allowed-host` | No extra names | Additional HTTP Host name, e.g. `mymac.local`; repeatable. |
| `--allowed-origin` | No other origin | Origin whose pages may call the API from a browser or webview, e.g. `tauri://localhost`; `'*'` for any; repeatable. |
| `--api-key` | `SPLASH_API_KEY` or none | Require a bearer token or `x-api-key`. |
| `--no-webui` | Off | Disable the chat page. |

The startup summary and `maximum_context_tokens` in `/status` show the effective
context limit. `/v1/models` and `/v1/models/{id}` report the same limit as
`max_model_len` and its compatibility alias `context_length`, including model
aliases. Clients can impose a smaller limit. With enough memory, request the
full native window with `--max-context 256K`. This is a capacity limit, not a
guarantee of a fast first token for a long uncached prompt. If the model
cannot fit, startup prints a memory budget breakdown and stops.

`splash pi` adds a `splash` provider to Pi's `models.json` (`splash-<port>` for
a server on another port), preserving other providers, settings and sessions.
The browser chat and agent launchers connect to the running server; a model
need not appear in a client's catalog to serve it by its full repository ID.
`splash opencode`, `pi` and `hermes` configure an output limit per response
of 32K tokens (`CLIENT_RESPONSE_TOKENS` in `install/clients.py`); OpenCode and
Hermes, which reserve it out of the context they compact at, get a quarter of
a context under 128K instead. Hermes 2026.9.7 and later ignore it and, like
Codex, leave the limit to the server; Claude Code keeps its own, which
`CLAUDE_CODE_MAX_OUTPUT_TOKENS` raises.

`splash hermes` runs Hermes in the `splash` profile (`splash-<port>`) of the
user's Hermes root, `~/.hermes` or the root `HERMES_HOME` belongs to, and
creates it with `hermes profile create` on first use. It writes only the
profile's model settings; Hermes's tools and the root's `config.yaml` remain
the user's.

Earlier versions gave Hermes a home of its own, `install/agents/hermes` in a
checkout or `~/Library/Application Support/Splash/runtime/hermes` in a
release, which Hermes took for a root: a Hermes that manages its own runtime
installed its tools there and pointed the `hermes` command at them. While that
directory still exists, run `hermes pm install` once in a normal shell; it
installs Hermes's tools under `~/.hermes/tools` and points the `hermes` and
`hermes-acp` launchers in `~/.hermes/hermes-agent/.hermes/bin` at them. Once
those launchers no longer name the old directory, or do not exist, as with
Hermes before its managed runtime, delete it with the sessions Splash started
there.

### KV cache precision

Select the target KV format when starting the server:

```bash
splash serve --model mlx-community/Qwen3.8-27B-4bit --kv-format bf16
```

BF16 avoids target KV quantization, uses approximately twice the target KV
memory, and can be slower at long contexts. Model weights are unchanged.
Restart to switch formats. Omit `--kv-format` or use `--kv-format int8` for the
default. The [SSD cache](#disk-cache) supports both formats, preserving their
stored bytes without further quantization; it does not survive a restart.

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
environment. Unset, the model's template default is unchanged. Effort names are
passed to the template using the same mapping as per-request values, not token
budgets.

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --default-reasoning-effort none
```

`/apply-template` uses the same default. Anthropic `thinking` keeps its protocol
semantics (off when omitted); judgment endpoints always disable thinking. The
built-in chat page sends no effort unless the user picks one.

A Chat request's `chat_template_kwargs`, as vLLM and SGLang accept them, are
passed to the template as variables and outrank the effort, so
`{"enable_thinking": false}` turns reasoning off. They cannot set what Splash
passes itself, such as `tools` or `add_generation_prompt`.

## Upstream model loading

`install/upstream.py` installs a model from its upstream repository: it
inspects the target, pairs the draft trained for it and decides when to follow
the Hub. The result is an assembly, a local directory of links to the sources'
Hub snapshots, published atomically. The other installer modules each own one
part: `hub.py` the sources, the Hub cache and its pins; `assembly.py` the
assembly layout, its build, verification and garbage collection, and the
metadata derived from a GGUF; `families.py` the registry; `legacy.py` Splash
packages; and `models.py` model IDs, selections, the installation lock and the
command line (`install/models.py --model ID prepare|verify|link`, where `link`
prints the selection link). The assembly's
`model.json` records the resolved sources and selected formats. The native
loader reads it, and `splash serve`, `test-http-real` and the HTTP regression
benchmark hold it while they run, so a concurrent installation cannot collect
the assembly they serve. It is local installation metadata, not a file model
publishers supply.

A target is identified by its own metadata: an MLX config's `text_config`, or
the one `gguf.model_config` derives from the selected GGUF's header, read with a
few HTTP range requests before any weight download. The registry
(`families.FAMILIES`) states each supported architecture's signature and the
draft trained for it; repository names and model-card `base_model` fields play
no part. An MLX target must declare affine 4-bit, group-64 `quantization` in
`config.json`. Native source adapters validate model geometry, quantization,
tensor shapes and draft compatibility again before execution. Remote Python
code is not loaded.

```bash
splash serve --model mlx-community/Qwen3.6-35B-A3B-4bit
splash serve --model unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M
splash serve --model mlx-community/Qwen3.8-27B-4bit --language-only
```

A model ID with `--revision`, `--language-only` or `--draft-model` is a
separate installation from the same ID without them.

### Revisions

Installation resolves each source's revision to a commit once, downloads by
that commit and records it in `model.json`, so a repository update cannot mix
files from different revisions. It pins those snapshots in the Hub cache
(`refs/splash/<installation>/<commit>`), so pruning the cache cannot remove files
an installed model links. Publishing a new assembly retires the installation's
other pins, in the repositories it links and in those its predecessor linked,
so pruning can free what no installation links any more.

Every start resolves the target's revision (the default branch, or
`--revision`) with one Hub request of at most 5 seconds (`hub.HUB_TIMEOUT`),
and when the Hub answered, the default branch of the draft's repository
([Drafts](#drafts)) with another; `hub.Repository.resolve` alone decides
whether the Hub is asked:

- The installed commits: the assembly's links, sizes and times are checked and
  it starts. It is re-assembled first when the draft's repository moved or
  this release changed the GGUF metadata adapter; if the new draft cannot be
  fetched or is not the family's ([Drafts](#drafts)), the installed one is
  kept.
- A new commit: only changed files are downloaded, and the new assembly
  replaces the installed one atomically once published.
- No answer, or a new commit that cannot be installed: the installed model
  starts, with one line naming the Hub's reason or, on stderr, the
  installation attempt that failed.
- A 40-hex `--revision` never moves, nor does its draft, and
  `HF_HUB_OFFLINE=1` forbids the Hub: both start a verified installation
  without a request.

There is no update flag; to stay on one commit, pass it as `--revision`. A
missing assembly, or one that no longer verifies, is built again. Without the
Hub it is built from a cached snapshot, of the commit the `--revision` names,
or else the one the installation recorded or pinned, or one the Hub cache
records for the branch, never of another revision. Only files downloaded before
are available, which is enough to rebuild a damaged or deleted assembly; a new
selection needs the Hub once for its draft's default branch. The installer
never rewrites upstream files.

### Model storage

Allow space for the target and BF16 draft downloads plus prepared copies of
their weights: up to about 40 GB in total for the Qwen3.8-27B 4-bit examples
and 48 GB for Qwen3.6-35B-A3B. Other variants have different sizes. Downloads
use the Hugging Face cache; prepared weights use `~/Library/Caches/Splash/weights`
(`SPLASH_WEIGHT_CACHE` relocates them). Later starts reuse prepared weights,
and `brew upgrade splash` preserves models and agent sessions.
See [weight preparation](#weight-preparation) for cache validation and cleanup.

### Model cache

To download new models to another disk, set the cache location before serving:

```sh
HF_HUB_CACHE=/Volumes/Models/huggingface splash serve --model mlx-community/Qwen3.8-27B-4bit
```

`HF_HUB_CACHE` selects the Hugging Face download cache. Alternatively, set
`HF_HOME` to relocate the Hugging Face home directory, including its default
`hub` cache. Model links and agent sessions stay in Splash's data directory;
existing downloads are not moved. Prepared weights have their own cache,
which this does not move ([Weight preparation](#weight-preparation)).

### Drafts

Each family names the repository of the DFlash2 checkpoint trained for it
(`Draft` in `families.FAMILIES`), which holds it as the release publishes it:
`config.json` and BF16 `model.safetensors` (or the shards its index names).
Installation downloads only those files and follows the repository's default
branch as it follows the target's ([Revisions](#revisions)); `--draft-model`
accepts another repository, followed the same way, or a local directory that
holds them. A checkpoint is installed only when its configuration states the
family's draft signature (`Draft.signature`), every field and value native
loading requires, so a draft of another architecture never replaces one that
loads. Native loading validates the configuration against the target and
prepares the draft like a target ([Weight preparation](#weight-preparation)):
`DraftCheckpointLoader` (`DraftCheckpoint.cpp`) plans the packed draft files
of a Splash package, `layer-<N>.bin` and `model.bin`, and `AffinePreparation`
quantizes each projection to 4 bits in groups of 64 as MLX's affine
quantization rounds it and copies every other tensor as stored. For both
families the prepared files are byte for byte the Q4 drafts of the Splash
packages.

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

Both sources prepare the packed `vision/model.bin` layout, which the one BF16
vision operator reads: BF16 tensors are copied, and F32 or F16 tensors are
converted under the exact-BF16 rule of [weight preparation](#weight-preparation).
Unsloth's mmproj stores its 1-D tensors, patch embedding and position table as
F32, all of them BF16-exact, and prepares byte-identical to the packed file.
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

### Weight preparation

Source adapters write a model's target, draft and vision tensors into prepared
files: an MLX target, the DFlash2 draft and any vision tower into the packed
layouts of Splash packages, which run the same kernels, and a GGUF target into
the `MDGG0001` layout of the GGUF kernels. Each adapter is a loader, which
validates the source's metadata and plans its files, and a writer:
`AffineTargetLoader` (`AffineTarget.cpp`) and `AffinePreparation` for an MLX
target, `DraftCheckpointLoader` (`DraftCheckpoint.cpp`) and `AffinePreparation`
for the draft, `GgufTargetLoader` (`GgufTarget.cpp`, planned by `GgufImage.cpp`)
and `GgufPreparation` for a GGUF target, `VisionLoader` and `VisionPreparation`
for an MLX or GGUF vision tower. They open their files through `PreparedFiles`,
the `PreparedWeights` cache with the load's guards; the target and draft loaders
keep their planned images in `PreparedImages` (`PreparedFiles.hpp`), whose
sizes `preparedModelWeightBytes` sums. `AffinePreparation` reorders
an MLX target's codes, scales and biases into 256-row tiles without
requantization, quantizes the draft's BF16 projections into the same tiles
([Drafts](#drafts)) and computes GDN decay as `float(-exp(double(A_log)))`,
which may differ by one float ULP in this small vector from packages produced
with MLX's float exponential. `GgufPreparation` repacks GGUF blocks ([GGUF
targets](#gguf-targets)).

Preparation never rounds a target or vision weight, and rounds the draft's
projections only as the packages' drafts are rounded. A tensor it converts to
BF16 (vision tensors stored as F32 or F16, a GGUF's convolution taps and
time-step bias) must be exactly representable in BF16; otherwise preparation
fails, naming the tensor and, for a vision tensor, its file.

The cache is `~/Library/Caches/Splash/weights`, or the directory
`SPLASH_WEIGHT_CACHE` names; nothing else selects it. It holds an additional
copy of the weights about the model's size, its prepared target, draft and
vision tensors. Preparing needs that much free disk space plus a 2 GiB reserve:
before anything is written, the factory (`ModelFactory.cpp`) constructs the
vision tower's loader (`planVisionLoader`, which the vision encoder test
shares), the draft's and the target's, and checks once for the space their
missing files add beyond the entries they supersede (below), plus the largest
file written while the entry it replaces remains, plus the reserve. After a
preparation-identity change, preparing thus needs little more than its largest
file. Uninstalling a model does not delete possibly shared prepared weights.
With Splash stopped, entry directories can be deleted; deleting the whole cache
causes preparation at the next load.

A prepared file's key hashes its adapter's preparation identity, its plan, and
the bytes, type and shape of every source tensor it reads, located and hashed
within its file's tensor data. An edit to a source's metadata only (a GGUF chat
template, a safetensors header), `config.json` or files the component does not
read keeps every key. The preparation identity is a build-generated fingerprint
of only the code that writes the bytes, the files listed per adapter in
`INPUTS` of `dev/tools/weight_preparation_identity.py`; inference, parser,
planner and reader changes keep it. The hashes of the images prepared from the
test fixtures, in `dev/tests/fixtures/weight-goldens/goldens.json`, fail the
tests on any change of prepared bytes. The README beside it gives the
procedure for an intended change: the key the new bytes need, and the order in
which the independent layout oracles and the hashes are updated.

Each entry records in `source` its component (such as `target/layer-0.bin`),
the digest of the source data it was written from and the source path.
Publishing an entry removes the complete entries it supersedes: the same
component from the same source data under another key, which an earlier
preparation identity wrote. Entries of other sources or revisions, which
installations may share, stay. Removal happens under the converter lock, so no
entry being written is touched, and a running process keeps the files it has
mapped until it unmaps them. Two builds of different preparation identities
sharing one cache supersede each other's entries at every start; give a
development build its own `SPLASH_WEIGHT_CACHE`.

One writer per cache serializes conversion; complete cache hits never wait for
this lock. Each output's disk space is preallocated before writing.
Interruption, disk-full errors and memory-pressure rejection cannot publish
partial files; concurrent external disk activity can still exhaust the volume.
Every converter-lock acquisition removes abandoned writes, and so does a warm
start that finds the lock free; a warm start that cannot lock or clean the cache
(read-only, or without `flock`) leaves them and still loads. Replacing an
invalid entry also drops its digest proof. Retrying reuses previously completed
files, which are read-only. Cold preparation reports each artifact's progress.

Cold source hashing and output validation stream bounded buffers. Unchanged
files reuse a digest proof tied to inode, size, birth time, mtime and ctime, so
a remounted volume keeps its proofs; a write or replacement invalidates it.
This is not a full disk scrub on every startup. Preparation uses uncached
destination I/O. Every adapter sizes its conversion steps to one staging bound,
input and output together, of 32 MiB (`kWeightPreparationStagingBytes`),
whatever the tensor, layer or expert count, inside a 64 MiB admission reserve
that also covers source metadata.
Complete rows and multiple row tiles are processed together where possible,
avoiding per-row I/O and small GPU waits. Startup runs two checks
(`RuntimeResources.mm`), both stopped by cancellation. `admitWeightPreparation`,
which the loaders receive as `admitConversion`, admits the conversion workspace
on a cache miss, before anything is allocated and again before each chunk: it
requires normal memory pressure and host headroom for the startup reserve plus
the 64 MiB workspace. Cache hits, bounded source verification and every other
Metal operation of startup pass `admitMetalOperation` instead, which critical
pressure or too little headroom for the startup reserve still fails.

`WeightFile` maps completed files, prepared or packed, read-only into one
no-copy Metal buffer, so no model-sized anonymous allocation holds the weights.
It maps a prepared file only if a digest proof covers the very file it opened
and matches the digest its entry records (`requireVerifiedFile`), so a file
replaced or changed after `prepare` checked it is refused.
Runtime admission counts prepared weights, draft and vision exactly once
(`preparedModelWeightBytes`, which `tune-kernels` and the runtime oracle use
too). Before loading, startup refuses a model whose prepared weights, with the
pipeline and runtime reserves, one lane's state, the KV runway and any disk tier
state staging, exceed the hard budget, so a model that can never fit is not
prepared. File backing does not make Metal-resident pages reclaimable.
Every buffer the backend allocates or wraps belongs to one residency set
attached to its command queue (`MetalBackend::allocateBuffer`): weights, KV
extents, state cells and draft rings, and scratch alike stay wired between
requests until 10 minutes pass without a command, and the next command wires
them again. Memory returns to macOS when the engine releases it, never because
macOS compressed or dropped an idle buffer. macOS page cache, driver
allocations and other applications still affect memory pressure and swap.

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

`loadQwenTarget` (`QwenTargetLoader.hpp`) reads a target's files
(`QwenTargetFiles`: packed files, or the files `AffineTargetLoader` or
`GgufTargetLoader` prepared) through the format that stores them.
`AffineTargetFormat`, for packed and MLX-prepared files, reads every
projection, a fused one too, as one affine Q4 tensor and the norms as bf16.
`BlockTargetFormat`, for prepared GGUF images, reads each GGUF tensor as one
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
  segment of the GDN input projection), F32 or BF16, which preparation widens
  to the F32 values it equals.

Of Unsloth's files in September 2026 that covers every file of Qwen3.8-27B and
Qwen3.6-35B-A3B, from UD-IQ1_S up, but UD-Q8_K_XL and BF16, whose BF16 tensors
need kernels that do not exist yet. llama-quantize's default type selection
stores alpha and beta in the file type's format (Q4_K in a Q4_K_M), as in
lmstudio-community's files, so those load too. PQ2_0 is Prism ML's type 142,
`block_pq2_0` of PrismML-Eng/llama.cpp, which upstream GGML does not define:
2-bit codes q worth d (q - 1) with one half d per 128 weights. A format's
image takes the bits per weight of its GGUF blocks, but for Q3_K's and Q6_K's
padded meta units (1/16 bit more) and IQ3_S's chunk words (4.06 bits for its
3.44).

Prism ML's GGUFs, such as `prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0`, store
every projection for rotated inputs: the `prism.hadamard.*` metadata names the
tensors whose weights multiply H (D x), H the normalized Walsh-Hadamard
transform of each block of 1024 inputs and D an explicit sign per input, and
the token table, whose rows are stored as H (D e). The engine runs that one
form, on dense targets whose rotation names exactly the tensors the planner
repacks (every quantized projection and the head, and alpha/beta when
quantized), a PQ2_0 token table, and GDN value heads in grouped order (the
installer screens the parameters, `GgufFile` and the planner check the rest).
A rotated projection rotates its input once into `LinearScratch::rotated`
(`gguf_rotate`, in fp32 and rounded once to bf16) before its quantized
segments, whose kernels are the format's, while float segments read the input
as it is; the register tile prepares its table from the rotated rows, so the
input's producer writes plain rows. The token table gathers each row through
the inverse (`gguf_embed_rotated_pq20`).

At load time the engine validates the GGUF metadata, including the rotary
embedding and norm epsilon the kernels assume (`rope.freq_base`,
`rope.dimension_count`, `attention.layer_norm_rms_epsilon`, and no
`rope.scaling.type` but `none`), and plans the `MDGG0001` layout.
`GgufPreparation` stages rows in image order on the CPU within the
staging bound, splitting rows wider than it into column chunks, runs the
`gguf_repack` kernel, and writes its planes into a prepared file. Embeddings
and F32 sections use bounded direct copies.

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

A GGUF kernel of one quantized tensor names its epilogue last: `a` none, `r` residual, `g` the
up pass with the silu gate. The staged ones are `gguf_decode_<format>_m<rows>_<e>` and
`gguf_prefill_<format>_<e>`, the register ones `gguf_decode_sg_<format>_l<lanes>_<e>`, and the
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
and the decode-only value tables in `runtime/metal/abi/QuantTables.h`, which no prepared byte
depends on; weight preparation's repack ABI is `runtime/metal/abi/GgufRepack.h`. The tables are
llama.cpp's and the decoding follows its Metal kernels: both keep llama.cpp's MIT notice in
`THIRD_PARTY_NOTICES`, which the package ships.

The tests' CPU reference (`dev/tests/engine/GgufFormatReference.hpp`) must reproduce the golden
hashes of upstream GGML's dequantization (llama.cpp 7ab4ee7; for PQ2_0, which upstream lacks,
PrismML-Eng/llama.cpp 01ae597) in `gguf-reference`, and `gguf-planner` checks the planner's
plans; both run in `make test-engine-cpu`.
`make test-engine-metal` runs `gguf-preparation`, which checks every format's planes, as the
production executor and its `gguf_repack` kernel prepare them, bitwise against the reference,
the prepared alpha/beta, norm, convolution and router bytes and the golden images; then
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

## Legacy Splash packages

Splash packages, such as `incoai/Qwen3.8-27B-Splash`, are the prebuilt format
that predates upstream loading, and `--model` still accepts them. They contain
`manifest.json`, packed `target/`, `draft/` and `vision/` weights and
`tokenizer/`; the manifest lists artifact paths, sizes and SHA-256 hashes.
Qwen3.8-27B packages use schema 3 / `splash-packed-q4`, Qwen3.6-35B-A3B
packages schema 4 / `splash-packed-q4-moe`. Compatible community fine-tunes may
use any nonempty manifest model name. Native loading validates geometry, tensor
sizes, binary headers, tokenizer and target/draft compatibility, and maps the
packed files without preparation. `install/legacy.py` installs a package as a
selection link to its verified Hub snapshot, pinned like an assembly's
sources. An installed package starts without a Hub request. A package has no
variants, so a `:VARIANT` suffix is rejected, and `--revision`,
`--language-only` and `--draft-model` require an upstream model ID.

## Code and API boundaries

- `server/`: OpenAI Chat/Completions/Responses, Anthropic Messages/count_tokens,
  typed judgments, templates, streaming and input processing. No client-version
  branches.
- `runtime/engine/`: scheduling, memory admission and reusable request state.
- `runtime/model/`: target/draft execution and vision.
- `runtime/ops/` and `runtime/metal/`: operators and Metal kernels.
- `install/`: launcher, client configuration and model installation.
- `dev/`: maintained tests, benchmarks and build/release tools.

Within `server/`, `server.py` owns HTTP and startup; `frontend.py` prepares
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
`splash serve` shares with the server once, each with its check, default and
help, and how the launcher passes it on; it imports only the standard library,
since the launcher parses them before `.venv` exists.

Tests substitute the values the native runtime holds constant (the Metal
command timeout and residency keep-alive, the KV tier's transfers, the input
queue's bound, the latency window, the prefill checkpoint interval and the
resource wait) and its clocks and live host-memory estimate through
`runtime/TestConfig.hpp` with `test::ScopedTestConfig`, never through a
production parameter; `make architecture-check` keeps production from writing
that configuration.

Tools can be combined with structured answers. Tool argument framing resolves
local references and projects object fields through schema composition. The
original schema validates complete arguments, including cross-field conditions,
dependencies and property-count rules that framing alone cannot enforce; array
item bounds and `multipleOf` above 64 are left to that validation as well, as
are patterns the grammar cannot compile (look-around, word boundaries,
backreferences). The framed schemas of one request are limited to 16 MiB. Extra
properties use JSON-encoded values; statically typed strings retain raw text,
so their patterns are checked on the complete call.
`tool_choice: "none"` renders the tools like any other choice and only
prevents calls. Remote schema references, parameter names containing XML
delimiters and `unevaluatedProperties` combined with `patternProperties` are
unsupported. Hosted search is unsupported; configure client-owned tools such as
MCP. Omitted effort uses the model default.
`response_format` constrains generation and validates final output; it does not
inject formatting instructions into the prompt. Clients should describe their
output requirements in their own messages.
Hidden thinking signatures use a persistent user key; imported encrypted thinking
preserves visible history without recovering the private reasoning.

`/status.admission` distinguishes memory and concurrency waits, counts the
requests held back behind one refused memory (`held_behind_refusal`, the
refused request included while a pass defers it; during recovery, the suspended
ones and those of a strictly higher priority) and those waiting for a disk
restore (`restoring`), reports suspended requests, recovery draining and the
oldest current wait age, which for a request holding admission closed runs from
when its wait began. Memory transitions also appear in the console. When macOS
runs short of memory, growth that no request in service needs pauses and the
cache gives memory back, a paced pass at a time, down to one lane's state
buffers and one KV extent. A pass counts only memory that leaves the engine; a
cached state it evicts first refills the lane's state buffers it keeps. A
request in service keeps growing within `--max-memory`, first into cached pages
no request holds: it takes cached KV, and a cached state only where it sits on
the KV that goes next, since a state's buffers give it no page. A new request
waits while another is in service unless it can start from what the engine
already holds, and with none in service it starts. A request whose start was
refused memory, under host pressure or at `--max-memory`, holds back the
requests that arrived after it until it starts, fails or is cancelled, so the
lanes that finish leave their memory to it; a higher priority is not held back.
It keeps holding them back while it waits for a prefix another lane is
computing. After a suspension, suspended requests resume before new work of
their priority or below, one at a time, higher priority first and then in the
order they arrived; one refused memory holds back the suspended requests after
it, and their resource waits do not run out while it does. A request of a
strictly higher priority than every suspended request waits for neither. When
every lane is taken, or a new request's start is refused memory that no copy
being written to disk is about to free, the resident of the lowest priority
below it is suspended for it (`cache.priority_suspensions` in `/status`) and
resumes once a lane and its memory are free. A new request below the priority
of a running lane neither starts nor takes a lane until no lane of that
priority runs, since it could not run before. Critical pressure evicts every
unpinned cache entry and stops all growth. `/ready` stays healthy under warning
pressure, and reports 503 while macOS reports critical pressure; requests
already running continue, and one that needs more memory is suspended until the
pressure lifts.

PDF input supports base64 documents within a shared 64 MiB source/rendering
budget and the native 64-image limit (one image per page). Model context and
isolated rendering limits also apply. URL inputs, opening passwords and citations
are unsupported.
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
to generate a fixed number of tokens. Tools and structured output generate
under a grammar, which decides where the output ends, so combining them with
`ignore_eos` returns 400. The engine receives it as bit 0 of the request
frame's flags word, which rejects undefined bits.

Streaming requests accept `"return_progress":true` (default false). Before output,
`prompt_progress` reports `{total, cache, processed, time_ms}`: prompt tokens,
initial cached tokens, completed tokens including cache, and elapsed milliseconds
since prefill admission. Updates follow completed chunks and never regress during
recovery; they are not a time estimate. Chat uses empty-delta chunks, text
completions empty-text chunks, Responses `response.in_progress`, and Messages
`ping`. Queueing and prompt preparation do not advance this counter.
Non-streaming requests cannot enable it.

`GET /status` returns instance identity and the effective context limit as JSON.
Proxy consumers can use these fields; additional fields may be added:

| Field | Meaning |
| --- | --- |
| `requests.submitted`, `completed`, `cancelled`, `failed` | Native request counters since engine start |
| `memory_actual.current_bytes`, `peak_bytes` | Metal allocations, not process RSS |
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
that long fails the engine, which then restarts.

`GET /metrics` exposes the same counters in Prometheus text format.
`splash_kv_free_allocated_pages` counts free pages of allocated extents, not
remaining capacity; memory headroom is `splash_memory_headroom_bytes`. Both
endpoints require the API key when authentication is enabled. Consumers should
tolerate missing native fields while the engine is unavailable, and counter
resets after an engine restart. Chat and text completion streams include token
usage when the request sets `"stream_options":{"include_usage":true}`; their
non-streaming responses always include usage. A proxy must consume these fields
to display statistics.

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

`/metrics` also exports fixed latency histograms in seconds, with a bounded
set of stages in `/status.latency`. HTTP duration includes body upload and
response writing for admitted API requests. Preparation, queue, template,
tokenization, output grammar preparation and image preparation are measured
separately; preparation includes its nested stages. Tokenization covers the encoding call, including reuse when
available. Histogram buckets are cumulative and labeled by upper bound.
HTTP TTFT (`http_ttft`) starts before upload and ends at the first native
token event; native TTFT (`metrics.ttft_ms`, per-request
`request_latency.ttft_ms`) starts when the engine receives the request.
Output intervals are between native token events, which can contain
multiple speculative tokens; they are not per-token latency. Native queue timing
is recorded from successful completions. These histograms live with the HTTP
process and survive a native engine restart.

HTTP bodies require Content-Length, and a browser's
Origin must match Host or be one `--allowed-origin` names. `--allowed-host`
permits additional hostnames. Request
logs omit bodies; full crash traces require explicit `SPLASH_CRASH_TRACE=1` and
can contain private conversation data. A frame over 16 MiB, such as a request
with large images, is kept only as a marker with its size and SHA-256
(`omitted_frames` counts them), and a trace missing engine input that way
cannot be replayed. `python -m server.crash_trace <trace>`, run from the
checkout root, replays a trace.

A request keeps its reusable model state at the last whole 32-token page before
its generation prompt, the text a chat template appends to open the reply: the
next turn may render it differently, so a follow-up resumes from there.

Until the request ends, suspended or not, that replay point is in use, and so is
the KV it restores through. When the last request using it ends, the point
becomes the newest ordinary state; the older points on its chain keep their age.
Cache victims come in three classes: checkpoints, then ordinary states and KV
(first the KV no state restores through, which saves no prefill), then what is
in use. No work displaces anything of a class above its own. Memory for running
requests takes what is in use after everything else. A start that a resident
lane holds back takes nothing in use: it waits for that lane. Nor does the lane
that yields first when no lane's growth fits, such as one just started beside a
decoding lane: its own suspension pays for the memory, and while other lanes fit
it waits for them, resident. A publication in use takes cached KV and states in
the same order, then the oldest state in use; of the KV it takes only leaves
whose page frees at once, and only while an extent can be emptied; the extent is
released at once, and the snapshot follows. An ordinary publication (a junction)
makes room the same way short of what is in use, and an optional checkpoint
takes only checkpoints; neither drops a state whose write must wait for the
write in flight. A disk copy in use may displace the oldest copy in use, and
ordinary or optional work never displaces anything in use. Nothing in use is
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
batch. Constrained requests use a separate batch for the host mask exchange.
A mask request the server leaves unanswered for 5 s fails that request with
the retryable `mask_timeout` (HTTP 503), so a stalled grammar cannot hold the
batch's command.
A lane's arithmetic can depend on the batch it decodes in and on how its
prompt is chunked. Concurrent requests change both, and chunk boundaries also
come from the cached boundary a request resumes from and from the junctions
and checkpoints earlier requests left. So a greedy or seeded request repeats
its output when it runs alone with the same cached prefixes; alongside other
requests, or with other prefixes cached, it can differ.

Long prefill uses disposable rolling checkpoints every 4096 tokens; none is
planned within one prefill chunk (2048 tokens) of where the request resumes or
of its replay boundary. Contended prefill adapts toward a 500 ms slice, keeping
2048-token chunks for long unopposed work. Without contention, a prompt that
can finish within one 2048-token chunk ends its chunk at its last row instead
of sharing it with a longer prompt. While requests of the same or a higher
priority decode (a lane waiting for its token mask is owed nothing), each slice
owes them decode time, `--decode-share` times its own, before the next slice
runs.
These policies do not extend client deadlines. Memory recovery waits are
bounded: after a suspension, new work waits for resident requests only while
memory is still short, and at most for the 30 s resource wait; suspended
requests then resume first, each within its own resource wait. A request of a
strictly higher priority than every suspended request waits for neither. A
resource wait's limit restarts whenever a lane submitted before the waiting
request, or admitted before the request was first refused memory or was
suspended, has work in flight, since that lane holds memory the request waits
for until it finishes; other lanes do not extend it. Readiness does not
guarantee that a request-sized allocation fits. A request that cannot fit even
alone, after every cached prefix was evicted, fails with 400
`capacity_exhausted`, naming `--max-memory` and `--max-context`; retrying it
fails the same way.

### Disk cache

`--max-cache-disk` adds an optional SSD tier for cached request states (GDN cell
plus draft ring) and KV pages. Default: `0` (off). RAM and disk copies share the
same block tree and recency order. Restoring a prefix keeps its disk copy, so
its next eviction needs no write while that copy remains cached.

Without the tier, a request that runs out of memory cannot publish its progress
checkpoints and replays its prompt after each suspension. With the tier off,
startup suggests it in one line when memory may not hold the advertised
context: the memory plan within what the host had available at startup beyond
its reserve and the warning margin (`EngineMemoryPlan::contextTokensWithin`).
The estimate is conservative, since macOS compresses other applications further
once the engine loads. The tier does not raise the context limit.

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
replaced or no longer needed. Matched KV restores start from the root toward
the selected state, in that order as the tier takes them, with the state read
alongside. Cancellation drops unsubmitted, unshared reads; submitted transfers
drain before their buffers can be reused. Restored states remain usable even
when there is no room to promote them into RAM cache. Promotion takes only the
RAM of a state that keeps a disk copy.

Two unlinked temporary files share one quota for live slots. A full quota
replaces the oldest redundant copy first, then the oldest sole copy, across
both KV and states. Sole copies of states in use, and the KV they restore
through, make room only for a copy that is itself in use, and last; an ordinary
state that finds no other room is dropped. A quota smaller than the working set
can cause repeated reads and writes; it is not a write-rate limit. A freed slot
returns its quota at once and its blocks to the volume (`F_PUNCHHOLE`) once its
file's IO worker finishes the transfer it is in, so the two files together
occupy about the live slots instead of each keeping its high-water mark;
`/status` reports that as `disk.file_bytes` beside `disk.used_bytes`. Until the
punch the other file can take that quota, so allocated blocks can briefly exceed
`--max-cache-disk` by the slots freed but not yet punched, bounded by the
transfer each file's worker is in: about 2% of a 10G tier in a 25-minute churn
run on an M5 Max. A volume sized exactly to the quota can therefore fill; the
write that finds it full stops that file's writes for the rest of the process,
as below, while serving continues. On a volume that cannot punch holes the
files keep the blocks of freed slots, which the runtime reports once. Closing
the server releases both files.

Transfers use `pread`/`pwrite` with `F_NOCACHE`, every one an aligned range:
whole 1 MiB chunks of 16 KiB-aligned memory that start at an aligned offset of
the slot (a state's staging buffer and lane buffers) move directly, everything
else through the file's own 1 MiB buffer. The KV tier takes no Metal memory.
The state staging buffer, one state (109 MiB for 35B, 187 MiB for 27B), is a
buffer of the backend's like every other: resident, and set aside by the memory
plan within `--max-memory` when the tier starts. A quota too small for one
state leaves the tier disabled and sets nothing aside.
A failed write disables further writes to that file. Failed KV writes retain
RAM pages; failed state writes invalidate the disk copy. A failed read
invalidates its cached data, allowing lookup to fall back to the surviving
prefix.

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
batch token budget before the per-slot boundary checks, which re-tokenize the
prompt once per option. Those checks also observe the request deadline, so an
oversized or expired request is rejected without paying for every option.
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

Requests carry the score-token IDs and Done events the selected f32 logits; the
server and the engine must speak the same native wire version. Scoring requires
2–255 distinct, in-vocabulary tokens, no images or generation constraints, and a
zero output budget. It may use the full context window because no generated
token needs a reserved position. The final prefill chunk runs the target head
but no sampling policy or DFlash decode. Successful scoring emits no Tokens
event, finishes with Stop, and reports zero decode time. Cancelled requests
carry no logits.

A non-finite logit row is a per-request failure, not an engine fault: a score
logit, or a token the sampling kernels could only select outside the
vocabulary. The engine reports `model_result_invalid` for that request alone,
before it publishes the failing step's cache state or any output, and the rest
of the batch finishes normally. Prompt chunks that already succeeded keep the
blocks they committed, exactly as they do for a cancelled request. GPU faults
and broken engine invariants stay fatal and still mark the runtime unhealthy.

## Validate

```sh
make check
make install test-real test-http-real MODEL=mlx-community/Qwen3.8-27B-4bit
```

`make check` needs no model weights. `make check-native-cpu` builds production
and runs native CPU tests without a GPU; `make check-native-metal` requires a
supported Metal device and runs the kernel tests under shader validation, and the Linear pipeline
resource check without it. They
include the preparation of small synthetic MLX, GGUF and vision sources and the
GGUF kernels on synthetic tensors. Hosted CI runs CPU checks and sanitizers.

The real-model targets take `MODEL` exactly as `splash serve --model` does,
and `REVISION`, `DRAFT_MODEL` and `LANGUAGE_ONLY=1` as its `--revision`,
`--draft-model` and `--language-only`, and run the installation
`make install MODEL=...` with the same options prepared in this checkout's
`install/models`:

| Target | Runs |
| --- | --- |
| `verify-models` | the installer's restarts without the Hub, `verify --full`, and the prepared-weight record (`dev/tools/installer_restarts.py`, [Release check](#release-check)) |
| `test-real` | vision parity with the family's fixture in `dev/tests/fixtures/vision-parity/` when the installation serves vision, and the native model runtime oracle |
| `test-http-real` | the HTTP frontend on an isolated server (`dev/tests/smoke_real.py`) |
| `test-agent-real` | the five official clients through `splash serve` (`dev/tests/agent_real.py`), in `AGENT_SCENARIO` `complete` (the default) or `smoke` |
| `test-release-real` | the HTTP smoke and all five clients on one `splash serve` |
| `test-performance-real` | the native decode and partial-prefix benchmark, or with `BASELINE` its ABBA comparison with that build (`dev/benchmarks/backend_regression.py`) |
| `release-check` | one model on this Mac ([Release check](#release-check)) |

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

The source formats load differently: an MLX target is prepared into the packed
layout, a GGUF target into its own layout for the GGUF projection and MoE
kernels, and a package's packed files are mapped as they are.

On a 24 GB Mac, `test-agent-real` stops a client's workflow at macOS's warning
memory pressure, which the smaller GGUF variants such a Mac uses can reach under
an agent's load; `SPLASH_TEST_PRESSURE_STOP=4` stops only at critical pressure,
to observe how the engine sheds its cache. The runtime oracle in `test-real` has
no production memory guard: the prepared weights, and then what the runtime
allocates as it runs, must fit in what macOS has available above its reserve,
so it stops, naming what it needs, while other programs hold that memory. With
only desktop applications open, a 24 GB Mac runs it for those variants.

`make test-engine-cpu` builds the affine source oracle so it cannot break
unnoticed, but no target runs it because it needs real models: after
`make all build/engine-tests/affine-source-oracle`, pass it
`build/splash.metallib`, an installed MLX model's `target` directory and the
matching installed package to compare every prepared byte.

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
`BASELINE`: a checkout whose `build/` holds `splash`, `splash.metallib` and
`engine-tests/backend-benchmark`. `release-check` fails without it. The
baseline must load the model: it is the previous release's build when that
loads the model. Splash 1.0.x loads only Splash packages, so for 1.1, the
first release that loads upstream models, an upstream model's baseline is a
build of the last commit before the change under test; the legacy package can
always be compared with 1.0.2. From a clean checkout:

```sh
make check test-sanitizers                      # once, model-free
make check-native-metal                         # once on each Mac
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
family, so it runs once on each Mac. Per model, `release-check`:

- runs the runtime oracle and, when the installation serves vision, vision
  parity (`test-real`);
- restarts the installer offline, with the Hub unreachable
  (`HF_ENDPOINT=http://127.0.0.1:9`) and with an empty `HF_HUB_CACHE`: each
  restart must start the same assembly within 10 seconds, name the Hub's
  reason on one line when it asked the Hub ([Revisions](#revisions)), and
  download nothing. It then hashes the sources and records in
  `prepared.json` the component and SHA-256 of every prepared-weight entry
  the installation loads (`verify-models`; a legacy package is only hashed);
- runs the HTTP smoke, which for a text-only installation checks the 400s
  instead of images (`test-http-real`);
- compares this build with `BASELINE`, which must have another build
  identity, in ABBA order (`test-performance-real`): output tokens and
  acceptance must be identical (`EXPECT_OUTPUT_CHANGE=1` allows changed
  outputs with acceptance within 0.02), and so must the prepared bytes,
  which a baseline of another preparation identity prepares into a cache of
  its own; decode and prefill GPU time may regress by at most the larger of
  2% and twice the run's own ABBA spread, and a spread above 5% fails as
  inconclusive.

Results go to `build/release/<owner>--<repo>[--VARIANT]/`. Preparation does not
depend on the GPU, so each model's `prepared.json` must be identical on the
two Macs. The unpinned `verify-models`, run after the pinned ones while the
default branch still names the pinned commit, resolves the branch online, and
its unreachable-Hub restart must fall back with the Hub's reason. The agent
clients depend on neither the model's format nor the GPU: run the smoke
scenario once per Mac, with the five clients split between the Macs, and
`AGENT_SCENARIO=complete` for one model when the client integration changed.
Expect about 1.5 hours on an M5 Pro and 2.5 hours on an M3 Max, most of it in
the three 27B comparisons. A laptop can cap its GPU power during a long
comparison and so make it inconclusive; rerun `make test-performance-real` for
that model alone once the Mac has cooled.

When the native wire layout or the status schema changed since the last
release, bump `kProtocolVersion` (`runtime/engine/Protocol.hpp`) and
`PROTOCOL_VERSION` (`server/protocol.py`) together: builds between releases
share a version while its layout changes, and a server refuses an engine of
another version.

### Local benchmarks

Experimental live verification for greedy MoE requests is off by default.
`SPLASH_LIVE_VERIFY_THRESHOLD=0.1 ./splash serve --model <model>` enables it;
values must be in `[0, 1]`, and zero retains the original graph. Each request
keeps its anchor and a contiguous prefix whose cumulative draft confidence
meets the threshold. Inactive rows skip routed and shared experts. Sampled
requests, constrained batches and dense targets retain full verification.
The fixed eight-row layout and existing draft files stay compatible.

To evaluate a threshold, pass `DECODE_PROFILE_ARGS='--live-verify-threshold 0.1'`
to `make benchmark-decode-profile MODEL=<model>`. The profiler reports live
rows, tokens per cycle and fused throughput alongside kernel timings. The
native runtime oracle also accepts `--live-verify-threshold`. Choose a
threshold from end-to-end comparisons on the intended model, device, weight
format and workload; confidence is a heuristic and no speedup is guaranteed.
Dense-row packing needs separate measurements before implementation.

From a source checkout with the model installed, use the native benchmark for
prefill, decode and batch measurements:

```sh
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit BASELINE=/path/to/baseline
```

The first characterizes this build: the decode widths B1-B4 and a 14,096-token
partial-prefix request, three samples each, in
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
version and status schema as this one (the server refuses any other), then run
from the candidate checkout:

```sh
.venv/bin/python -m dev.benchmarks.http_regression \
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M \
  --baseline-binary /path/to/baseline/build/splash \
  --contexts 2048,10000 --samples 5
```

It takes any installed model and, for an upstream one, holds its assembly for
the whole run, so every round serves the same model. It starts isolated servers
in ABBA order, compares matched cold, exact-prefix and decode requests by the
release check's speed rule and prepared bytes (decode by
`metrics.decode_cycle_ms` per output token, so host work between commands
counts; against a baseline that does not report it, both versions by
`metrics.decode_wall_ms` per output token, as each comparison's `metric`
says), and saves `build/release/http-regression.json`. With `--burst N` it
sends N requests of each context at once instead, 16 output tokens each, and
reports each build's replay points lost to the burst
(`replay_state_publication_failures`), its decode rate and advertised context;
`--follow-up`, which a burst run needs to pass, then sends each conversation's
next turn and compares their time to first token and how many resumed at their
replay point.
It does not contact your running server. Use the same power mode and charger,
stop other GPU workloads, and report chip/GPU cores, memory, Splash version,
model revision, actual input/output token counts, and cache hits with results.
Keep cold prefill, cached TTFT and sustained decode separate; a UI token rate
alone does not measure end-to-end agent performance.

For slow tool-bearing requests, the `latency` section of `/status` separates
preparation, tokenization, grammar preparation, native queueing and HTTP TTFT. Grammar preparation
includes construction, compilation/cache lookup and per-request cloning;
it does not include generation-time masks. The `grammar_cache` counters show
whether compiled output grammars are reused. Tool definitions still contribute
tokens to the prompt; saving their JSON alone cannot avoid model prefill.
Existing exact-prefix caching reuses model work while the server remains alive.
Text requests also reuse tokenized history at literal message-end boundaries
when the tokenizer supports independent encoding there. This process-local
cache retains at most four prefixes and 8 MiB of text/token storage; it falls
back to full encoding for other tokenizer pipelines. `/status.tokenizer_cache`
reports its usage. It does not alter prompt text, token IDs or the GPU KV cache.
Server restarts require recomputation. The SSD tier (`--max-cache-disk`, see
[disk cache](#disk-cache)) keeps evicted KV pages and states during a server
session; its temporary files do not survive shutdown.

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

To publish, tag the verified release commit in `incoai/splash` with the
version (no `v` prefix), preserving existing history and tags. Create a GitHub
Release with the runtime archive, bottle, checksum files and `SHA256SUMS`.
Open a pull request in `incoai/homebrew-tap` replacing `Formula/splash.rb`
with `dist/splash.rb`; its URLs must point to the published release assets.
After the tap update merges, verify a fresh install and an upgrade from the
previous release through the public tap, including a real model request and
preservation of user data. Run `brew audit --strict --online incoai/tap/splash`.

Before publishing, verify that default model and draft repositories are
publicly accessible. Check the installed bottle on a supported Mac without
developer tools; building from the source tree is not an installation check.

The runtime package allowlists engine, Python, server and launcher files; tests,
benchmarks and developer documents are excluded. User model links survive
upgrades; downloads remain in the Hugging Face cache.
