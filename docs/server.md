# Server

## Build and run

```sh
git clone https://github.com/incoai/splash.git
cd splash
make -j4
./splash serve --model mlx-community/Qwen3.8-27B-4bit
```

`--model` names an upstream Hugging Face model: an MLX repository
([MLX targets](models.md#mlx-targets)) such as `mlx-community/Qwen3.8-27B-4bit`, an NVFP4
repository of Model Optimizer ([Model Optimizer targets](models.md#model-optimizer-targets))
or compressed-tensors ([Compressed-tensors targets](models.md#compressed-tensors-targets))
such as `nvidia/Qwen3.8-27B-NVFP4` or `unsloth/Qwen3.8-27B-NVFP4`, or a GGUF
repository and variant, `OWNER/REPO:VARIANT`, such as
`unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`.
Splash identifies the model from its own metadata and pairs the DFlash2 draft
trained for it. The first serve sets up Python dependencies and downloads the
model and its draft; each start loads the weights into memory
([Weight loading](internals.md#weight-loading)) and follows the model's revision
([Revisions](#revisions)). Splash packages, the prebuilt format of earlier
releases, are no longer loaded ([Splash packages](models.md#splash-packages)). Public
repositories need no login; private or gated ones need `HF_TOKEN` or `hf auth login`, and a gated
one its terms accepted on its Hub page. Ctrl+C
stops serving, and a second Ctrl+C stops the engine at once; stop before
upgrading.

### Engine restarts and sleep

An engine that fails is restarted at once. If it fails again within 60 s of
starting, the next restart waits 5 s, and each such failure in a row after it
doubles the wait, up to 60 s (failed restarts count); 60 s of service starts
the count anew. Splash stops restarting at once after a failure every restart
would meet again: a setting the engine refuses, a Mac that cannot run Splash,
model files it cannot serve (missing, moved or corrupt), a missing engine
executable, or a start whose GPU commands another process's memory starved
(another Splash server?), which a retry would take from that process. A
restart that failed for want of memory the Mac refused is retried, at most 60 s
apart, for as long as it fails, and neither counts toward the six below nor
starts them anew. For any other cause it stops after the sixth failure in a
row, each within 60 s of starting. Once it stops, requests get 500
`engine_failed` until the server is restarted; its message names the crash
trace when
`SPLASH_CRASH_TRACE=1` recorded one
([crash traces](internals.md#failures-and-crash-traces)). Meanwhile generation requests get
503 `engine_recovering`, whose message names the last failure and whose
`Retry-After` the time until the next bound on the restart's start: when it
should be ready, as long after its spawn as the start before took, then up to
30 s more while it waits for memory the Mac refused ([errors](api.md#errors)). An engine whose
loop leaves a status request unanswered for 30 seconds, while requests are
pending or during a background status refresh, is failed and restarted by the
same rules; so is one that leaves a frame unread for 30 seconds. A request the
engine had started at each of two engine failures in a row that a request may
cause (an execution failure, broken output, or a crash by a fault signal; not
a kill, nor a command short of GPU memory) is refused with 400
`request_quarantined` until the server restarts. A restarted engine is asked
for the first engine's context, with `--max-context` as without, and serves
what its plan holds of it ([Neural Engine prefill](internals.md#neural-engine-prefill)): one
whose plan holds less, as once the Mac's memory budget shrank, serves less, and
the server serves that until a later restart regains it; `/status` reports it
as `capabilities.max_context_tokens`.

Ctrl+C or SIGTERM stops the server: it stops its engine with SIGTERM, which
gives a persistent cache's newest restore points up to 6 s to reach the disk,
and kills an engine still running 15 s later; later SIGTERMs are ignored, and a
second Ctrl+C stops the engine at once. The app stops a server with one signal
and kills it only after 30 s.

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

Completion suggests commands, the models in `install/suggested-models.tsv`
(the list the desktop app offers too) and installed models, a GGUF's
`OWNER/REPO:VARIANT` included, without network access. It also completes local
paths for `--model-dir`, `--draft-model` and `--mmproj`.

## Server configuration

### Server options

`splash serve --help` lists every option with examples:

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | Required | The upstream model, `OWNER/REPO`, or a GGUF's `OWNER/REPO:VARIANT`; with `--model-dir`, the name the API reports. See [upstream model loading](models.md#upstream-model-loading). |
| `--model-dir` | None | Prepare a local GGUF file or GGUF/MLX model directory without the Hub. See [local model files](models.md#local-model-files). |
| `--mmproj` | None | A local GGUF target's vision projector; requires `--model-dir`. |
| `--revision` | Default branch | Select an upstream target branch, tag, or commit. See [revisions](#revisions). |
| `--draft-model` | The family's released DFlash2 draft | A compatible repository or local directory in its place. See [drafts](models.md#drafts). |
| `--language-only` | Off | Skip vision loading; image and PDF input is rejected. See [vision](models.md#vision). |
| `--offline` | Off | Start the installed model without asking the Hub, as `HF_HUB_OFFLINE=1` does. See [revisions](#revisions). |
| `--port` | `SPLASH_PORT` or `8000` | HTTP port. |
| `--host` | `127.0.0.1` | HTTP bind address. See [network and authentication](#network-and-authentication). |
| `--allowed-host` | No extra names | Additional HTTP Host name, e.g. `mymac.local`; repeatable. |
| `--allowed-origin` | No other origin | Origin whose pages may call the API from a browser or webview, e.g. `tauri://localhost`; `'*'` for any; repeatable. |
| `--api-key` | `SPLASH_API_KEY` or none | Require this key, as a bearer token or `x-api-key`, on every route but the chat page, `/health` and `/ready`. |
| `--no-webui` | Off | Disable the chat page. |
| `--max-memory` | Auto | Lower the ceiling on Metal allocations, e.g. `28G`; not combined process RSS. See [memory and context](#memory-and-context). |
| `--max-context` | Auto | Set the context limit, e.g. `100K` or `1M` (K = 1024 tokens, M = 1,000,000), up to the model's limit; past the 256K native window a model with YaRN turns it on. See [memory and context](#memory-and-context). |
| `--kv-format` | `int8` | Target KV storage: `int8` or `bf16`. See [KV cache precision](#kv-cache-precision). |
| `--state-format` | `fp32` | Recurrent state storage: `fp32` or `bf16`. See [state precision](#state-precision). |
| `--ple-memory` | Auto | Qwen3.8-Flash-Next: read the PLE n-gram table from the SSD through a host row cache of this size, e.g. `2G`; `off` loads the whole table. Auto loads it where memory holds it, else streams it through its share of the budget. See [PLE table streaming](#ple-table-streaming). |
| `--expert-memory` | Auto | MoE targets: the memory routed experts keep, e.g. `16G`; they stream through a store of exactly this size, which neither grows nor gives memory back, and read the other experts from the SSD when a step routes them; a size that holds every expert, or `off`, loads them all. Auto loads them all where memory holds them, else gives them what the context leaves, given back down to their floor while requests need it. See [streamed experts](#streamed-experts). |
| `--sparse-kv-memory` | Auto | Qwen3.8-Flash-Next: memory for the KV blocks read back from the SSD for requests whose KV is there, e.g. `1200M`; never below its floor, which auto keeps. See [KV on the SSD](#kv-on-the-ssd). |
| `--idle-release` | `10m` | Time without a request before the engine unwires its memory and frees the weights; the next request restores them. `off` keeps both, though `POST /v1/memory/release` still frees the weights. See [weight loading](internals.md#weight-loading). |
| `--max-cache-disk` | `auto` | SSD quota for cached KV pages and states: `auto` follows the volume, a quarter of its free space beyond a floor of 10 GiB (5% of a volume under 200 GiB), at most 32 GiB; a size such as `16G` is held as given while the volume keeps that floor; `0` turns the cache off. Kept for the session, or across restarts with `--persistent-cache`. See [SSD cache](#ssd-cache). |
| `--max-cache-disk-writes` | `0` (no limit) | The most the SSD cache writes a day, e.g. `200G`. See [SSD cache](#ssd-cache). |
| `--persistent-cache` | Off | Keep the SSD cache across restarts. See [persistent cache](#persistent-cache). |
| `--cache-dir` | `~/Library/Caches/Splash/prefix-cache` | Where `--persistent-cache` keeps its files; needs `--persistent-cache`. |
| `--served-model-name` | None | Additional model name the API accepts; repeatable. See [API model aliases](#api-model-aliases). |
| `--announce-served-name` | Off | Report the first `--served-model-name` in responses; needs `--served-model-name`. |
| `--default-reasoning-effort` | `SPLASH_DEFAULT_REASONING_EFFORT` or the template's | Effort of Chat and Responses requests that set none. See [default reasoning effort](#default-reasoning-effort). |
| `--thinking-token-budget` | `SPLASH_THINKING_TOKEN_BUDGET` or none | Thinking tokens of a request that sets no budget. See [thinking token budget](#thinking-token-budget). |
| `--thinking-end-text` | `SPLASH_THINKING_END_TEXT` or the template's close | Text that ends thinking where a budget runs out, `</think>` its last token; unset, the close the chat template writes after a reasoning. See [thinking token budget](#thinking-token-budget). |
| `--max-request-size` | `128M` | Largest HTTP request body. See [request limits](api.md#request-limits-and-timeouts). |
| `--max-image-pixels` | `4194304` | Maximum resized pixels per image. An image's vision scratch grows with its patches (pixels / 256), to about 600 MiB at the default. |
| `--request-timeout` | None | Time a request may take from its arrival; a request's own `timeout` can only shorten it. |
| `--queue-size` | `32` | Requests admitted at once, running or waiting; more get 503 with `Retry-After`. |
| `--decode-share` | `0.5` | Decode time owed per unit of prefill time while other requests generate. Higher keeps their output faster during a long prompt and slows that prompt; `0` alternates one command each. |
| `--disable-ane` | Off | Prefill on the GPU alone. By default a dense model's long prompts also use the Neural Engine when that is faster. See [Neural Engine prefill](internals.md#neural-engine-prefill). |
| `--verify-rows` | `auto` | The rows of each draft block decode cycles verify: `auto` lets the verify budget choose a MoE target's rows, which is faster but makes its greedy and seeded outputs vary from run to run, and verifies every row of a dense target; `all` verifies every row the output limit leaves on every target, so those outputs repeat when a request runs alone with the same cached prefixes. See [drafts](models.md#drafts) and [sampling](api.md#sampling). |
| `--allow-idle-sleep` | Off | Let the Mac sleep automatically while requests run; by default it stays awake until they finish (the display may still sleep). |

Sizes (`--max-memory`, `--max-cache-disk`, `--max-cache-disk-writes`,
`--max-request-size`, `--ple-memory`, `--expert-memory`, `--sparse-kv-memory`) are a whole number of bytes, or one with a `K`, `M` or
`G` suffix in binary units: `28G` is 28 GiB, and `28GB`, `28GiB` and `28g` say
the same. With a suffix the number may have a fraction, rounded down to whole
bytes: `2.5G` is 2,560 MiB. Token counts (`--max-context`) are a whole number, or one with a `K`
suffix of 1024 tokens or an `M` suffix of a million: `100K` is 102,400 tokens and `1M` is
1,000,000. `--max-memory`, `--max-context`, `--max-cache-disk`,
`--ple-memory`, `--expert-memory` and `--sparse-kv-memory` also take `auto`, their default,
`--ple-memory off` and `--expert-memory off` load the whole table and every expert, `--max-cache-disk 0` turns the SSD cache off,
and `--max-cache-disk-writes 0`, its default, sets no write budget. Durations
(`--idle-release`, `--request-timeout`) are seconds, or a number with an `s`,
`m` or `h` suffix, such as `90s`, `30m` or `1.5h`; `--idle-release off` turns
the timer off, and only `POST /v1/memory/release` then frees the weights.

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
`/health`, `/ready` and browsers' `OPTIONS` preflights stay public. The chat
page asks for the key once the server answers 401 and does not persist it. Use
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
such as `--max-memory 28G`, and the log says when a larger one is ignored. The
automatic plan takes the whole budget, its streamed stores starting with no more
than the host's memory leaves the engine when it starts (below); on a Mac that
also runs other applications, `--max-memory` leaves them room whatever they hold
later: the automatic expert store and PLE row cache are planned smaller with it.
The budget caps Metal allocations, not combined process RSS. The context limit
is the model's native window, 256K tokens for every supported family, when the
budget holds it beside the weights and, unless `--disable-ane`, the
[Neural Engine split](internals.md#neural-engine-prefill) this Mac's calibration of a dense
model takes, and otherwise as much as the budget holds beside them.
`--max-context` sets it, such as `--max-context 100K` (K is 1024 tokens, M a
million), up to what the budget holds without the split: beyond what the split
leaves, the split takes fewer channels or the GPU runs alone, with a warning.
Past the native window a model with YaRN turns it on, up to the model's limit
(below). A larger value than the budget holds stops startup with the most the
model and memory allow.

One plan sizes everything within the budget (`engine::planStreamedMemory`).
Its floors come first: the weights that load, the arenas, the runtime
reserves, the SSD cache's state staging, one lane's state and the KV runway,
the stores of the parts that stream at their smallest (or at the size
`--expert-memory` or `--ple-memory` sets), and for a block-sparse
target the floors of a request whose KV is on the SSD
([KV on the SSD](#kv-on-the-ssd)). The rest goes by priority. Under auto, a MoE
target's routed experts load with their layers where that leaves KV its
allowance, and otherwise stream through a store of all that the allowance and
the PLE table's share leave ([streamed experts](#streamed-experts)); the PLE
table too keeps the allowance first: it loads where the rest holds it beside
the allowance, and otherwise streams through a row cache of what the allowance
leaves, up to its share, at most the rows of every id a 128K-token prompt
stages ([PLE table streaming](#ple-table-streaming)); KV and the states take
what remains. The allowance is the KV of a context, in the fewest pages whose
extents hold it as one request's KV capacity rounds them: a block-sparse
target's first 64K tokens, kept in memory, or `--max-context` where that is
less. A dense target's allowance is `--max-context`, so that its experts load
whole wherever that context fits beside them and stream to hold it where it
does not, or else the first 64K tokens too: its experts load whole only where
they leave that much context, so a larger budget never serves a shorter one.
Nothing in
the plan rests on the host's memory at the start: an automatic expert store's
cap is all that the budget gives it, and it starts at its floor, the chunks
above it released; the memory ledger grows it into the room the host leaves
before the engine is ready, and later as memory allows, as after giving them
back ([streamed experts](#streamed-experts)). The chunks it starts without stay
in the plan (`memory_plan.budget.host_held_back_bytes`), so that what loads
whole, the floors, the context and the KV admission keeps in memory follow the
budget alone, and a relaunch serves its first engine's context whatever the
host holds then. The automatic SSD cache gives its state staging up where that
alone would stream a part, unless the target's KV spills to it. A size of
`--expert-memory` or `--ple-memory` sets its store exactly, charged among the
floors, and the plan holds the context and KV beside it; an expert store of
that size is all its floor. `--sparse-kv-memory` sets its own part. Under auto
the plan holds them beside the expert store at its cap, and what the store
gives back while memory is short only adds room ([streamed
experts](#streamed-experts)). A size below the part's smallest store, a
`--sparse-kv-memory` beyond what the floors leave and a `--ple-memory` size for
a table that cannot stream are refused, and startup logs how each part was
sized. Only a loaded model sizes
the arenas, which rest on its weights' formats and grow where a part
streams: prefill groups for streamed experts, a step's staged rows for a
streamed table. One load gives them for whatever streams, so a start loads
its weights once where the plan without the arenas already loads each part
as the plan with them does, as for a model with no part to stream, and
otherwise loads them again with the stores that plan gives (`Weights loaded
again`): automatic routed experts the first plan streams load at their
smallest store until then; a store `--expert-memory` sizes loads once. An
automatic store sized for a run that spills keeps all of it as its floor.
Where the run then cannot spill (the SSD cache turns off or its KV file does
not open), the weights load once more for a store above its smallest size,
whose floor is then 85% of its slots; a store at its smallest size is all its
floor either way, and the weights do not load again.

A block-sparse target serves the window where its floors hold a request of
that length and the SSD cache has room for its KV, and otherwise the longest
context they hold, or the KV memory holds if that is more. `/status` reports
the bounds under `memory_plan.budget.context_bounds`: the model's `window`,
one request's KV in memory (`resident`), the longest context the floors hold
(`spilled`) and the room the SSD cache has for active KV beside every lane's
state (`disk`), and in `bound` which one sets the limit (`model`, `memory` or
`disk`). Its context therefore follows the free space of the cache's volume
at a first start; a dense target's never does, nor does a block-sparse one's
with `--max-cache-disk 0`. A relaunch keeps its first engine's context
whatever room the volume has left since: where its plan spills, the cache
opens however little room the volume has, and each request's commitment
checks the room as it starts. The plans that size the streamed stores take
the room the volume leaves when the start begins, before the weights load.

On the Qwen families, a `--max-context` past the native window, up to the
1,010,000 tokens Qwen validates these models to, turns on YaRN RoPE scaling,
as Qwen describes for them: the factor is the context over the native window,
so that the stretched window is the context asked for, such as 2 for
`--max-context 512K` and about 3.81 for `1M`. Flint's full layers have no RoPE
to stretch, so its native window is its most, and a longer `--max-context`
stops its startup. Without `--max-context`, or at 256K or less, YaRN stays off, so
answers and the persistent cache are the native window's. The budget must
hold the context: about a million tokens of the 27B's int8 KV on a 64 GB Mac,
more of the 35B's, whose KV per token is a third of the 27B's; where it does
not, startup stops with the most it holds, which a `--max-context` of that
much then serves. The scaling is YaRN's defaults and static: every request,
short or long, takes the same factor, which changes answers at every length,
so pass more than 256K only when a request needs it, as startup says. Each
such context has its own factor, and so its own persistent cache. It scales
the target's attention RoPE alone; the draft keeps its own, which keeps its
acceptance at a million tokens. A long uncached
prompt is slow to prefill, as attention grows with depth. Past the native
window a prefill command takes fewer rows, so that its attention does no more
than a full chunk's at the native window's end, which keeps it well within the
120 s command watchdog on every supported Mac: 512 rows at a million tokens. On an M5 Max the
35B prefills a million tokens in about 45 minutes and then decodes at about a
quarter of its short-context speed, and the 27B takes about half an hour for
512K. The KV of one long request also leaves the cache little room.

The startup summary and `maximum_context_tokens` in `/status` show the effective
context limit, and a block-sparse target's startup line its bounds. `/v1/models` and `/v1/models/{id}` report the same limit as
`max_model_len` and its compatibility alias `context_length`, including model
aliases. Clients can impose a smaller limit. This is a capacity limit, not a
guarantee of a fast first token for a long uncached prompt. If the model cannot
fit, startup prints a memory budget breakdown and stops.

### PLE table streaming

Qwen3.8-Flash-Next's PLE n-gram table (`per_layer_token_embd.weight`, 320M rows
of 160 values, 28.8 GB in IQ4_NL; other row widths come from GGUF metadata)
is read by a handful of rows per token: 16 per position, eight heads each of
the 2- and 3-gram hashes of the token and the tokens before it. Where memory
cannot hold it beside everything else (`--ple-memory auto`, the default,
[memory and context](#memory-and-context)), or with `--ple-memory SIZE`, such
as `--ple-memory 2G`, the table stays in its GGUF on the SSD: no `ple.bin`
image is written, and the weights shrink by the stored table size. Startup
memory requirements also include the cache and Metal staging buffers. The
rows each command gathers are read through a host cache with a SIZE-byte
cache and working-memory reservation, auto's share of the budget: at most
the rows of every id a 128K-token prompt stages, 2M rows, 237 MiB with
IQ4_NL's 90-byte rows. A BF16 or F16 table streams as stored, 320 bytes a
row (the gather rounds F16 values to bf16), so auto, which reserves rows by
count, reserves about 1.9 times the bytes of a Q8_0 table. The cache hits only rows read before, and a request
reads few distinct rows (a 128K-token prompt about 1.1M): caches from
256 MiB, which no run filled, to 2 GiB hit as often, while each GiB more of
the expert store makes a 24 GB Mac decode 11-23% faster, so the experts take
the rest.
The engine refuses a SIZE that holds no row, and a SIZE that holds the
whole table loads it with the weights instead, as `--ple-memory off` does. A
table whose rows the GGUF does not store as the gathers read them (NVFP4,
which the loader rebuilds, or any table of a rotated GGUF) loads with the
weights, and the engine refuses a SIZE for it.
This reservation is not a strict bound on host allocations: prefetch scratch,
excess vector capacity, job metadata, allocator overhead and reader stacks are
additional, and oversized stages grow their scratch. Other models ignore a
size with a warning.

- A step's rows are staged before its PLE layer gathers them: the CPU copies
  each distinct row the step reads once into a small Shared buffer of the
  step's scratch, from the cache or, for the rows it lacks, with uncached
  (`F_NOCACHE`) row reads (90 bytes for IQ4_NL, 170 for Q8_0) by the engine's
  32 SSD readers, which streamed experts share (`SsdReaders`,
  `runtime/model/SsdReaders.hpp`), and the staging thread itself
  (`SparseRowStore`, `runtime/model/SparseRowStore.hpp`). The gather then
  reads them with the table's own kernel, so a row's values are computed
  exactly as from the table.
- A prefill knows its rows on the CPU and stages them before its command is
  submitted. A decode cycle's rows are those of the draft's proposals, which
  only the GPU has: the verify hashes them on the GPU, and a host step of the
  same command stages them between the hash and the gather, keeping the GPU
  awake through it ([Streamed experts](#streamed-experts)).
- A prefill command reads ahead the rows of the prompt tokens that follow its
  rows, as many as a command holds, while it runs; the next command's first
  stage finds them cached or finishes what is still in flight. A read ahead
  runs only while no stage or expert load reads.
- A row enters the cache only after the file was found unchanged since it was
  read. The table is read from the source its metadata was parsed from, so a
  new file at its path changes nothing; a write to that file, a failed read
  (after one retry) or a short file fails the step before the command that
  requires those rows is submitted. The engine stops as it does for a weight
  read error, and the server starts it again. A row a read-ahead fails to
  read is dropped, and the stage that needs it reads it again.
- The cache counts as fixed runtime memory in the plan
  (`streamed_table_bytes`): the governor charges it beside Metal's
  allocations, and the startup memory audit bounds it by the plan and counts
  it toward the hard budget beside Metal's footprint since the weights that
  serve began to load, not a first load that held the table whole before the
  plan with the arenas streamed it. It is not released while idle.
- Once the weights that serve loaded, before the engine is ready, the engine
  writes every page of the cache's rows, ids, recency links and index
  (`SparseRowStore::commit`), so that the cache holds all of its host memory
  from the start: the host memory the engine measures from then on, which
  the start's checks and the expert store's growth go by, counts the whole
  cache rather than the rows read so far. Auto's cache takes its share at
  most; a `--ple-memory` size takes all of it. Stages and prefetches allocate their
  working memory as they run.
- `/status` reports it under `streamed_table`: the table and its bytes, the
  cache's size, capacity, rows and evictions; per use (`prefill`, `decode`)
  its stages, requested and distinct rows, hits, reads, hit rate and stage
  latency (`stage_ms`); the reads' bytes, retries, errors and latency
  (`read_ms`); the prefetches; and `poisoned` with the `failure`. A latency is
  a histogram in milliseconds: `p50`, `p95` and `p99` (within a factor of
  two), `max`, `total` and `samples`.

### Streamed experts

Where memory cannot hold every routed expert of a MoE target beside the
context (`--expert-memory auto`, the default,
[memory and context](#memory-and-context)), or with `--expert-memory SIZE`,
such as `--expert-memory 16G`, its routed experts stay in their files on the
SSD, and a store holds as many of them as fit (`ExpertStore`,
`runtime/model/ExpertStore.hpp`): the weights shrink by the routed experts'
bytes, and the store counts in their place at its cap
(`target_weights_bytes` and `host_held_back_bytes`). Under auto its cap is
all that the plan leaves the experts; it starts at its floor, the chunks
above it released, which it takes as memory allows, and gives memory back
down to its floor while requests or the Mac need it
([memory and context](#memory-and-context)). The floor is every slot when KV
can spill to the SSD cache, else 85% of them. With `--expert-memory SIZE`
the store is exactly SIZE bytes, its slots, staging and tables, rounded to
whole slots and chunks, which startup says where they differ: its cap and
its floor both, so it neither grows nor gives memory back, and the context
and KV take the rest of the budget. A SIZE below every expert
streams them even where memory holds them all. A store holds at least a
layer's experts of each format mix and every expert of a layer whose mix no
other layer shares; the engine refuses a SIZE below that, and `--expert-memory
off` loads every expert, as a SIZE that holds them all does.

- Each MoE layer of a command loads the experts its rows route that are not
  resident: after the layer's routing, a host step of the command claims
  slots for them, recycled by adaptive replacement (ARC: the experts decode
  steps route again stay while the others pass through, and a prompt's
  experts go first), and reads them from their
  files into a staging of a decode cycle's experts on the engine's SSD
  readers, and a repack kernel writes them into their slots before the expert
  passes run. A decode step's load is one such round; a prefill command's
  loads take as many rounds as its rows may route experts, each staging half
  of the staging while the other half is read ahead. The kernels read every
  routed expert through the layer's slot table and the chunks that hold its
  slots, so resident experts, one chunk of every expert, and streamed
  experts run the same kernels.
- An expert's rows a GGUF stores as they are are read straight into the
  staging (`F_NOCACHE`, widened to whole 16 KiB pages, which the kernel reads
  without the buffer cache: an M5 Pro reads 460 KiB pieces from four threads
  at 9.3 GB/s instead of 8.1, with less than half the CPU).
- While the GPU repacks a round or runs a layer, the next round's experts or
  the next block's likely experts are read ahead: those the next block's
  router picks from this block's input in a decode step; after a long prefill
  chunk's last repack, one more host step reads nearly all of the next
  block's into the whole staging. A load takes the read-ahead over without
  waiting for its reads of experts it does not route, and a read ahead of an
  expert no load routes may fail without failing anything. A decode step
  loads and predicts only the experts its lanes' verify rows route
  ([Drafts](models.md#drafts)); the selector writes those rows before the step's load.
- Experts of one format share their slots, held in chunks, each a buffer of
  its own of at most the device's largest buffer (`maxBufferLength`). One
  chunk holds each format's floor: under auto every slot when KV can spill
  to the SSD cache, else 85% of them, or every slot of the store
  `--expert-memory SIZE` sizes;
  the rest lie in chunks of about 1 GiB
  (more where they would take more than 47, since the expert kernels' address
  table holds at most 48 chunks), the last of fewer slots; a floor larger than
  the device's largest buffer takes the elastic chunks it needs too. The kernels
  reach the chunks through their GPU addresses, which the backend writes for
  each expert dispatch as it encodes the command, and makes them resident for
  that dispatch's encoder rather than for every command on the queue
  ([Residency](internals.md#residency-and-kv-extents)). Chunks are few
  because each one a decode encoder makes resident costs GPU time: 36 chunks
  of about 512 MiB cost a streamed decode cycle 2.3% of its GPU time on the
  M5 Max, about 0.065% a chunk. A store whose floor is all of it has no
  elastic chunk. With a floor of 0.85, a format that starts at its cap has
  about one elastic chunk per 6.7 GiB of its slots (0.15 of them, in chunks of
  about 1 GiB), one where they are less, so a 17.2 GiB store, a floor and 3
  elastic chunks, costs about 0.25%; one that starts below its
  cap has as many of the cap's chunks as the slots above its lower floor take,
  at most 7 in a store under 6.7 GiB. A size of one block (layers in formats
  of their own) is one chunk of every expert.
- A decode step's loads keep the GPU awake through their host steps
  ([Host steps](internals.md#host-steps)), so the work after a load starts at once and at
  full clocks rather than after the GPU's wake-up.
- A prefill command runs a group of up to four chunks of 2048 rows layer by
  layer, so each layer's experts are read once for all of them, a prompt's
  with images too. Since every command reads nearly all of each layer's
  experts however few its rows, short prompts ready together share one
  command, and one that would take longer than the 500 ms slice runs in
  passes over the layers, each its own command of about the slice: other
  requests' decode steps run between passes, and requests that arrive
  meanwhile start and join the next command, or, at a higher priority, give
  this one up, which starts again after them. A cancel of a command's only
  request waits for one pass at most, one of several for the command's last. A pass reads its own
  layers' experts, so passes add no reads; a command that starts again reads
  its passes' layers again ([Admission and eviction](internals.md#admission-and-eviction)).
- A load checks its files unchanged after it read them. A read is tried once
  more when it fails. A load that fails (a routed expert cannot be read, or
  a file was written) fails its command, and every later load fails too.
- The chunks above the floor are elastic. Where a request in service needs
  memory the engine's limit refuses, the engine takes what can go in one
  fixed order ([admission and eviction](internals.md#admission-and-eviction)):
  cached prefixes nothing in use holds that leave memory at once, by the
  cache's order; the caches the model rebuilds; the cached prefixes that
  first need a copy on the SSD cache, whose writes it waits for rather than
  take what follows; on a target whose KV can stay on the SSD cache, that
  request's own [spill](#kv-on-the-ssd); then the store's last chunk, one at
  a time; and only then anything in use. A start, optional work and memory
  the host refused never take a chunk. When the host runs short of its 1 GiB
  margin, the engine gives it cached prefixes and the caches the model rebuilds,
  never a chunk; macOS's warning alone takes nothing. A chunk given back keeps
  its experts that fit: they are copied to free slots below it, the experts the
  slots' recycling would recycle first evicted to make room, and later steps
  read the evicted ones from the SSD again. Under macOS's critical pressure the
  engine gives the Mac, each second, the 1 GiB margin and what requests the host
  refused wait for, in the same order, store chunks above the floor among them,
  never the store below its floor.
- The store's cap is a reservation the engine repays: between commands, the
  store tries to take one chunk back, one it gave back or started without,
  its slots empty, once 30 s have passed since its last try and since a
  request's claim last went unmet or took a chunk, a prefill between its
  passes, its own spill or something in use (a request the cache served does
  not count), as long as no request waits for memory and the weights are not
  released. The chunk takes memory `--max-memory` leaves free, which needs
  normal pressure and 1 GiB to spare beyond the chunk; for what that lacks
  the engine evicts cached prefixes that leave memory at once, only when they
  cover it, never one in use, the prefix the newest conversation resumes from or
  those that requests waiting to start would resume from. A chunk such prefixes
  pay for whole comes back under macOS's warning too while the host keeps its
  margin. A prefix that would first need a copy on the SSD cache stays; the
  engine copies such prefixes ahead for what the chunk lacked, so a later try
  finds them free to go. A store a burst shrank thus comes back a chunk per 30 s
  once requests stop going short, even when the burst's KV, cached, fills the
  room. Each chunk given back or taken back logs a line
  (`Expert store gave back`, `Expert store took back`). Elasticity serves
  concurrency, the prefix cache and the rest of the Mac; the advertised context
  is planned with the store at the cap the engine's limit gives it and does not
  change.
- The store's live chunks and staging are released while idle with the
  weights and restored with them: its floor and staging, as the request
  waiting needs them; its elastic chunks come back afterwards as above, a
  chunk at a time ([Weight loading](internals.md#weight-loading)).
- `/status` reports the store under `streamed_experts`: its bytes at its cap
  and at its floor (`cap_bytes`, `floor_bytes`, counted as `--expert-memory`
  counts them), its live chunks and those at the cap (`chunks`,
  `cap_chunks`), the chunks it gave back and took back (`shrinks`,
  `regrows`), its live chunks' bytes (`arena_bytes`) and staging bytes, its
  live slots (the live chunks, bytes and slots are zero while the store is
  released), its experts and those resident;
  per use (`prefill`, `decode`) its loads, the distinct experts they routed,
  those read and of them those read ahead, the experts read ahead whole, the
  hit rate, the bytes read and the host steps' round trips, one per round
  (`host_steps`: `pickup_ms`, `work_ms`, `resume_ms`); and the `failure` that
  stopped it.

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

### State precision

The linear-attention (GDN) layers of the targets carry a recurrent state,
which every lane holds and every cached state keeps, in memory and on the SSD.
Select its storage when starting the server:

```bash
splash serve --model mlx-community/Qwen3.8-27B-4bit --state-format bf16
```

BF16 halves the recurrent state, most of a cached state's bytes, so memory and
the [SSD cache](#ssd-cache) hold nearly twice as many cached states, and a
decode step reads and writes half the state. The kernels compute in FP32
either way and round the state to BF16 where they store it: at the end of each
prefill chunk, at each state a chunk captures inside it (a replay point,
junction or checkpoint), and at the end of each decode step. So where a
prompt's chunks end and its states are captured also changes its state in the
last bits ([sampling](api.md#sampling)). Restart to switch
formats; the SSD cache keeps each format's states apart. Omit `--state-format`
or use `--state-format fp32` for the default, the precision the models'
configurations ask for.

### SSD cache

The SSD cache keeps cached prompts on the SSD when memory runs short. Evicted
KV pages and request states that a later request could resume from are written
to the SSD, and a request that matches them reads them back instead of
computing that part of its prompt again. Without it, a request that runs out of
memory cannot keep its progress checkpoints and replays its prompt after each
suspension. It does not raise a dense target's context limit; a block-sparse
target keeps the KV of its long requests there ([KV on the SSD](#kv-on-the-ssd)).

While memory runs short, the cache also copies each conversation's newest
restore point at least 2048 tokens deep there ahead of need, the state its next
turn resumes from and the KV of the prompt up to it, while its memory copy
stays. It does so while a reclaim has reached cached KV or a state that had no
copy there within the last minute: 10 s after the point was published, once
its request has ended, unless the conversation has gone on from it by then, so
a step of a tool loop that goes on within that time costs no write. A point
that falls due while memory is not short waits, and is copied once memory runs
short if it is still its conversation's newest. When memory runs short again,
that conversation leaves memory at once instead of waiting for its writes.
Without memory pressure the cache writes nothing ahead, so a Mac whose memory
never runs short spends no SSD writes on it. Nothing else is copied ahead: not
the KV a reply adds past the point, which no state resumes from, a shorter
prompt, which is quicker to compute again, or a progress checkpoint. These
copies take only the room the cache has free beside its other copies and the
room spilling requests hold for their KV: they give up no copy, a point that
room cannot hold is not copied, and a full cache gives up the copies of what
memory still holds first. They wait while the cache reads a prompt back, writes
what memory needs now or writes a spilling request's KV, and while a request
waits for that memory, and take at most a quarter of its writes in flight. The
[persistent cache](#persistent-cache) copies every point, whether memory has
run short or not and while its request runs too, and makes room for it.

`--max-cache-disk` sizes it. The default, `auto`, holds at most a quarter of
what its volume has beyond the space it keeps free (10 GiB, or 5% of a volume
smaller than 200 GiB), its own files counted, and at most 32 GiB. A size such
as `16G` is held as given while the volume keeps that space free, so on a
volume of its own the cache may take all but that. Free space is what `df`
reports, without the purgeable space the Finder counts. The engine measures it
twice a second and whenever a write finds the volume full: as other files fill
the volume the cache shrinks, giving up its oldest copies the way a full quota
does, and it grows back as they go. Several servers on one volume each settle
under `auto` at a share that leaves the free space above the floor, and with
sizes take its room in turn, down to the floor. A size keeps no margin above
the floor: while a local snapshot, such as Time Machine's on the startup disk,
keeps the blocks the cache frees, the copies it rewrites can take free space
below the floor, as far as a full volume, before the snapshot goes. Off a local
volume, such as a network share, the quota alone is the limit. `0` turns the
cache off. A full cache keeps the only copy of the prefix the newest
conversation resumes from and of those that requests waiting to start will
resume from: only a spilling request's KV displaces them.

State writes stage through a buffer of one state (64 MiB for 35B, 153 MiB for
27B) that the memory plan sets aside within `--max-memory`; on a 24 GB Mac
serving 27B that costs 4,096 tokens of context. A size sets it aside whatever
it costs, as before, and a quota too small for one state leaves the cache off.
`auto` sets it aside only where the memory plan holds the model and the
requested context beside it, and gives it up to the
[Neural Engine split](internals.md#neural-engine-prefill) where the plan holds the split
only without it; otherwise startup says why the cache is off, and the plan is
as without it. Memory alone decides this, so every start of a dense model on a
Mac advertises the same context, whatever room the disk has; a block-sparse
target's follows the room for its KV
([memory and context](#memory-and-context)). The disk decides only whether
the cache's files open: `auto` opens them only where its limit, a quarter of
what a local volume has beyond the floor, would hold two states and 1 GiB of
KV, on the volume of every place the cache may keep its files. A relaunch
whose plan spills opens temporary files whatever room their volume has, as
the first engine's context it keeps may rest on them. Where no files open,
startup says why the cache is off, and the staging stays unused.

A write that finds the volume full while it shows room beyond the floor met a
limit the engine cannot see, such as a volume quota, or a volume that filled
and freed again before the engine measured it. Once a second write does, the
cache keeps within what it holds, serving and replacing its copies within that
until the server stops, and the log says so once. Off a local volume a second
write that finds the volume full does the same.

`--max-cache-disk-writes SIZE` sets a write budget that spares the drive: the
most the cache writes a day on average, such as `200G`, at most a quarter of a
day's at once. It is off by default, when every write the cache needs goes
ahead. Once set, the budget decides which states are written: while it is
spent, a state a later request could reuse is dropped instead of written, so
is a suspended request's progress checkpoint, and restore points wait for their
copies. It paces writes rather than capping them: the state a conversation
resumes from is written all the same, KV pages are written only for a state, to
keep the chain of one on the SSD reachable or to complete a restore point's
copies, and pass it too, and each server process has a budget of its own that
every start begins with a quarter of a day's of. All of these count against the
budget, so later states pay for them. A quota smaller than the working set can
cause repeated reads and writes.

Startup prints one line: where the files are, the limit and why, the write
budget when one is set, and that `--max-cache-disk 0` turns the cache off.
`/status` reports under `disk` the quota (`capacity_bytes`), what the volume
leaves of it now (`limit_bytes`), `used_bytes`, `volume_available_bytes`, what
holds writes back (`writes_gated`: `no`, `budget`, or `closed` once a file
stopped taking writes), `write_budget_bytes`, the bytes the next state's
write not in use may take now, and `writes_deferred_ms`, how long writes held
back for SSD reads a step waited for. Without a write budget `writes_gated` is
never `budget` and `write_budget_bytes` is 0. It counts the KV pages copied
ahead (`kv_copies`), and under `write_behind` the restore points queued for
their copies, the one being copied among them (`waiting`), copied
(`durable`), gone on from first (`unneeded`) and not copied for want of room
or a file that takes writes (`refused`). An idle engine can report points
queued. Without the persistent cache a point is copied only while a reclaim
has reached cached KV or a state with no copy on the SSD within the last
minute, 10 s after its publication once its request has ended; otherwise it
waits for memory to run short, which may never happen, and is not work the
engine owes. With the persistent cache each point is written once it falls
due, 10 s after its publication at the earliest, and at a clean stop (up to
6 s), so there `waiting` is SSD work still to come and drains by itself.

Restarts recompute everything unless `--persistent-cache` is on (see
[persistent cache](#persistent-cache)); otherwise the cache keeps evicted KV
pages and states for the session only, in temporary files whose space returns
when the engine exits.

### Persistent cache

`--persistent-cache` keeps the SSD cache across restarts of the same model, so a
restart, an upgrade or a crash keeps conversations' prefixes; an upgrade that
changes what the namespace digests (below), such as a weight image's size,
starts its cache cold once, and one that only lays out images of the same sizes
and headers anew keeps it. It follows
`--max-cache-disk`, which sizes it as it does the SSD cache; it needs the SSD
cache, so a start with `--persistent-cache` and `--max-cache-disk 0` is
refused.

Files live in `~/Library/Caches/Splash/prefix-cache/<namespace>/` (mode 0700;
`--cache-dir` sets the root, which the engine keeps out of Time Machine when it
creates it). The namespace digests the weights' manifest, the KV and state
layouts, the YaRN factor (of a `--max-context` past 256K) and the cache format, so no model
reads another's cache. The manifest names each image's path, declared size and
header, and the SHA-256 of the assembly's `model.json`, which records the
resolved sources and each linked file's path, size, times and digest
([Installation](internals.md#installation)). One engine holds a directory at a time;
another waits 20 s, then uses temporary files, as
`auto` does when the root's volume is not local or leaves the cache too little
room.
One limit, the quota or what the volume leaves, bounds the whole root, and this
cache may use all of it: other models' caches nobody holds count against it,
and while the root holds more, as this cache grows or the volume fills, those
caches are removed, the least recently used first, before this one gives up a
copy of its own. A cache nobody holds when a running server removes it may be
the one another server is starting to open; that start then begins cold.
Every start, with `--persistent-cache` or without it, removes the namespaces
under its root (`--cache-dir`, which needs `--persistent-cache`, else the
default) that nobody used for 14 days, so a cache left behind in the default
root when the persistent cache is turned off goes too; one under another
`--cache-dir` goes at the next start that names that root. Only a directory
named as a namespace, 32 hex digits, counts as a cache: nothing else under the
root is touched.

The restore points the SSD cache copies ahead ([SSD cache](#ssd-cache)) are
what the next start takes back. The persistent cache copies every point,
whether memory has run short or not, and while its request still runs too, so
that a start after a crash takes it back, and makes room for it as for any
copy: a full quota keeps restore points whole, dropping copies no point needs
first, then the oldest point. A point larger than what the cache may hold now,
its quota or what the volume leaves it, is not written. A request that computes
blocks of a point's chain again leaves their copies to the point: they count as
its copies, not as the request's KV. One whose KV spills writes its own over
them before its pages spill.

A start takes back every recorded point whose chain is whole. A clean stop
writes the waiting points (newest first, up to 6 s, whatever the write budget)
and syncs the files. After a crash the next start takes back what reached the
disk, each state with the class it had last: a checkpoint a request went on to
reuse as an ordinary state comes back as one. Every start serves its first
minute on probation: after two starts in a row fail within it, or one whose
engine reports its own failure, the next start begins empty. `/status` reports
all this under `disk` (`persistent`, `taken_back`).

### KV on the SSD

Qwen3.8-Flash-Next attends only the blocks its index selects
([sparse attention](internals.md#sparse-attention)), so the KV of a long request can stay
on the SSD cache while memory keeps each page's index and the blocks decode
steps read. No option turns this on: it needs only the
[SSD cache](#ssd-cache), and with `--max-cache-disk 0` the target's context is
what memory holds, as a dense target's is.

- Each attempt to start a request decides it; a resume keeps it. A request
  whose KV, its prompt and `min(max_tokens, 4096)` tokens of output, is 2,048
  tokens or fewer, or takes no more in memory than spilling it would at its
  peak, keeps its KV in memory, and so does one whose KV, beside what the
  requests in service are projected to grow into, fits the memory free now
  with what a start may take (the cached prefixes nothing in use holds and
  the caches the model rebuilds, never the
  [expert store](#streamed-experts)'s chunks), counted in whole extents with
  the states its prefill holds; any other spills. A request
  that cannot spill, the SSD cache closed or its quota short, or whose KV
  fits in the pages a spilling request keeps whole anyway, keeps its KV in
  memory.
- A decoding request in memory whose KV cannot grow spills from then on
  rather than yield its lane or fail, where that covers what it lacks: what
  spilling the pages only it holds frees, past its first 2,048 tokens and
  its last 64 pages, less the hot block cache's floor its reads need
  (390 MiB), which takes a request of about 46K tokens or more. Its pages whose copies landed before,
  as a prefix restored from the SSD cache has them, spill at once, and it
  waits while the others' copies land. From then on it is a spilling request
  like any other. No other request and no shortage of the host's makes a
  request spill, and a request in prefill does not. When other memory is
  short, a state, a spilling request's reads, or the host's, what can go
  goes in a fixed order: a cached prefix, the caches the model rebuilds,
  then, for a request in service, a chunk of the store above its floor, of
  which a target that can spill has none.
- A spilling request writes each committed page once to the SSD cache. Once
  the copy has landed, the page keeps only its index, 24 KiB of the int8
  page's 414 KiB. Alone at the memory wall, it waits while its writes land
  rather than fail. Its first 2,048 tokens, which rows with less history read
  whole, its last 64 pages and up to two prefill groups' pages still being
  written stay whole. The memory the pages free leaves with the extents they
  empty, all of them once every spilling request decodes, but one while a
  spilling prompt still grows into it or the host holds growth back.
- A decode step reads, after each attention layer selects its blocks, the
  selected blocks not in memory into one hot block cache that all requests
  share, a page's in one read from its first missed block to its last on the
  32 SSD readers, and attends them there.
  The cache holds at least, for n decoding requests, n + 1 times the blocks a
  request's 8 rows select, 8,192 blocks a layer (390 MiB) for one, taken
  whole as a spilling request starts decoding; `--sparse-kv-memory` raises
  its target, toward which it grows only while a request holds a floor. A
  decoding request's floor goes only with the request. While no request
  reads above the floor, the cache's memory, up to 975 MiB after the last
  spilling request ends, is among the first to go when memory is short, as
  a cache the model rebuilds. It shares the readers' memory
  with the history staging below, which the plan sets aside the larger of,
  so a target within the staging takes nothing from the expert store.
- A prefill streams each attention layer's history from the SSD cache into
  staging, a layer's pages read whole (36 KiB each) while the previous layer
  runs. A second buffer, which lets the next layer's reads overlap this
  layer's attention, is planned where the readers' memory holds it already,
  or where what the streamed stores and the first 64K tokens of KV leave
  holds it: where a store binds that is nothing, and the store keeps the
  room; otherwise the buffer takes it from the KV kept resident beyond them.
  A prefill whose history is on the SSD runs alone, and with others waiting
  it takes the fewest rows whose work covers its reads. The staging holds
  nothing between prompts: a decoding request's floor of the hot block cache
  takes its place, and the next prompt that streams allocates it again.
- The next turn finds its pages' index in memory, or reads it back, and
  continues without prefilling the conversation again; with
  `--persistent-cache`, after a restart too.
- The `--max-cache-disk` quota holds spilled KV beside the cached copies.
  Spilled KV of requests in service may take all of the room its volume has
  beyond the floor, while `auto` keeps cached copies to a quarter of it: a
  request of 1,010,000 tokens takes 13.5 GB of slots, about 24 GiB free with
  the floor.
  A spilling request commits slots for its KV when admitted; one the cache
  cannot hold even alone is refused, one that other requests' commitments
  keep out waits, and one whose prompt fits but not its output ends at the
  bound it was admitted with, which Messages reports as
  `model_context_window_exceeded`. A request that cannot spill, the cache
  short of room, its file closed or its write budget spent, keeps its KV in
  memory where it fits the context memory holds of one request
  (`context_bounds.resident`). Past what it committed, as when its output
  outgrows the 4,096 tokens admission counted, a request commits each
  page's slot as it writes it, while the room lasts; its pages stay in
  memory beyond it, and the other requests' committed pages are still
  written.
- The KV a spilling request writes is in use and always written, so one long
  request may overdraw `--max-cache-disk-writes` once. Until the budget
  covers what spilling requests were promised, a new request that would
  spill keeps its KV in memory where it fits and is refused otherwise
  (`kv_write_budget`, 503 with a `Retry-After` of the seconds until the
  budget covers them again as it refills).
- A block that cannot be read back fails only the requests that read it, as
  `kv_unreadable`, which they may retry; the copy is dropped.

A request's outputs are the same whether its KV spills or not, bit for bit,
for the same schedule and cache hits. `/status` reports the plan's floors
under `memory_plan.budget.kv_spill` and what spilling does under `kv_spill`
([status](api.md#status-readiness-and-metrics)). Startup logs one line: the context
and what bounds it, the KV a request may keep in memory, the hot cache and
its floor, the staging buffers and the room the SSD cache has for active KV;
or why long requests keep their KV in memory.

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
repository's default branch ([Drafts](models.md#drafts)). A new commit downloads only
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
files, and records there the commit of a default branch, as `hf download`
does (`refs/main`). [Installation](internals.md#installation) gives the mechanics.

### Model cache

To download new models to another disk, set the cache location before serving:

```sh
HF_HUB_CACHE=/Volumes/Models/huggingface splash serve --model mlx-community/Qwen3.8-27B-4bit
```

`HF_HUB_CACHE` selects the Hugging Face download cache. Alternatively, set
`HF_HOME` to relocate the Hugging Face home directory, including its default
`hub` cache. Model links stay in Splash's data directory; existing downloads
are not moved, and an installed model keeps reading its files where they are,
also when a new draft has Splash assemble it again.

### Files Splash writes

| Path | Holds |
| --- | --- |
| `~/Library/Application Support/Splash` | The data directory of a packaged install: `models/`, the installed models' links into the Hugging Face cache and the metadata derived from GGUFs; `runtime/`, the locks of running servers; and `thinking.key`, which encrypts the Messages reasoning a response hides. A source checkout keeps the first two in `install/models` and `build/runtime`. |
| Hugging Face cache (`HF_HUB_CACHE`, by default `~/.cache/huggingface/hub`) | Model and draft downloads, with Splash's pins under `refs/splash` and the commit of a default branch as `refs/main` ([revisions](#revisions)). |
| `~/Library/Caches/Splash/prefix-cache` (`--cache-dir`) | The [persistent cache](#persistent-cache), with `--persistent-cache` only, within the [SSD cache](#ssd-cache)'s limit. Every start removes the caches there that nobody used for 14 days. |
| `$TMPDIR/splash-cache-*` | The [SSD cache](#ssd-cache)'s files, on by default, without `--persistent-cache`, or when another process holds the persistent cache's directory or it cannot be used: one of KV pages and one of states, together at most about the SSD cache's limit: under `auto` a quarter of the volume's free space beyond its floor, up to 32 GiB, or the size `--max-cache-disk` sets while the volume keeps its floor. Each is unlinked as it is created, so no directory lists it and macOS's storage settings count it as System Data, and its space returns when the engine exits, even after a crash. `--cache-dir` does not move them; they follow `TMPDIR`. |
| `~/Library/Logs/Splash/crash` | [Crash traces](internals.md#failures-and-crash-traces), with `SPLASH_CRASH_TRACE=1` only. |
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
rejects `none` renders by `enable_thinking` alone while thinking is off. A
template that cannot give the thinking a request asks for, one that always or
never thinks, renders as it can, and the request reasons as its prompt does. A
template that refuses a request's messages returns 400 with its reason.

Flint's template has no thinking switch: its generation prompt is
`<|assistant|>` alone, whatever the effort or `enable_thinking`, and the model
decides each turn whether to reason. Both reach its template and change
nothing; a reply whose output opens with `<think>` returns that block as its
reasoning, and any other reply is all answer
([chat formats](models.md#chat-formats)).

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --default-reasoning-effort none
```

`/v1/models` reports, as `reasoning`, what efforts do to the model's default
template, rendered as at startup: `thinks`, whether a request without an effort
reasons; `efforts`, the efforts besides `none` that the template renders, and
differently, each named by the one of those rendering alike that no alias
stands in for (`low`, `medium` and `xhigh` for Qwen3.8; none for Qwen3.6, whose
template switches reasoning on and off alike at every effort); and `default`,
the one of them a request without an effort renders like,
`--default-reasoning-effort` in place of the template's (`xhigh` for `high` on
Qwen3.8). For Flint, whose model decides, `thinks` is `null` and `efforts`
empty.

`/apply-template` uses the same default. Anthropic `thinking` keeps its protocol
semantics (off when omitted) unless the request's `chat_template_kwargs` set
`enable_thinking`; judgment endpoints always disable thinking. The
built-in chat page offers Auto, the efforts `/v1/models` reports and Off, and
no menu for a model that decides; it sends no effort unless the user picks one.

The console's Started line shows how each chat request reasons: the effort it
sent, or `absent` and the default it took, its `enable_thinking` when it set
one, and whether it thinks, as in
`Started · input 2,473 · cached 0 · effort absent → low · thinking on`.

A Chat or Messages request's `chat_template_kwargs` are passed to the template
as variables and outrank the effort, and in Messages `thinking`, so
`{"enable_thinking": false}` turns reasoning off and reaches the template as
`reasoning_effort` `none` too, as effort `none` does, and `true` turns it on,
at the template's default effort where the effort is `none`. `enable_thinking`
must be a boolean; null leaves it to the effort. They cannot set what Splash
passes itself, such as `tools` or `add_generation_prompt`.

## Thinking token budget

A reasoning model can spend its whole output limit thinking and return no
answer. A thinking token budget caps the thinking: once a request has selected
that many tokens in its thinking block, the engine selects the thinking end,
which closes the block with `</think>`, and the model answers within the output
limit that remains. Its `reasoning_tokens` then equal the budget plus the end's
tokens before `</think>`. Where a grammar constrains the output, as for
structured output and in the cases [tool calls](api.md#tool-calls) lists, thinking can
end only between whole characters: where the budget's last token holds part of
a character, as a byte of a rare Chinese character or of an emoji can, the
request thinks on until the character is complete, a token or a few, and its
`reasoning_tokens` count those tokens too.

- Chat Completions and Responses take `thinking_token_budget`, a non-negative
  integer; `-1` sets no cap, and an absent or null value takes the server's.
- Messages caps `{"type": "enabled"}` thinking at its required `budget_tokens`,
  an integer of at least 1024, as in Anthropic's API; one at or above
  `max_tokens`, which Anthropic's API allows with interleaved thinking, sets no
  cap. Adaptive thinking takes the server's budget.
- `--thinking-token-budget N` (or `SPLASH_THINKING_TOKEN_BUDGET`) is the
  server's budget; unset, thinking has no cap.
- `--thinking-end-text TEXT` (or `SPLASH_THINKING_END_TEXT`) is the server's
  thinking end: text without the template's special tokens, then `</think>` as
  its own token, at most 64 tokens in all. The server checks the text with the
  model's tokenizer when it starts, and stops with an error on one that breaks
  these rules. Unset, the end is the close the chat template writes after a
  reasoning, which the server probes at startup: a line break then `</think>`
  for Qwen's templates, `</think>` alone for Flint's. The next request then
  renders the close of a reply whose thinking the budget ended as the tokens
  the budget selected, so that its prompt keeps to the reply's tokens through
  the cut and can read the reply's cached KV past it; an end that closes
  otherwise, such as `</think>` alone on Qwen's templates, departs from the
  reply at its close. Where the cut follows a space or a line break, the prompt
  still departs there: Qwen's templates trim them from the reasoning.

A model whose thinking is cut mid-thought may go on planning in its answer, and
may close that with a `</think>` of its own. A chat answer drops every `</think>`
after the reasoning's end, the budget's or the model's own, and with thinking
off: only the tag goes, the newlines around it stay, and a tool call's arguments
and a structured answer's JSON keep theirs. A text completion keeps its text as
written. An end text that turns to the answer first helps it answer at once,
such as this one adapted from the sentence Qwen suggests for its models: a blank
line sets it apart from the cut thought, and the model writes the one after
`</think>` itself.

```sh
splash serve --model mlx-community/Qwen3.8-27B-4bit --thinking-token-budget 12000 \
  --thinking-end-text $'\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n</think>'
```

A budget applies only where generation starts in a thinking block, and one the
output limit reaches first changes nothing; the output limit can also cut the
end short. Where the context leaves a request less room than its output limit,
the budget keeps its share of that room, and the answer the rest, a limit past
the budget plus the room counting as the budget plus the room: a budget of
24,576 under a limit of 32,768 with 20,000 tokens of context left ends thinking
at 15,000, and one of 8,192 under a limit of 2,147,483,647 with 122,072 left at
7,676. A model that opens its thinking itself, as Flint does, never starts
in one, and a budget caps nothing for it. Where the output may call a tool, a
call's opening ends thinking too, so the end never lands inside a call. The
engine takes any `<tool_call>` token for that opening, while the reply's
reasoning ends only where `<tool_call>` is followed by `\n<function=`: a stray
`<tool_call>` in the thinking ends the budget's count while the reasoning goes
on, and the request's `reasoning_tokens` can then exceed the budget. Only the
end is forced, and a close the model selects itself before the budget ends the
count. Text completions have no reasoning split and refuse a budget with 400.

## Agent launchers

Agents must already be installed; `splash claude|opencode|codex|hermes|pi`
connects to the running server. Arguments pass through, for example
`splash codex resume --last` or `splash hermes chat -q "Hello"`. `splash`
reaches the server on loopback without a proxy. Where the environment names a
proxy (`http_proxy`, `https_proxy` or `all_proxy`, in either case), it adds
`127.0.0.1`, `localhost` and `::1` to the agent's `NO_PROXY` and `no_proxy`,
keeping the hosts already there and leaving `*` as it is, so no proxy receives
the server's key. A proxy set only in macOS system settings is left to that
setting's own exceptions.

`splash pi` adds a `splash` provider to Pi's `models.json` (`splash-<port>` for
a server on another port), preserving other providers, settings and sessions.
Where every chat template of the server renders a later system message in
place (`/status` `chat_template.later_system`, [tokenizer and chat
templates](models.md#tokenizer-and-chat-templates)), the provider's model says so to
Pi, which otherwise folds the system messages it adds as a session goes on
into the first: the prompt then keeps its start, and the server its cache.
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
named), which nothing reads now and which can be deleted. While the default
directory remains, the native runtime says so when it starts, with its size.
