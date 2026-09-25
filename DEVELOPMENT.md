# Development

Use Apple M2 or newer with macOS 26.4+, Xcode 26 or newer, Python 3.12–3.14,
and a Metal 4 compiler with `uint4b_format` tensor support.
The macOS 26.2 SDK can compile the host code, but Xcode 26.2's default Metal
component cannot compile the kernels; select a newer Metal toolchain when
using that SDK. Packaged users need none of these development tools.

M2 compatibility was validated on an M2 Max (64 GB, macOS 26.6.2) with
`make -j4 all test-engine-cpu test-engine-metal`, including Metal shader
validation. The runtime still requires placement-sparse support and checks it
with an actual map/unmap probe. This validation covers the model-free native
suite; it does not establish full-model generation quality or M2 throughput.

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

HTTP request bodies are limited to 128 MiB; `serve --max-request-size 256M`
overrides this. Concurrent input bytes share a budget of at least 512 MiB
(or twice the request limit), including retained generation inputs. This is
an input-byte budget, not a process RSS limit: large ASCII/base64 strings can
use roughly twice their encoded size during JSON parsing alone. Decoded images
and object-heavy JSON need additional memory. Oversized requests return 413;
exhausted ingress capacity returns 503. Image and model context limits apply
independently.
Stored Responses history is charged before decoding. Uploads allow 30 seconds
of inactivity; total upload time is limited to 30 seconds plus the body size
at 512 KiB/s (286 seconds for 128 MiB), capped by the overall request deadline.
Timed-out uploads return 408 and release their input reservation.
`/status` reports `http.request_body_bytes` and `http.max_request_bytes`.

Source `install/completions/splash.bash` for Bash or
`install/completions/_splash` for Zsh after `compinit`. Completion suggests
commands, bundled official model IDs and installed models without network access.

## Server configuration

The default listener is `127.0.0.1:8000`. To accept LAN connections:

```sh
splash serve --model incoai/Qwen3.8-27B-Splash --host 0.0.0.0 --api-key YOUR_KEY
```

Connect to the server's LAN IP. `--host` selects the IPv4 bind address;
`--allowed-host NAME` accepts an additional HTTP Host name, such as a custom DNS
name or proxy hostname. It does not change the listener or allowlist client IPs.

Use `--port 8001` or set `SPLASH_PORT=8001` to select another port. Set the same
`SPLASH_PORT` in the local agent shell. Separate ports allow separate servers;
their memory limits are independent. The packaged agent launchers connect to
loopback, so use a listener that includes loopback when launching agents locally.

## API model aliases

Repeat `--served-model-name NAME` to accept additional API model IDs. The full
`--model OWNER/REPO` still selects the package. `/v1/models` lists that ID first,
followed by unique aliases; each alias's `root` identifies the loaded package.
Generation and scoring responses always report the real package ID, even when
requested through an alias. The model list and lookup support both names.

```sh
splash serve --model incoai/Qwen3.8-27B-Splash --served-model-name local-qwen
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
splash serve --model incoai/Qwen3.8-27B-Splash --default-reasoning-effort none
```

`/apply-template` uses the same default. Anthropic `thinking` keeps its protocol
semantics (off when omitted); judgment endpoints always disable thinking.

## Model cache

To download new models to another disk, set the cache location before serving:

```sh
HF_HUB_CACHE=/Volumes/Models/huggingface splash serve --model incoai/Qwen3.8-27B-Splash
```

`HF_HUB_CACHE` selects the Hugging Face download cache. Alternatively, set
`HF_HOME` to relocate the Hugging Face home directory, including its default
`hub` cache. Model links and agent sessions stay in Splash's data directory;
existing downloads are not moved. Prepared weights have their own cache,
which this does not move ([Weight preparation](#weight-preparation)).

## Model packages

Packages contain `manifest.json`, packed `target/`, `draft/`, `vision/` weights
and `tokenizer/`. The manifest lists artifact paths, sizes and SHA-256 hashes.
Dense packages use schema 3 / `splash-packed-q4`; MoE uses schema 4 /
`splash-packed-q4-moe`.

These formats encode Qwen3.8-27B and Qwen3.6-35B-A3B layouts. Compatible community
fine-tunes may use any nonempty manifest model name. Native loading validates
geometry, tensor sizes, binary headers, tokenizer and target/draft compatibility.
New architectures require engine support; ordinary HF weights need conversion.

### Vision

An MLX model's `vision_tower.*` tensors and a GGUF `mmproj` projector both
prepare the packed `vision/model.bin` layout, which the one BF16 vision
operator reads: BF16 tensors are copied, and F32 or F16 tensors are
converted under the exact-BF16 rule of [weight preparation](#weight-preparation).
Unsloth's mmproj stores its 1-D tensors, patch embedding and position table as
F32, all of them BF16-exact, and prepares byte-identical to the packed file.
Quantized MLX towers, deepstack projectors and mmproj tensors the tower does not
use are rejected.

### Weight preparation

Source adapters write a model's target and vision tensors into prepared files:
an MLX target and any vision tower into the packed layouts of Splash packages,
which run the same kernels, and a GGUF target into the `MDGG0001` layout of the
GGUF kernels. Each adapter is a loader, which validates the source's metadata
and plans its files, and a writer: `AffineTargetLoader` (`AffineTarget.cpp`)
and `AffinePreparation` for an MLX target, `GgufTargetLoader`
(`GgufTarget.cpp`, planned by `GgufImage.cpp`) and `GgufPreparation` for a GGUF
target, `VisionLoader` and `VisionPreparation` for an MLX or GGUF vision tower.
They open their files through `PreparedFiles`, the `PreparedWeights` cache with
the load's guards. `AffinePreparation` reorders codes, scales and biases into
256-row tiles without requantization and computes GDN decay as
`float(-exp(double(A_log)))`, which may differ by one float ULP in this small
vector from packages produced with MLX's float exponential. `GgufPreparation`
repacks GGUF blocks ([GGUF targets](#gguf-targets)).

Preparation never rounds a weight. A tensor it converts to BF16 (vision tensors
stored as F32 or F16, a GGUF's convolution taps and time-step bias) must be
exactly representable in BF16; otherwise preparation fails, naming the tensor
and, for a vision tensor, its file.

The cache is `~/Library/Caches/Splash/weights`, or the directory
`SPLASH_WEIGHT_CACHE` names; nothing else selects it. It holds an additional
copy of the weights about the model's size, its prepared target and vision
tensors. Preparing needs that much free disk space plus a 2 GiB reserve: before
anything is written, the factory (`ModelFactory.cpp`) constructs the vision
tower's loader (`planVisionLoader`, which the vision encoder test shares) and
the target's, and checks the space of every missing file they plan, plus the
reserve, once. Uninstalling a model does not delete possibly
shared prepared weights. With Splash stopped, entry directories can be deleted;
deleting the whole cache causes preparation at the next load.

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
preparation identity wrote, and entries of earlier Splash versions prepared from
the same source path. Entries of other sources or revisions, which
installations may share, stay. Removal happens under the converter lock, so no
entry being written is touched, and a running process keeps the files it has
mapped until it unmaps them. Two builds of different preparation identities
sharing one cache supersede each other's entries at every start; give a
development build its own `SPLASH_WEIGHT_CACHE`.

One writer per cache serializes conversion; complete cache hits bypass this
lock. Each output's disk space is preallocated before writing. Interruption,
disk-full errors and memory-pressure rejection cannot publish partial files;
concurrent external disk activity can still exhaust the volume. Retrying removes
abandoned writes under the converter lock and reuses previously completed
files, which are read-only. Cold preparation reports each artifact's progress.

Cold source hashing and output validation stream bounded buffers. Unchanged
files reuse a digest proof tied to device, inode, size, birth time, mtime and
ctime; a write or replacement invalidates it. This is not a full disk scrub on
every startup. Preparation uses uncached destination I/O. Every adapter sizes
its conversion steps to one staging bound, input and output together, of
32 MiB (`kWeightPreparationStagingBytes`), whatever the tensor, layer or expert
count, inside a 64 MiB admission reserve that also covers source metadata.
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
too). File
backing does not make Metal-resident pages reclaimable: residency wires them
until released. macOS page cache, driver allocations and other applications
still affect memory pressure and swap.

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

### GGUF targets

The native loader requires every tensor it reads to have a type it accepts for
that tensor and lists every unsupported tensor in one error:

- linears and experts: Q4_K, Q5_K, Q6_K, Q3_K, IQ4_XS, IQ4_NL, Q8_0 or IQ3_S;
- token embeddings: Q4_K, Q6_K or Q8_0;
- norms, the MoE router and shared-expert scalar gate, and the GDN
  convolution, decay and time-step bias: F32;
- GDN alpha and beta: both Q8_0 or both F32.

Of Unsloth's files in September 2026 that covers, for Qwen3.8-27B, UD-Q4_K_M
and every larger file but Q4_1, UD-Q8_K_XL and BF16, and for Qwen3.6-35B-A3B,
UD-IQ4_XS and every larger file but MXFP4_MOE, UD-Q8_K_XL and BF16. The smaller
files need IQ3_XXS, IQ2, IQ1 or Q2_K kernels and the others Q4_0/Q4_1, MXFP4 or
BF16 ones, which do not exist yet.

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
`decode_linear_gguf_prepare` when none did. On Apple10 (M5) the staged tile
(`LinearTile::GgufStaged`) runs the kernels of `runtime/metal/kernels/shared/gguf_linear.metal`,
which dequantize each weight once to half in threadgroup memory (`kernels/common/gguf_staged.h`)
for MPP `matmul2d`, the neural accelerator's path, on bf16 activations; a step of three request
lanes runs the 32-row tile over four lanes of storage. Prefill runs the staged kernels on both
families, chunks of up to 32 rows on the decode tiles. Every projection splits its K across
threadgroups by one rule (`decodeSplits`: each tile's tiers of threadgroups per core and inputs
per partition, from measured occupancy) that does not depend on the batch width. The MoE experts
(`runtime/ops/MoE.cpp`) run the same numerics per family over the grouped rows: the register form
in `linear_gguf_sgmatrix.metal`, the staged one in `kernels/shared/moe_gguf.metal`. The float
router and alpha/beta projections run in `kernels/shared/gguf_float.metal`, and the token rows are
gathered by one template in `kernels/shared/embedding.metal`. These plans are fixed rules of GPU
family, core count and shape: `Linear::setChoices` and `ExecutionPlans::install` reject tuned
entries for block projections and GGUF MoE blocks.

A GGUF kernel of one quantized tensor names its epilogue last: `a` none, `r` residual, `g` the
up pass with the silu gate. The staged ones are `gguf_decode_<format>_m<rows>_<e>` and
`gguf_prefill_<format>_<e>`, the register ones `gguf_decode_sg_<format>_l<lanes>_<e>`, and the
experts `moe_expert_gguf_m<rows>_<e>` and `moe_expert_gguf_sg_<e>`; the fused projections run
`gguf_decode_fused_m<rows>` and `gguf_decode_sg_fused_l<lanes>`. The norm, GDN and
attention-gate variants that also write a register kernel's input table carry `table64` (the
affine Q4 kernel's) or `table16` (the GGUF one's) in their names. The epilogue kinds and SiLU of
both GGUF families are in `kernels/common/gguf_tile.h`, and the MMA helpers every register
kernel uses, affine, GGUF or fp32, in `kernels/common/sgmatrix.h`.

The ABIs are in `runtime/metal/abi/Gguf.h`, which also defines the tile geometry the kernels
and `LinearGguf.cpp` share, and `MoE.h`; the image formats in
`runtime/metal/abi/QuantFormat.h`, their decoding in `runtime/metal/kernels/common/quant_formats.h`
and the decode-only value tables in `runtime/metal/abi/QuantTables.h`, which no prepared byte
depends on; weight preparation's repack ABI is `runtime/metal/abi/GgufRepack.h`.

The tests' CPU reference (`dev/tests/engine/GgufFormatReference.hpp`) must reproduce the golden
hashes of upstream GGML's dequantization (llama.cpp 7ab4ee7) in `gguf-reference`, and
`gguf-planner` checks the planner's plans; both run in `make test-engine-cpu`.
`make test-engine-metal` runs `gguf-preparation`, which checks every format's planes, as the
production executor and its `gguf_repack` kernel prepare them, bitwise against the reference,
the prepared alpha/beta, norm, convolution and router bytes and the golden images; then
`gguf-dequant`, the staged tile's dequantizer, built with the production Metal flags, against
the half rounding of every reference weight; `gguf-projection`, every GGUF projection through
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
decode tiles at one to four lanes and every K split, and marks the device policy's pick;
`make benchmark-gguf-moe` times one MoE layer at the 35B shape, GGUF against affine Q4, on the
device's plans and the other GGUF tile.

## Code and API boundaries

- `server/`: OpenAI Chat/Responses, Anthropic Messages/count_tokens, typed
  judgments, templates, streaming and input processing. No client-version branches.
- `runtime/engine/`: scheduling, memory admission and reusable request state.
- `runtime/model/`: target/draft execution and vision.
- `runtime/ops/` and `runtime/metal/`: operators and Metal kernels.
- `install/`: launcher, client configuration and model installation.
- `dev/`: maintained tests, benchmarks and build/release tools.

Within `server/`, `server.py` owns HTTP and startup; `frontend.py` prepares
requests and history; `backend.py` owns native request lifecycles. `judgments.py`
owns finite-choice prompts, validation and typed answer math. `output.py` parses
generated text for both streaming and complete responses, and `constraints.py`
compiles token constraints. `make architecture-check` prevents lower layers from
importing the HTTP entry module.

Tools can be combined with structured answers. Tool argument framing resolves
local references and projects object fields through schema composition. The
original schema validates complete arguments, including cross-field conditions,
dependencies and property-count rules that framing alone cannot enforce. Extra
properties use JSON-encoded values; statically typed strings retain raw text.
Remote schema references and parameter names containing XML delimiters are
unsupported. Hosted search is unsupported; configure client-owned tools such as
MCP. Omitted effort uses the model default.
Hidden thinking signatures use a persistent user key; imported encrypted thinking
preserves visible history without recovering the private reasoning.

`/status.admission` distinguishes memory and concurrency waits, reports suspended
requests, recovery draining and the oldest current wait age. Memory transitions
also appear in the console. Warning pressure can pause growth while `/ready`
remains healthy for work that fits existing allocations.

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

Chat completions also include a llama-server-style `timings` object, both in
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
TTFT starts before upload and ends at the first native token
event. Output intervals are between native token events, which can contain
multiple speculative tokens; they are not per-token latency. Native queue timing
is recorded from successful completions. These histograms live with the HTTP
process and survive a native engine restart.

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

Greedy and sampled requests can share an unconstrained decode batch; each lane
keeps its own sampling policy and RNG. Pure greedy batches retain their argmax
path. Constrained requests use a separate batch for the host mask exchange.

Long prefill uses disposable rolling checkpoints every 4096 tokens. Contended
prefill adapts toward a 500 ms slice, keeping 2048-token chunks for long unopposed
work. These policies do not extend client deadlines. Memory recovery waits are
bounded, but readiness does not guarantee that a request-sized allocation fits.

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
discovery reports an empty `release_date` because packages do not record one.
The official SDK is a client only, not a server dependency. API compatibility does
not imply Jev weights, accuracy, proprietary confidence semantics or calibration.

These are local model scores, not calibrated confidence. Probabilities are a
softmax over the declared answer slots. Choice/score `confidence` is normalized
entropy concentration, `1 - H(p) / log(K)`, not an estimate of correctness.
Score answers are probability-weighted level indices. Measure accuracy and
calibrate on representative held-out data before using decision thresholds.

Native wire version 6 appends score-token IDs to requests and selected f32 logits
to Done events; a version mismatch is fatal. Scoring requires 2–255 distinct,
in-vocabulary tokens, no images or generation constraints, and a zero output budget.
It may use the full context window because no generated token needs a reserved
position. The final prefill chunk runs the target head but no sampling policy or
DFlash decode. Successful scoring emits no Tokens event, finishes with Stop, and
reports zero decode time. Cancelled requests carry no logits.

A non-finite score logit is a per-request failure, not an engine fault: the
engine reports `model_result_invalid` for that request alone, before it
publishes the failing step's cache state or any output, and the rest of the
batch finishes normally. Prompt chunks that already succeeded keep the blocks
they committed, exactly as they do for a cancelled request. GPU faults and
broken engine invariants stay fatal and still mark the runtime unhealthy.

## Validate

```sh
make check
make install test-real test-http-real MODEL=incoai/Qwen3.8-27B-Splash
```

`make check` needs no model weights. `make check-native-cpu` builds production
and runs native CPU tests without a GPU; `make check-native-metal` requires a
supported Metal device and runs the kernel tests under shader validation. They
include the preparation of small synthetic MLX, GGUF and vision sources and the
GGUF kernels on synthetic tensors. Hosted CI runs CPU checks and sanitizers;
the hardware release gate runs the full suite.

Repeat model tests with `MODEL=incoai/Qwen3.6-35B-A3B-Splash`.
Before release, install all four agents and run `make release-check MODEL=...`
for both models from a clean checkout. It includes correctness, sanitizers,
real HTTP/client behavior and performance checks.

`make test-engine-cpu` builds the affine source oracle so it cannot break
unnoticed, but no target runs it because it needs real models: after
`make all build/engine-tests/affine-source-oracle`, pass it
`build/splash.metallib`, an MLX model's Hub snapshot directory and the
matching installed package to compare every prepared byte.

Compare performance on the same idle Mac with the same model and workload.
`make tune-kernels MODEL=...` measures the precompiled kernel candidates for the
installed model on this Mac against the policy defaults in `runtime/ops` and
prints, per key, the winner with its paired GPU and wall-time gain, spelled as
the enumerators it would install, or that the default is kept; it changes no
default and saves no profile. For a GGUF model
it measures only the attention kernels and the draft, and says so in its
header, since GGUF projection and MoE plans read no tuned choice
([GGUF targets](#gguf-targets)). Keep generated reports, profiles, local paths
and experiment notes out of the source tree and commits.

### Local benchmarks

From a source checkout with the model installed, use the existing native
benchmark for prefill, decode and batch measurements:

```sh
make test-performance-real MODEL=incoai/Qwen3.8-27B-Splash
```

It writes `build/release/backend-benchmark.json`. Repeat with the 35B package
for that model. This characterizes one build; it is not a comparison with
another engine or a test of agent task quality.

For a same-machine HTTP regression check, retain the previous `splash` binary
**and its adjacent `splash.metallib`**, then run from the candidate checkout:

```sh
.venv/bin/python -m dev.benchmarks.http_regression \
  --model incoai/Qwen3.8-27B-Splash \
  --baseline-binary /path/to/baseline/build/splash \
  --contexts 2048,10000 --samples 5
```

This starts isolated servers in alternating order, compares matched cold,
exact-prefix and decode requests, and saves `build/release/http-regression.json`.
It does not contact your running server. Use the same power mode and charger,
stop other GPU workloads, and report chip/GPU cores, memory, Splash version,
model revision, actual input/output token counts, and cache hits with results.
Keep cold prefill, cached TTFT and sustained decode separate; a UI token rate
alone does not measure end-to-end agent performance.

For slow tool-bearing requests, the `latency` section of `/status` separates
preparation, tokenization, grammar preparation, native queueing and TTFT. Grammar preparation
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
Server restarts require recomputation. The separate
[SSD cache proposal](https://github.com/incoai/splash/pull/3) preserves evicted
model state during a server session; its temporary files do not survive shutdown.

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
