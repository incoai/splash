# Internals

## Code and API boundaries

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
responses alike, in the family's chat format (`chat_format.py`), and
`constraints.py` compiles token constraints. `workers.py` runs the decoders of
untrusted input, PDFium for PDF pages (`document_worker.py`) and the engine's
`video-frames` command for video frames (`videos.py`), each in a disposable
process within the time, memory and output budgets it supervises. A decode
whose hardware decoder stalls exits with 75, and the server decodes again
with `--software`: VideoToolbox's software decoder on the compressed samples,
each range read from the keyframe before it to the keyframe after it, the
ranges on all cores. A codec with no software decoder exits with 69 from
`--software`, and only hardware is left to try. Messages and
Responses build streamed and complete responses from the same block sequence;
the one difference is a call cut by the token limit or a stop sequence, which a
complete Messages response leaves out. `make architecture-check` prevents lower
layers from importing the HTTP entry module, and keeps one import style in
`server/` and `install/`: a module imports its package's modules relatively.
The server runs as `python -m server.server`; `install/launcher.py` and
`install/models.py`, which run as scripts, import their siblings through a
PEP 366 header. `serve_options.py` defines the options
`splash serve` shares with the server once, each with its check, default,
help and help group, and how the launcher passes it on; it imports only the
standard library, since the launcher parses them before `.venv` exists.

Flash-Next's two hyper-connection norm kernels, which split a mix's rows
between them, must write the same bits, which only an uninstrumented run of
the production metallib on the release GPU and toolchain shows
(`build/engine-tests/qwen4-kernels build/splash.metallib bitwise`, which
`make test-engine-metal` runs). Shader validation is a separate bounds and
numerical-closeness gate; it does not establish bitwise equality of
production code. Before release,
compare full-model greedy outputs at the same prompts, cached prefixes and
batch schedule against the base. Repeat seeded sampled requests within each
build, and explicitly check base-versus-candidate seeded output when preserving
a drafting policy. Run a MoE target with `--verify-rows all` for these
comparisons: the verify budget its cycles otherwise follow makes greedy and
seeded outputs vary from run to run ([Drafts](models.md#drafts)). These are measured
gates, not a guarantee across arbitrary compiler versions or GPUs.

The values the native runtime holds constant (the Metal command timeout, the
KV tier's transfers, the input queue's bound, the latency window, the
write-behind budget, the prefill checkpoint interval and the resource wait),
its clocks and its live host-memory estimate are parameters of the component
that holds them: production leaves them at their defaults and tests set them.
`make architecture-check` keeps production from measuring durations on the
standard library's steady clock or its timed waits, which count sleep: the
runtime measures them on `AwakeClock` (`runtime/AwakeClock.hpp`), and
wall-clock instants on `system_clock`.

## Installation

`install/upstream.py` installs a model from its upstream repository: it
inspects the target, pairs the draft trained for it and decides when to follow
the Hub. The result is an assembly, a local directory of links to the sources'
Hub snapshots, published atomically. The other installer modules each own one
part: `hub.py` the sources, the Hub cache and its pins; `assembly.py` the
assembly layout, its build, verification and garbage collection, and the
metadata derived from a GGUF; `families.py` each family's draft repository;
`local.py` the assembly of local files ([Local model files](models.md#local-model-files)),
which inspects them and publishes them as `upstream.py` does a repository's;
and `models.py` model IDs, selections, the
installation lock, the command line (`install/models.py --model ID
prepare|verify|link`, where `link` prints the selection link) and running the
engine's checks. Since the installer checks a model with the engine, a source
checkout builds the engine before it installs, as `make install` and the first
`./splash serve` do, and a packaged install ships it. The assembly's
`model.json` records the resolved sources and selected formats. The native
loader reads it, and the server holds it while it runs, as `test-http-real`
and the HTTP regression benchmark do across the servers they start, so a
concurrent installation cannot collect the assembly they serve. It is local
installation metadata, not a file model publishers supply. It records each
file's path, size, times and digest: a Hub cache file's is the blob's name,
any other file's its SHA-256, which an assembly that replaces another keeps
for each file it still links unchanged in path, size and times, so only a
new or changed file is read for it, before the installation lock is taken.

Installation resolves each source's revision to a commit once, downloads by
that commit and records it in `model.json`, so a repository update cannot mix
files from different revisions. It pins those snapshots in the Hub cache
(`refs/splash/<installation>/<commit>`), so pruning the cache cannot remove files
an installed model links. Publishing a new assembly retires the installation's
other pins, in the repositories it links and in those its predecessor linked,
so pruning can free what no installation links any more.

`hub.Repository.download` first removes the partial files
(`<blob>.<random>.incomplete`) killed downloads left in the repository's blobs
folder, of any blob, each under huggingface_hub's lock on its blob. Front ends
such as the desktop app run `install/models.py --model ID prepare
--json-progress`, with which each download also reports its progress as JSON
lines on stdout, `{"event": "download", "repo": REPO, "done": BYTES, "total":
BYTES}`: done 0 first, at most four a second as the bytes received grow (what
the network has brought or what the Hub cache holds, whichever is more), and
done equal to total once it succeeds. It is an interface between Splash's front
ends and its installer, not a public one.

huggingface_hub downloads on threads it joins, which finish their files
before an interrupt reaches the program, so the download runs on a thread of
its own (`hub._interruptible`) and Ctrl+C stops the installer within a
fraction of a second: it unwinds its temporary files and ends without waiting
for those threads, and the repository's next download removes the partial
files they leave. A launcher run without writing bytecode (`-B` or
`PYTHONDONTWRITEBYTECODE`) sets `PYTHONDONTWRITEBYTECODE=1` in its environment
before `serve` starts anything, so the installer, the server and the server's
PDF workers write none either, and a packaged runtime inside a signed app
never gains `__pycache__`.

`hub.Repository.resolve` alone decides whether a start asks the Hub
([Revisions](server.md#revisions)), each request within `hub.HUB_TIMEOUT`:

- The installed commits: the assembly's links, sizes and times are checked and
  it starts. It is re-assembled first when the draft's repository moved or
  this release changed the GGUF metadata adapter; if the new draft cannot be
  fetched or is not the family's ([Drafts](models.md#drafts)), the installed one is
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
records for the branch, never of another revision. Splash downloads by
commit, which records no branch, so when the Hub names the commit of a
default branch, Splash records it as `refs/main`, as a download of the branch
does, once the cache holds that commit, when the Hub names it or a download of
it ends: a model another Splash, such as a source checkout, installed from the
default branch installs here too. Another installation's pins never stand
in, as a pin does not say which revision it was installed for. Only files
downloaded before are available, which is enough to rebuild a damaged or
deleted assembly. `install/cached.py` installs this way every model the Hub
cache holds whole that Splash serves, leaving out what the installer refuses,
for the desktop app to offer: a GGUF by the ID `install/suggested-models.tsv`
gives it, or else by its split parts' folder or by its name past the model's,
`MODEL-VARIANT.gguf` as `OWNER/MODEL-GGUF:VARIANT`, an ID that selects it from
the Hub's listing too. The installer never rewrites upstream files.

## Weight loading

Every start writes a model's target, draft and vision tensors into weight
images in memory, in the layouts the kernels read: any vision tower in the
BF16 layout of the vision operator, and the target, safetensors or GGUF, and
the DFlash2 draft in the `MDGG0001` layout of the GGUF kernels ([MLX
targets](models.md#mlx-targets)). Each source adapter is a loader, which validates the
source's metadata and plans its images, and a writer:
`SafetensorsTargetLoader` (`SafetensorsTarget.cpp`, planned by
`SafetensorsImage.cpp`) and `GgufPreparation` for a safetensors target,
`DraftCheckpointLoader` (`DraftCheckpoint.cpp`) and `GgufPreparation`
for the draft, `GgufTargetLoader` (`GgufTarget.cpp`, planned by
`GgufImage.cpp`) and `GgufPreparation` for a GGUF target, `VisionLoader` and
`VisionPreparation` for a safetensors or GGUF vision tower. `GgufPreparation`
repacks GGUF blocks ([GGUF targets](models.md#gguf-targets)) and safetensors tensors
without requantization, but for the Q8_0 of Flash-Next's BF16 or F16
hyper-connection projections, quantizes the draft's BF16 projections
([Drafts](models.md#drafts)), and computes a safetensors target's GDN decay as
`float(-exp(double(A_log)))`, which may differ by one float ULP from MLX's float
exponential.

Loading never rounds a target or vision weight but in the F32 values of
quantized tensors the kernels read unquantized, in the Q8_0 of Flash-Next's
BF16 or F16 hyper-connection projections ([GGUF targets](models.md#gguf-targets)), and
in two conversions: the GDN decay above and the RMSNorm weights below. The first are a safetensors
target's quantized shared-expert gate, and its quantized GDN alpha
or beta beside one of another format or an unquantized one, an MLX target's
([MLX targets](models.md#mlx-targets)) or a Model Optimizer or compressed-tensors
target's in NVFP4 or FP8, and a GGUF's quantized alpha or beta beside one of
another type, which the GPU decodes as GGML does. In the second conversion, a
Model Optimizer or compressed-tensors target's RMSNorm weight `w`, stored 1
below the weight the norm multiplies by, loads as
the F32 `1 + w` ([Model Optimizer targets](models.md#model-optimizer-targets)), which
can round where `|w| < 2^-16` or `|w| >= 2^24`. Loading rounds the draft's
projections only as MLX's affine quantization does ([Drafts](models.md#drafts)). A
tensor it converts to BF16 (vision tensors stored as F32 or F16, a GGUF's
convolution taps and time-step bias) must be exactly representable in BF16;
otherwise loading fails,
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
([upstream loading](../dev/benchmarks/upstream-loading.md#loading-time)).

At load time the engine validates the GGUF metadata, including the rotary
embedding and norm epsilon the kernels assume (`rope.freq_base`,
`rope.dimension_count`, which a Flint GGUF, whose RoPE rotates whole heads,
need not state, `attention.layer_norm_rms_epsilon`, and no
`rope.scaling.type` but `none`), and plans the `MDGG0001` layout. A Flint
GGUF's family keys hold its window, leading dense layers, shared expert's
slices, sigmoid router and normalized, unscaled routing weights
(`expert_weights_scale` 1) and its two layer patterns to the family's, and
a router bias (`ffn_exp_probs_b`), which its blocks would not add, is
refused.
`GgufPreparation` stages rows in image order on the CPU within the
staging bound, splitting rows wider than it into column chunks, and runs the
`gguf_repack` kernel, which writes their planes into the image. Embeddings and
F32 sections are copied in bounded steps; the F32 values of a quantized tensor
the kernels read unquantized are written the same way by the format's fp32
gather (`gguf_embed_f32_<format>`), the decode its token gather rounds to bf16.

Each image's record (`WeightFileRecord`) names what it was written from: the
SHA-256 of the record of every source file's digest, the assembly's
`model.json`, which the installer verifies at every start
(`ModelDescriptor::sourceIdentity`). The weight manifest fingerprints `/status`
reports (`loaded_model_layout_sha256`, `target_model_sha256`) cover it, so
models of one layout from different sources never share a fingerprint.

Runtime admission counts the images exactly once (`streamableWeights`, which
the runtime oracle and `weight-digests` use too). Before loading, startup
refuses a model whose images, with the pipeline and runtime reserves, one
lane's state, the KV runway and any disk tier state staging, exceed the hard
budget, so a model that can never fit is not loaded.

When the idle release passes without a request, the memory control between
commands releases the images' memory (`WeightImages::release`): their buffers
are freed, the weights' views stay the same handles, and a command that binds
released memory fails (`MetalBackend::releaseMemory`). The next request waits
while the same writers write the images again (`WeightImages::restore`), an
image per tick so that the loop keeps answering status and cancellations; this
takes about as long as the load at startup and logs `Weights restored in N s`.
The governor admits each image's memory (`MemoryGovernor::allocationAdmission`)
as memory the waiting requests need: neither the host's margins nor macOS's
warning refuse it, as for a request in service, and it fits the
engine's limit, since the cache does not grow while the engine holds no
request; critical pressure refuses it. The [expert
store](server.md#streamed-experts) is the exception to that fit: a chunk it gave
back while memory was short left room the cache may have taken since. So it
comes back in parts: its floor chunks and staging in one, which fits as the
images do, since they were live when it was released, then each elastic chunk in
a part of its own, as growth the store can do without
(`MemoryGovernor::growthAdmission`): the serving mark does not lift its host
checks, and it comes back only while macOS reports no memory pressure and the
host has the chunk and the 2 GiB recovery margin to spare. The first elastic
chunk refused stays released with those after it, which the control pass takes
back later as it does any chunk given back, and the restore goes on as if it had
come back: only the floor holds the restore up. A refused image, or one the
driver or its writer cannot allocate, stays released, and the images before it
stay. The
requests wait for memory, retrying every 0.1 s as a refused request does and
counted in `admission.waiting_memory`, and the log says `Weights wait for
memory to be restored`. Once the 30 s resource wait has passed since the first
refusal after the last image that came back, the restore gives up: the images it
wrote back are released, the waiting requests fail with `resource_timeout`, and
`weights.restore_failures` counts the give-up. When no request waits any more,
it gives up at once and counts nothing. Either way the next request starts over
from the first image. Any other failure (a read error, a source written in
place) stops the engine, which the server starts again. With the [Neural Engine
split](#neural-engine-prefill), the release unloads its program after the
images, and the restore loads it again in one tick more, after the last image
(`ReleasableMemory`).

### Explicit weights release

`POST /v1/memory/release` asks the engine for the same release without waiting
for the timer: the server sends a `ReleaseRequest` frame, and the engine,
between commands, releases the weights when it holds no request and no restore
is under way, then answers with its status document
(`NativeRuntime::handleRelease`). Nothing else changes: the next request
restores the weights as after the idle release (`Weights restored in N s`), the
prefix cache and the other caches stay as they are, and `--idle-release off`
does not stop it, so a host that shares the Mac with other memory-heavy work can
take the weights' memory back on demand instead of on a timer. The server
answers `200` when the document says the weights are released and not being
restored (`weights.released` and not `weights.restoring`, as a restore leaves
them released until its last image is back), whether this request released
them or they already were. An engine serving a request or restoring its weights
keeps them, and the server answers `409 engine_busy`. An engine that does not
answer within 5 s, as while it runs a long command, gets `503 engine_busy`, and
while the engine restarts or the server stops the release gets
`503 engine_recovering` or `server_shutdown`, as other requests do. The request
takes no body:

```sh
curl -X POST http://127.0.0.1:8000/v1/memory/release
```

`loadQwenTarget` (`QwenTargetLoader.hpp`) reads a target's images
(`QwenTargetFiles`: the images `SafetensorsTargetLoader` or `GgufTargetLoader` plans)
through `BlockTargetFormat`, which reads each tensor as one block-quantized
`QuantizedSegment` (a fused projection's tensors in output column order), the
norms as F32 (bf16 for an MLX checkpoint), and keeps the GDN output projection's
input in the source's value-head order, llama.cpp's tiled or MLX's grouped.
Every family's layout derives `QwenHybridLayout`, whose validator
(`requireQwenLayout`) it shares.

A family's target is a `QwenTarget` (`QwenTarget.hpp`) composed of the blocks
every Qwen target shares: a step's checks and plans, the GDN and attention
mixers, the FFNs of a residual of one row (a dense SwiGLU and a MoE block's
routing and experts), the draft's captures and the normed head. The Qwen3.5
architecture's target (`Qwen3_5Target`, `QwenTarget.cpp`), Flash-Next's
(`Qwen4ExpTarget`) and Flint's (`FlintTarget`) each compose them into their own
layer loop. What a family changes in a block is a value of its layout
(`QwenTargetDimensions`), read by the shared block rather than branched on: an
attention layer's variant (`QwenTargetGeometry::attentionVariant`: rotated by
the target's RoPE unless `unrotatedLayers` names it, over a sliding window of
`slidingWindow` keys where `slidingLayers` does), for which a step plans the
windowed layers apart; the learned sinks, which a layer's
`QwenAttentionWeights` carry where its geometry adds them; the MoE router and
the shared expert's slices and gate (`QwenTargetDimensions::moeBlockShape`);
and the leading layers of a MoE target whose FFN is dense
(`leadingDenseLayers`), whose blocks, and streamed experts, are the MoE layers'
alone. The output gate (`attentionGate`) and the RoPE pairs (`rotaryPairs`) are
values the attention kernels compile with their geometry (`kCompiledAttention`),
to which a target holds its layout's (`PagedAttention::compiles`). Flint
(`Flint.hpp`) has no GDN layers (`fullAttentionPeriod` 1): its 49 layers
attend, 35 over a window of the 512 keys before a row's with RoPE over whole
heads and 14 over every key without RoPE, each with its sinks; its first two
layers' FFN is dense and the others' a MoE of a sigmoid router and an ungated
shared expert of two slices; its norms are plain, as its GGUFs store them. Its
layer images (`GgufImage.cpp`, `FlintGguf`) hold its input norm, attention,
sinks, the norm before its FFN and its FFN, which `loadFlintWeights` reads with
the Qwen readers.

Every source loads into block planes: `Projection`, `BlockMoeWeights` and
`EmbeddingWeights` (`runtime/ops/Weights.hpp`, `MoE.hpp`) hold them whatever
the source container, and represent different operator contracts. Arena
sizing collects each projection's shape and reserves the vocabulary head only
for decode.

## Kernels

Every tensor keeps its stored format: the F32 norm multipliers, GDN decay,
shared-expert scalar gate, and GDN alpha/beta when a file stores them as F32
stay F32 and run in fp32, as llama.cpp keeps them. The MoE router runs in fp32
in the format its checkpoint holds it: a GGUF's F32 or BF16 values, or the
native rows of a quantized router (an MLX checkpoint's 8-bit affine one), each
weight the fp32 value the token gather decodes, so its scores are those of the
F32 tensor of its values bit for bit (Apple10 prefill chunks multiply the
router and alpha/beta on the neural accelerator as three bf16 parts per weight
that sum to it exactly, so only fp32 accumulation rounds). Its scores pick the
experts, so it stays off the block tiles, whose staged and prefill tiles round
the weights to half; in decode they are slower on its narrow grid as well: on a
32-core M5 Max the 35B's router as 8-bit planes takes 6.3, 8.7, 11.1 and 15.6
us at one to four lanes on the register tile, its native rows 5.2, 6.3, 7.4 and
8.5 us on the float projection (prefill chunks: 58-74 us on the prefill tile at
64 to 2048 rows, 10-184 us on the float tiles, faster up to about 550 rows;
weights DRAM-cold). The GDN convolution and time-step bias become bf16 under
the exact rule.

Decode runs one of two kernel families, chosen by GPU family in `runtime/ops/LinearGguf.cpp`. On
Apple9 (M3, M4) the register tile (`LinearTile::GgufRegister`) runs the kernels of
`runtime/metal/kernels/decode/linear_gguf_sgmatrix.metal`, which feed the codes themselves to
bf16 matrix operations, each chain of 32 inputs (16 for 16-input scales) closed by its fp32 scale,
so every output is the bf16 rounding of its fp32-accumulated sum. The coefficient
group is the inputs one scale covers (a format's `Group`: 16, 32, or the groups of a meta unit that
share one, as MLX affine's groups of 64 and 128, PQ2_0, Q2_0, FP8 and BF16 do): a simdgroup decodes
a column's coefficients into threadgroup memory once per group, and a format with a min adds its
term, the min times the sum of those inputs, once per group but at least once per 64-input span of
the table, from that span's sums (a group of 128 gathering four sums across two spans took 4.5-8.8%
more time at four lanes than a term per span). The MMA shares Apple9's FP32 pipe with that epilogue;
on a 10-core M4, against a decode and min term per 32 inputs, MLX 4-bit g64 at the 19 projection
shapes of the 27B and 35B and their drafts takes up to 2.7% less time per shape at one lane (none on
the 35B's GDN in-projection, most on the LM heads and the 35B draft's down projection), 1.7-3.7% at
two and three and 3.3-4.6% at four, which over a B1 DFlash2 cycle's calls of those shapes is 16.67
instead of 16.89 ms on the 35B and 165.80 instead of 168.06 ms on the 27B; MLX g128 takes 1.7-1.9%
less in total at one to three lanes and 3.6% at four, though not 4-bit 2048 x 4096 at one to three.
A chain over a whole 64-input group instead leaves a lane two dependent chains per simdgroup where
it has four, and took 1-6% more time at one to three lanes. A zero point is such a term too,
entered where it costs no epilogue: codes with one enter as 160 + code - zero, and the term's last
chain starts from -160 times the sum of its inputs, from the table's sums of 32 once per group and
span where a scale covers 32 inputs or more (Q4_0, Q5_0, IQ1_S, PQ2_0, Q2_0), and from the table's
seed of 16 inputs per chain where scales cover 16 (Q6_K, Q3_K, IQ1_M). On the 10-core M4, against a
decode and a seeded chain per 16 inputs, those five formats at projection shapes of 2048 to 17408
columns take 1.4-7.1% less time at one lane and 3.5-8.0% at two to four. The zero point as an
epilogue term instead, b * sum with b = -(128 + zero) s, took up to 3.3 points more (IQ1_S at one
lane, 2.0-2.7 in Q4_0 and Q5_0 at two to four), and so did chains of 32 seeded with the sum of two
16-input seeds; code - zero as the operand, its chains starting from zero, took 2.0-5.9% less time
than seeded chains at three and four lanes in Q6_K, Q3_K and IQ1_M but up to 3.5% more at one and
two (Q3_K at one lane), and on the 32-core M5 Max up to 3.9% more at one lane (IQ1_M at 512 x
2048). They read their activations as the
Table16 table (`kernels/common/gguf_sgmatrix.h`) that the input's producer writes, or
`decode_linear_gguf_prepare` when none did. The formats whose operands those kernels build from
grid lookups, IQ3_XXS, the IQ2 formats and IQ1 (`apple9StagesFormat`), and Q2_K from two lanes
decode faster on the staged tile there, which a projection all of whose segments are in them
takes wherever the tile holds its lanes' rows unpadded. F16, whose 11-bit significands bf16 does
not hold, has no register kernels (`quant_register_format`), and they read a tensor in rows of
whole coefficient units (`quant_register_tensor`: 256 inputs for a format of several groups per
meta unit, else 64), each K partition whole units (`quant_register_unit`): a projection with a
segment they do not take, or a gate/up pair with one, stages at every lane count, and its decode
storage holds the staged tile's rows. On Apple10
(M5) the staged tile (`LinearTile::GgufStaged`) runs the kernels of
`runtime/metal/kernels/shared/gguf_linear.metal`, which dequantize each weight once to half in
threadgroup memory (`kernels/common/gguf_staged.h`) for MPP `matmul2d`, the neural accelerator's
path, on bf16 activations. A decode step whose grid is narrow there, fewer than one and a half
threadgroups per core (half that in formats of fewer than four bits per weight), runs on the
register tile instead, whose 128 threads per column tile keep more of its cores' loads in flight
than the staged tile's 64 (`apple10Registers`). Its coefficient groups apply there too: on the
32-core M5 Max the 35B's MLX 4-bit g64 projections of 2048 columns, which take the register tile
at one lane, take 1.6-2.2% less time at their splits (GDN out, the draft's down and its context
commit's fc), and the 320 x 10240 hyper-connection projections, which take it at every lane,
0.4-1.8% less in MLX 4-bit and 8-bit g64 at one to three lanes and 4.4-5.2% at four, BF16 the
same; with a zero point's term per group, against a decode and a seeded chain per 16 inputs, Q4_0,
Q5_0, Q2_0, PQ2_0 and IQ1_S at 320 to 2048 columns, at the lanes that take the tile, take 1.0-4.2%
less time at one lane and 2.1-5.9% at two to four. Partitions of under 512 inputs at sixteen splits
(one or two coefficient units) take 2-10%
more time at one to three lanes, which the tiers never pick, as sixteen keep 512 inputs per
partition; at eight, partitions of one unit took 4.1% less to 0.8% more. A step's weights, and
its meta unit when it enters a new one, load while the previous step's matmul runs
(`gguf_staged_steps`): their DRAM round trips overlap a matmul. On Apple10 from 16 cores each
column tile of a decode projection starts its walk over its K partition at its own step
(`staged_first_step`, `LinearConfig::spread`), so the tiles do not all wait on the same freshly
written slice of the input at once; the MoE experts, prefill, Apple9 and fewer cores walk in
lockstep. There a dense gate/up pair whose tensors share their format may also decode in one
dispatch (`LinearConfig::oneGateUpPass`), each simdgroup staging both tensors' columns, where
the staged split law below takes it. Its output is bitwise the two passes' at the same K split;
where its split differs from the two passes', its sums are reassociated. A step of three request
lanes runs the 32-row tile over four lanes of storage. On Apple10 one lane runs the 16-row tile
over two lanes of storage (`LinearConfig::sixteenRows`), its padding lane computed and
discarded, where its staged plan is unsplit and the projection's quantized segments store fewer
than four and a half bits per weight together (`ProjectionShape::underFourAndHalfBits()`), an
unsplit gate/up pair where both its tensors' own plans do: the lane's rows are bitwise the 8-row
tile's, in less time (the 27B UD-IQ3_XXS decodes one lane 3.3-3.6% faster on a 12-core M6).
Split, the padding lane's partials double the split's traffic: on the narrow grids where the law
splits K most (the 35B's, the 122B's and Flash-Next's output projections, 1 to 1.5 threadgroups
per core unsplit, in four or eight partitions) the 16-row tile takes 4-36% more time than the
8-row tile at the same split on a 32-core M5 Max (IQ3_XXS, IQ2_XS and Q2_K, and IQ4_XS where it
stages), against 2-8% less at 4 to 20 threadgroups per core on the M6 and 7% less to 2% more on
the 27B's 5120-column projections in four on the M5 Max. Q4_K's 4.5 bits take more time on it on
the M6. Prefill runs the staged kernels on both families: the 128-row prefill tile
(`LinearTile::GgufPrefill`), and the staged tile for chunks of up to 32 rows. On Apple10 chunks of
33 to 64 rows run the staged tile's 64-row height too, which reads the weights once where the prefill
tile costs about as much for 33 rows as for 64; a fused projection's chunk runs its segments there in
one dispatch, as decode does.

Apple10's staged tile splits K across threadgroups, and runs a gate/up pair in one pass or two,
by one wave law (`apple10Staged` in `runtime/ops/LinearGguf.cpp`), for decode steps and short
prefill chunks alike: it takes the split (of at most eight partitions keeping 512 inputs each)
and passes of least predicted time, ties to fewer partitions, then one pass. Each tile kernel
has an occupancy (`StagedOccupancy`): a core runs `resident` of its threadgroups at once, so a
grid runs in waves of that many per core (10 for the two-simdgroup tile, 16 for the
one-simdgroup pair, both measured); a grid of one wave takes at least `saturating` threadgroups'
time per step (with fewer a core waits on its chains' latency), and a last wave after others at
least `following`'s, since it starts as the one before drains. Each threadgroup streams 64
columns of a K step of the projection's bits per weight (`ProjectionShape::weightBits16`, the
most a format stores where the formats are unknown, which bounds the scratch), and a split
grid's threadgroups each also store their fp32 partials of the tile's rows, which the last
partition reads back. The constants come from sweeps of 2 to 40 threadgroups per core on the
32-core M5 Max, the 20-core M5 Pro and the 12-core M6 (`gguf-projection-benchmark`, which times
a gate/up pair in both pass modes; MLX 4-bit and six GGUF formats at 2048 and 5120 inputs,
gate/up pairs in MLX 4-bit): time per byte steps up past 10 and 16 threadgroups per core at one
and two lanes (past 8 or 9 of the two-simdgroup tile on the M5 Pro), and the 32-row tile of
three and four lanes steps past 9 on the M5 Max, 8 on the M5 Pro and 11 on the M6, which the law
leaves out. The floors are fitted, not measured: a grid search over every measured case's
fastest split gave 6 and 10 for `saturating` (a lone chain measures 5.4 and 9 threadgroups'
bytes) and 2 and 3 for `following` (one threadgroup per core past a full wave measures 2.7-3.5
of the tile's and 6-8.5 of the pair's); they stand in for what the law leaves out. Floors at the
measured trailing waves take as much time in total on the sweeps but leave more cases slower.
The law's totals on the sweeps (1.1%, 1.6% and 0.05% less time than the earlier rules on the M5
Max, the M5 Pro and the M6) are in-sample, the fit's own records; the ABBA decode runs and a
16-core emulation are the only out-of-sample checks, and the down projections' 17408 inputs,
pairs in other formats and low-bit pairs take the law unmeasured. On the 12-core M6 the 35B draft's
fc projection (2048 columns over 16384 inputs) runs two partitions instead of four at three and
four lanes, 1.0-1.2% (2 us a cycle) more, within the 1.2-1.4% the benchmark varies between
repeats; the law keeps it. The benchmark times the 27B's IQ4_XS gate/up pair 6.5% slower in two
passes than in one at three and four lanes on the M5 Max (318.0 against 298.5 us), yet in the 27B
UD-Q4_K_M's decode cycles at a 1024-token prompt the law's two passes took 0.8-1.2% less time there
than a build running IQ4_XS pairs in one pass (1.2-1.8% at one and two lanes, six runs each), so the
law keeps them.

The law tells a gate/up pair's pass modes apart only where a last wave falls below its floor. On
the M5 Max the 27B gate/up pair runs two passes (one pass would put 17 threadgroups on each
core, the 17th a second wave alone: 333 us against 245 at one lane). Where both modes' last
waves clear their floors the law predicts the same time for both, and the tie takes one pass,
which the sweeps measure faster in 133 of 152 tied cases (by up to 22%) and slower at 20 and 36
one-pass threadgroups per core (by up to 20%, most at one lane): a 20-core M5 Pro (27.2 per
core) and a 40-core M5 Max (13.6) run one. A 16-core M5 Pro puts 34 one-pass threadgroups on
each core (16, 16 and 2), where the law takes two passes at every lane count; emulated by width
on the M5 Max, two passes take 9-12% less time than one unsplit pass, the earlier rule's plan,
at one and two lanes but 7% more at three and four (the 32-row tile's lower residency), so there
the 27B gate/up pair's three- and four-lane steps lose about 7%. The law keeps that loss: a 32-row
residency of 8, which gives those steps back, sends the 32-core M5 Max's 27B pair to one pass at
three and four lanes, 19-26% slower than its two passes, and a pair efficiency factor (0.92-0.97)
fits the sweeps worse; no 16-core Mac has measured the case. Fewer than 16 cores run two
passes whatever the law (`spreadsStagedDecode`): a 12-core M6 took 2-11% more in one. The law's
plans depend on the batch width, since partials grow with the rows, so a request's greedy output
may change with the requests batched with it (the 27B MLX's decode cycles of three and four
lanes and the 35B MLX's of two to four, and of one at other prompt lengths, yield other tokens per
cycle than before), and with it a step's time: on the 32-core M5 Max's 35B MLX at two lanes, where
only the draft's gate/up pair changes (one partition to two), cycles took 0.5-1.1% more at a
1024-token prompt but 0.9% less to 0.7% more at 640, 896, 1280 and 1536 tokens, tokens per cycle
moving both ways, so that step follows the text rather than the kernel. The
register tile and Apple9's staged tile, whose matrix operations share the FP32 pipe or run exact in
registers without a residency step in their time, split by tiers (`decodeSplits`: threadgroups per
core, inputs per partition and partitions, measured on a 40-core M3 Max, Apple9's staged tile taking
the register tile's of eight partitions) that do not depend on the batch width.
Partitions stop at eight, but the register tile splits a grid that eight leave below three
threadgroups per core into sixteen while partitions keep 512 inputs (Qwen3.8-Flash-Next's
hyper-connection down projection, 320 columns over 10240 inputs; measured on Apple10). The MoE
experts
(`runtime/ops/MoE.cpp`) run the same numerics per family over the grouped rows: the register
form in `linear_gguf_sgmatrix.metal`, the staged one in `kernels/shared/moe_gguf.metal`, which
Apple9 takes on GPUs of 10 cores or fewer (the base M3 and M4), and on larger ones for experts
mostly in the formats it stages (`MoeShape::expertFormat`) and for experts with an F16 weight
(`MoeShape::registerFormats`). One `MoeShape` serves all of a model's
MoE layers, so one F16 routed or shared expert tensor in any layer stages every layer's experts
there. A block routes as one of the routings the kernels compile (`ops::kCompiledMoeRoutings`):
its router weights a row's top k experts by the softmax of their scores (`MoeRouter::Softmax`,
the Qwen families) or by each one's sigmoid over the sum of theirs (`SigmoidNormalized`, Flint),
and every row also visits the `MoeShape::sharedSlices` slices of the shared expert, ids `experts`
on, of the routed experts' width, each weighted by the sigmoid of the block's scalar gate
(`sharedGate`) or by 1. A shared expert wider than the routed ones runs as such slices in the
grouped passes: the planner keeps its gate and up rows and lays its down projection out as column
windows (`gguf::TensorRows::windows`), so slice s's rows of each projection follow slice s - 1's
(`moe_gguf_segment`). The select kernel names the routing's select suffix
(`moe_route_select_f32_sigmoid_s2`), the grouping and expert passes its slices suffix
(`moe_group_routes_s2`, `moe_expert_gguf_m8_a_s2`); the softmax router's one gated slice is the
unsuffixed instances. The expert passes launch the live tiles alone: the grouping kernel writes
each pass's grid with the tile count, and the pass reads it as an indirect dispatch
(`ComputeDispatch::indirectThreadgroups`). They find a routed expert through the block's slot
table and the chunks that hold its slots (`ops::ExpertChunks`): slot s below floor_slots is at
place s of the floor chunk, chunk 0, and slot s from floor_slots on at place
e - (c - 1) * chunk_slots of elastic chunk c = 1 + e / chunk_slots, where e = s - floor_slots (a
branch uniform per threadgroup). The passes reach the chunks' regions through a table of
`MOE_EXPERT_CHUNK_ADDRESSES` GPU addresses per chunk (`MoE.h`, `moe_expert_chunk`,
[Residency](#residency-and-kv-extents)); a resident block's experts are one chunk of its routed
planes, floor_slots = chunk_slots = its slots, a streamed block's the expert store's floor and
elastic chunks of its size (`model/ExpertChunkLayout.hpp`,
[Streamed experts](server.md#streamed-experts)). The staged tile runs gate and up in one pass, a gate
and an up simdgroup on the same columns, bitwise the two passes' intermediate; the simdgroups
share one pair table where gate and up share their routed and their shared formats.
An expert's only tile of at most `MOE_GGUF_COLUMN_ROWS` live rows runs on the column tile
(`kernels/common/gguf_column_tile.h`) rather than the staged one, and the two sum a row's
products in different orders: a route's output depends on how many rows the step routes to its
expert, so decode results of GGUF MoE targets on the staged tile, the 35B's included, are not
invariant to the batch (`kernels/common/moe_expert_slab.h`). Under the default `--verify-rows
auto`, which rows a decode step routes also follows the verify budget ([Drafts](models.md#drafts)), which
learns from timing and earlier requests, so a MoE target's decode results can differ between runs
of one request.
The float router, QSA indexer and alpha/beta projections run in `kernels/shared/gguf_float.metal`,
which runs two projections of one input in one dispatch where they take the same tile (the
indexer's query and key, each on the tile of its own width), and the token rows are gathered by
one template in `kernels/shared/embedding.metal`. The float projection reads a quantized format's
rows (`QuantizedSegment::rows`) through the gather's decode
(`kernels/common/gguf_embedding_formats.h`), the MLX affine formats as runs of 8 codes with their
block's scale and bias, read a step ahead and decoded once for the step's products, and writes
their products in fp32, the router's scores. The 35B's 8-bit affine router (256 x 2048, 0.56 MB
a layer against its F32 values' 2.1 MB) trades the bytes it saves for its decode: at one to four
lanes it takes 5.2, 6.3, 7.4 and 8.5 us on a 32-core M5 Max against 6.7, 7.1, 7.5 and 8.2 for its
F32 values (4% slower at four lanes), 6.3, 8.3, 10.4 and 12.6 against 14.7, 16.1, 16.5 and 17.9
on a 12-core M6, and 10.9, 14.7, 17.9 and 21.1 against 22.2, 23.9, 26.2 and 28.0 on a 10-core M4;
the 122B's (256 x 3072) 6.7, 8.4, 10.0 and 11.5 against 9.0, 9.5, 10.0 and 10.9 on the M5 Max
(`gguf-projection-benchmark float-af8g64 256 2048`, weights DRAM-cold). In the 35B's one-lane
decode on the M5 Max its dispatch takes 8.4 us a layer against 9.45 (`decode-profile`, ABBA, 4
runs of 96 cycles a build): 0.04 ms of a 14.7 ms cycle, within the cycle's noise. Prefill chunks
of 1024 to 2048 rows on the M5 Max's accelerator tile, where the decode lies on each step's
critical path, take -10% to +7% of the F32 values' time, and the M4's simdgroup tile -3% to +6%
at 64 to 2048 rows.
A float projection takes the neural accelerator tile on Apple10 from 16 rows and a grid of its
64 x 32 tiles of 1.5 per core for BF16 values, 2 for F32 values and 3 for a quantized format's rows,
by the bf16 parts the tile stages a weight as (one, three, three decoded; `Linear::ggufFloatTile`).
Where the two tiles take the same time on a 32-core M5 Max and a 12-core M6
(`gguf-projection-benchmark float-<format>`, rows in steps of 64): F32 at 2.0-2.67 tiles per core
for 256 columns of 2048 or 3072 inputs, and 1.75-2.0 (M6) or 2.0 and above (M5 Max) for the 64 and
96 of GDN alpha/beta; quantized rows at 2.67-4.0 (8-bit and 4-bit affine, Q8_0, Q4_K). Over those
points the rule's tile takes at most 13% (F32) and 6% (quantized rows) longer than the faster one,
where 1.5 for every weight took up to 42% and 61% longer; on the M5 Max the 35B's native router rows
take 52.6 us at 512 rows on the simdgroup tile against 71.8 on the accelerator's. BF16 crosses at
0.67-1.33, under its rule, which keeps a decode step's 16 to 32 rows of a 512-column router off the
accelerator tile on 12 cores (1.6-1.8x the simdgroup tile's time there). These plans are fixed
rules of GPU family, core count, shape and format.

A rotated projection rotates its input once into `LinearScratch::rotated`
(`gguf_rotate`, in fp32 and rounded once to bf16) before its quantized segments,
whose kernels are the format's, while float segments read the input as it is;
the register tile prepares its table from the rotated rows, so the input's
producer writes plain rows. The token table gathers each row through the inverse
(`gguf_embed_rotated_pq20`).

A GGUF kernel of one quantized tensor names its epilogue last: `a` none, `r` residual, `g` the
up pass with the silu gate. The staged ones are `gguf_decode_<format>_m<rows>_<e>`, with gate and
up in one pass `gguf_decode_<format>_m<rows>_gate_up`, and
`gguf_prefill_<format>_<e>` (and `gguf_prefill_<format>_r_leading_inputs` over a view of the
leading inputs of wider rows), the register ones `gguf_decode_sg_<format>_l<lanes>_<e>`, and the
experts `moe_expert_gguf_m<rows>_<e>` and `moe_expert_gguf_sg_<e>`, with gate and up in one pass
`moe_expert_gguf_m<rows>_gate_up`; the fused projections run
`gguf_decode_fused_m<rows>` and `gguf_decode_sg_fused_l<lanes>` in decode and, on Apple10, in
prefill chunks of 33 to 64 rows (`gguf_decode_fused_m64`), and in longer prefill chunks one
`gguf_prefill_<format>_a` per format over that format's segments. The norm, GDN,
attention-gate and attention-gather variants that also write the register kernels' input
table carry `table16` in their names. The epilogue kinds of both GGUF families are in
`kernels/common/gguf_tile.h`, the SiLU and sigmoid every kernel shares in
`kernels/common/activation.h`, and the MMA helpers every register kernel uses, GGUF or fp32, in
`kernels/common/sgmatrix.h`.

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
also as column windows, the alpha/beta, norm, convolution and router bytes they write and the
golden images; then `gguf-dequant`, the staged tile's dequantizer, built with the production Metal
flags, against the half rounding of every reference weight; `gguf-rotation`, `gguf_rotate` and the
rotated PQ2_0 token gather bitwise against the fp32 butterflies and within one bf16 step of fp64;
`gguf-projection`, every GGUF projection through `ops::Linear` with each tile forced, so both
decode tiles run on every GPU, each in every format it takes (the register tile all but F16), at
one to four lanes, every K split and epilogue, fused segments, every format's gate with the next
format's up and the prefill tiles, each output inside the fp64 bound of
`GgufFormatReference.hpp`, the one-pass gate/up bitwise against the two passes at the same K
split, and one lane on the staged 16-row tile bitwise against the 8-row tile; and
`gguf-moe`: the float projections on both float tiles and the MoE layer on every GGUF
plan, the staged 8- and 32-row tiles and the Apple9 register tile whatever GPU runs it, each in
every format it takes as each routed and shared expert tensor, against fp64, bitwise against
the full grids and two gate/up passes, and with the experts in permuted slots across a floor
chunk and elastic chunks of other, few slots, the last of fewer, bitwise as in one chunk, and
refuses floor and elastic chunks one unit short, a malformed chunk list or chunks made for
another block; `gguf-moe-e512` runs all of it at Qwen3.8-Flash-Next's 512 experts and
`gguf-moe-sigmoid` at Flint's routing, a sigmoid router's top 6 and two ungated shared slices;
and `expert-store`, streamed experts at four shapes bitwise against the resident ones, in loads
of every kind, through a release and a restore, in a floor and three smaller elastic chunks,
beside a size of one block, which the store reads whole into its floor
chunk, and refuses a size its largest buffer holds in more than 48 chunks and a size of one
block it does not hold; `expert-chunk-layout`, in `make test-engine-cpu`, checks the chunk
layout rule itself. The goldens and how to regenerate them are
in `dev/tests/fixtures/weight-goldens/`; with `SPLASH_GGML_ORACLE=<libggml-base.dylib>`,
`gguf-reference` also compares the reference with GGML directly and prints GGML's hashes.

Two benchmark tools repeat the measurements behind the GGUF split law and tiers and the MoE plans,
with the weights DRAM-cold (each command of the projection benchmark reads a ring of weight copies
of at least 384 MiB, however small the tensor). `make benchmark-gguf-projection
GGUF_PROJECTION_ARGS='q4k 5120 8192'` times one projection (up to three fused formats and widths,
then `K` and an optional epilogue) on both decode tiles at one to four lanes (one lane also on the
staged 16-row tile, `staged16`; a gate/up pair on the staged tile in two passes, `staged`, and where
the walks spread also in one, `pair` and `pair16`) and every K split, and marks the device policy's
pick; widths emulate core counts, since the policy counts threadgroups per core. Or, with a trailing
`prefill=R[,R...]` after the epilogue and the round count, one format's 128-row prefill tile at each
chunk of `R` rows (more than 32), and for `R` up to 64 the staged 64-row tile at every K split; the formats `float-bf16` and `float-f32`, and `float-<format>` for
a quantized format's native rows (such as `float-af8g64`), time the float projection
(`ops::addGgufFloat`, widths of any multiple of 8) on both float tiles instead, marking
`Linear::ggufFloatTile`'s pick; `make benchmark-gguf-moe` times one MoE layer at the 35B shape,
or at Qwen3.8-Flash-Next's or Flint's (`GGUF_MOE_ARGS='flash-next'` or `'flint'`), on the
device's plans and the other GGUF tile.

Target attention runs the geometries of `kCompiledAttention` (`runtime/ops/PagedAttention.cpp`):
query heads over KV heads of a head dimension (`_d128` in the names of 128-wide heads' kernels;
256 is unnamed) in a value order, each compiled with the variants its layers take, which a layer
selects with an `ops::AttentionVariant`. Its queries and keys are rotated by the geometry's
RoPE, or not (NoPE: `qkv…_nope`, which binds no tables); and each row attends its whole history,
or a window of the keys before it (`…_w512`: a row at position p attends p − 512 to p). A
windowed split and reduce start at the page of their tile's first row's window start
(`splash_attention_first_page`), each row masking the keys before its own, and a windowed verify
lane's or prefill tile's splits stop at the 18 pages such a window reads
(`kv::windowAttentionPages`). A geometry with learned sinks (`reduce…_sinks`) binds the layer's,
an F32 per query head, to the reduce, which adds each to its row's softmax after the splits, in
their order, as a key with no value; one without an output gate (`gather`) gives the
out-projection its attention rows as they are. The RoPE tables are of the target geometry's pairs
(`rope_build_tables`, or `rope_build_tables_p64` for 64). `make test-engine-metal` runs
`attention-variants`, every variant through the production encoders against fp64, and
`make benchmark-attention-sweep ATTENTION_SWEEP_ARGS='--shapes flint-window,flint-full'` times
Flint's two kinds of layers with their sinks: a window of 512 keys with RoPE, and the whole
history without.

## Sparse attention

Qwen3.8-Flash-Next's full-attention layers attend sparsely (QSA): past 512
complete blocks of 4 tokens, a query row attends only to the 512 blocks its
indexer scores highest and to its tokens after its last complete block. The
indexer (four query heads and one key head of 128) keeps each block's pooled
key, bf16, in its page's index, in the index plane of the page's extent apart
from its K/V (`PageStorage::indexOffset`), which its kernels reach through each
lane's index table (`ops::BlockIndex`, `DecodeTensor::IndexTable`), written as
its page table is: a page whose K/V went to the SSD keeps its index
([KV on the SSD](server.md#kv-on-the-ssd)). A step projects its rows' indexer queries and
keys, pools the blocks they complete from its rows' keys and, for the tokens
of a block before its first row, the sequence's pending keys, scores every
complete block for each row in one MPP product per tile of 8 rows by 32
blocks, and selects each row's blocks into a bitmap by a radix select of its
scores, 11 bits a pass, the first over the row and the later ones over the
threshold bin's scores in threadgroup memory, the bitmap written a word per
simdgroup from ballots (`kernels/shared/block_indexer.metal`). The pending keys,
the raw keys of the up to 3 tokens of the block a sequence has not completed,
are its state's (after the PLE history in a state cell,
`QwenTargetDimensions::indexerPendingBytes`): a prefill writes them after its
rows and at each state it captures, and a verify's commit after the rows the
lane retained, so a block a rejected row completed is pooled again by the
next step from the committed tokens. No page holds a raw key. Queries and
pooled keys rotate by the same target RoPE row (`target_rope_pair`, the
queries' from the step's tables, a pooled key's at its block's first token),
with YaRN's frequencies and attention factor when YaRN is on. A row's
scores and bitmap cover every block of the window the model serves; a prefill
chunk scores its rows in batches that keep the native window's score scratch,
256 rows at 256K, 64 at 1,010,000.

A layout that keeps values token-major (`kv::Layout::valuesTokenMajor`)
attends sparsely, on kernels compiled for 24 query heads over two KV heads
(Qwen3.8-Flash-Next's): a query row attends only to the blocks of 4 tokens its
block mask names within its causal history, the mask a bitmap per row over the
history's blocks (`metal/abi/PagedAttention.h`). The rows of a tile (a verify
lane's 8, or 8 of a prefill chunk's) attend through one list, the union of
their blocks in block order, each entry a block and the rows of the tile that
attend it (`attention_block_lists`, `ops::PagedAttention::addBlockLists`).
The split kernels (`kernels/shared/sparse_attention.metal`) share a list out
among their splits in units of 8 blocks, at least 48 a split
(`SPLASH_SPARSE_SPLIT_BLOCKS`), so that a lane's split count follows its own
list; each step gathers 8 blocks of keys (4 in BF16), and then their values,
into threadgroup memory for the MPP QK and PV of the page loop, and the
softmax masks each row's tokens in blocks the row does not attend and past its
causal end. A tile without a list attends every block of its history.
Token-major values make a block's values one contiguous run as its keys are;
the dense page loop keeps them dimension-major, on which its PV runs faster. A
layout may also keep an index of its keys (`kv::Layout::indexDimension`), per
layer each block's bf16 pooled key, in the index plane of its pages' extents
(`PageStorage::indexOffset`, `indexSpans`), part of the page wherever the host
copies or clears it (`PageStorage::spans`).
`make benchmark-attention-sweep ATTENTION_SWEEP_ARGS='--shapes fn --qsa shared'`
times the sparse kernels over synthetic selections (`--qsa` also takes
`independent`, `overlapping`, whose unions are the ~1,500 blocks real
selections make, and `pages`), and `--qsa indexer` the indexer itself before
them; each case reports the mean entries of its lists.

Where some of a verify's listed blocks are on the SSD, a host step after the
layer's selection resolves each lane's list through the hot block cache
(`KvBlockCache::addResolve`) into a location per entry: the block's place in
the cache, or `SPLASH_ATTENTION_PAGED` for a block its page holds. The located
split (`AttentionBlocks::locations` and `locationPages`) reads each block where
its location says, and a command none of whose lanes reads spilled KV encodes
the splits as before. A prefill whose history is on the SSD attends through a
staging of one layer's pages, ordinary entries of a one-layer
`kv::PageStorage`: the history stream reads the spilled pages into it, and the
stage kernel (`PagedAttention::addStageLayer`, `kernels/shared/kv_stage.metal`)
copies those still in the pool, so the prefill kernels run unchanged.
`attention-sweep --located` moves the listed blocks into a block cache and
times the located split against the default one.

A block-sparse target declares its attention to the engine
(`model::SparseKvShape`, `ModelCapabilities::sparseKv`): its blocks' tokens,
the blocks a row selects and its verify rows, and the rows that read whole
pages instead, those with at most `denseHistoryTokens` of history and its
first `sinkTokens` and last `localTokens`. The memory plan sizes the hot block
cache and the floors of [KV on the SSD](server.md#kv-on-the-ssd) from it, and the
cache keeps whole the pages those rows read. A target that declares
`localTokens` keeps its KV in memory: the cache keeps whole only the dense
history on every path, not each holder's last pages. Its family code reaches K/V that
left memory through the command's `model::SpilledKvReads`: `verify` after a
layer's block lists, which locates each listed block of a page that holds
only its index in the hot block cache for the attention; `prefill` for each
chunk, which returns the table and layer of the history staged for it; and
`layerEnd` after a prefill layer, which hands the history's reads on to the
next. A command none of whose requests holds such a page encodes as without
them.

## Neural Engine prefill

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
ANE's weights from the block planes one layer ahead into double-buffered
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
the start warns how much context any split leaves. The
[SSD cache](server.md#ssd-cache)'s automatic state staging gives way to the split: where
the fewest channels fit only without it, the cache turns off. Without
`--max-context` the engine serves the automatic context, which while the split
may run is what the plan holds with the split this Mac's calibration of the
model takes (below), the GPU alone's if it found no gain, whatever the start
then makes of it: a split that failed leaves the GPU alone at the same context,
and the memory the split would hold stays with the KV cache. Every start of a
model on a Mac that takes one calibration so serves the same context; one
calibrated anew may serve another, after the split lost to the GPU alone, which
forgets the calibration (below), after a start whose calibration failed and
left none, or once macOS cleared the directory that keeps it. The server's
relaunch of the engine after a failure keeps the first engine's context
whatever it calibrates: the server passes that context
(`serve-native --relaunch-context`), the relaunch decides the rest as the first
engine did, the SSD cache's staging among it, and its split takes at most the
channels whose plan holds the context, calibrated within them and remembered
for them, or leaves the GPU alone with a line, not the warning a refused
`--max-context` gets. A relaunch whose plan holds less, as once the Mac's
memory budget shrank, serves that, and the server stops relaunching
([engine restarts](server.md#engine-restarts-and-sleep)).
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
remembered calibration among them, the automatic context, the same in each
where the split may run, and a relaunch's, which holds the first engine's
context where a calibration made anew would serve another.
`make test-engine-metal` runs `ane-ffn`: its kernels
against CPU references for every image format under shader
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
the model runtime oracle with the GPU alone, with the GPU alone over a BF16
state (`--state-format bf16`), and with the split as a start makes it, which
must not be unavailable, and whose automatic context must be what the plan
holds with the split it runs.

## Host steps

A command can run CPU work between its GPU work. A host step
(`metal::EventStep`, `CommandGraph::host`) runs once all the work before it
has completed, on one thread of the backend's, and holds the work after it
until it returns, through a shared event of its command's: it reads what the
GPU wrote before it and writes Shared buffers the GPU reads after it, and it
must not call the backend. A step after a failed command buffer is skipped,
and one that throws fails its command with its message. A step's `then`, if
any, runs on the same thread once the step let the work after it go: CPU work
the GPU need not wait for, such as starting reads that would hold the GPU's
start back if they began as the step ends. It does not run after work that
threw, and must not throw: the command may have completed by then, so a
`then` that throws marks the backend unhealthy. A step's work may run after
submitCommandAsync or wait() threw (its encoding failed after its first
buffers were committed, or the watchdog or a shutdown gave the command up),
so the work owns what it touches.

Metal fails a command buffer whose event wait is still unmet 5 s after its
queue reached the wait ("Caused GPU Timeout Error"), and with it the command
and the engine. The backend commits a command's buffers as it encodes them, so
the queue reaches a command's first host step, or Neural Engine wait, as the
command is submitted, whatever GPU work runs ahead of it, and each later one
once the one before it was met (macOS 27, M5 Max and M5 Pro; a wait committed
behind 4.5 s of work and met 1 s after it fails at 5.0 s). The work between
two steps, with the step, must therefore end within 5 s, which the
scheduler's prefill slices keep with room to spare. Work outside them runs as
a command of its own: a prefill command's image encodes, seconds of GPU work
on a small GPU, run first, and the prefill command is submitted once they
have completed ([Vision](models.md#vision)).

A step about as short as the GPU's wake-up can keep the GPU awake: one GPU
thread spins through it until the CPU ends it, so the work after it starts at
once and at full clocks. The spin sees the CPU end it only through a memory
scope Metal's public headers leave out (`runtime/metal/kernels/host_step.metal`),
so it builds apart from the metallib, into `build/host-step.metallib`, and a
Metal compiler that refuses it only warns. Before the first command with such
a step, the backend checks that the spin sees the CPU end it; without the
library, or if it does not, the steps run without the spin, with a warning.
A completed command reports each step's round trip (`HostStepTiming`) and the
time its GPU work waited on its steps (`CommandTiming::hostSeconds`), which the
scheduler counts as the command's fixed cost
([Admission and eviction](#admission-and-eviction)).

## Residency and KV extents

Every buffer the backend allocates belongs to one residency set
(`MetalBackend::allocateBuffer`, `metal/Residency.hpp`): weights, KV extents,
state buffers and scratch alike stay wired between requests until the idle
release (`--idle-release`, 10 minutes by default) passes without a command, and
the next command wires them again. The command queue holds only a set of the KV
extents, which kernels reach by address: Metal makes them resident for every
command buffer, as it does each buffer a dispatch binds. A queue that held every
buffer would have each command buffer wait for all of them before it starts,
which with memory nearly full takes 100-170 ms for 1-2% of them; a streamed
decode command, whose host steps start about a hundred a cycle, would wait at
nearly every cycle. `--idle-release off` keeps every buffer wired, and the
weight images allocated unless an [explicit release](#explicit-weights-release)
gives them back, while the engine runs; KV extents return only through the
reclaims below, memory pressure among them. Memory returns to macOS when the
engine releases it, never because macOS compressed or dropped an idle buffer.
macOS page cache, driver allocations and other applications still affect memory
pressure and swap.

A buffer a dispatch binds is resident for it; one kernels reach by address is
made resident in one of two ways. KV extents (`BufferReach::Addressed`) join
the command queue's set, since nearly every command reads KV. A dispatch may
instead list buffers in an address table (`metal::AddressTable`,
`CommandGraph::addresses`): the backend writes their GPU addresses as it
encodes the command, after the dispatch's parameters (`setBytes`, so at most
512), and makes them resident from that dispatch to the end of its encoder
(`useResources`, for reading, or reading and writing, declared once per table
and usage in each encoder), never for the queue.
The addresses are those of the buffers' memory when the command is encoded, so
a buffer released and restored since (which comes back at another address) is
reached where it is, and a table that reaches released memory is refused like
a binding of it. The routed experts are reached this way, the expert store's
chunks as a resident model's routed planes: a decode command reads few of
them, and in the queue's set the 18 GiB of a streamed 35B's chunks made every
command buffer wait for all of them, a decode cycle ten times longer.

KV pages live in extents: ordinary shared Metal buffers of one size per pool,
between half and one and a half times 128 MiB, in which every tensor region of
each attention layer starts 64 KiB-aligned, sized to leave the fewest of the
budget's pages unused (`Layout::extentPagesFor`). The regions follow the
layers, except that a layout with a sliding window (`Layout::windowTokens`,
over the layers `windowedLayers` names) places the regions of the layers no
window bounds first and then the windowed layers', each in layer order
(`Layout::region`): what a row reads of a page every window has passed lies in
its first regions, and a page's SSD slot holds them first too. Each extent has
a form (`kv::PageForm`), which holds the first of a Full extent's regions
(`Layout::regions`) and then, for a layout with a block index, their index
plane: a Full extent every layer's; an Index extent none, only the plane, for
pages whose K/V is on the SSD, and only a pool that spills allocates one; and a
Slim extent, only of a layout with a sliding window whose slots hold a page's
spans in order (`Layout::slimmable`), those of the layers no window bounds, for
a page every row's window has passed. A layer's region sits at one offset in
every extent that holds it, so the kernels reach a Slim page as they reach a
Full one, through a real page table entry; only an Index page's is absent. A
page moves to another form, its references with it, when its K/V leaves memory
or comes back, or when it gives up its windowed layers' K/V
(`KvPool::movePages`), and the copy takes what both forms hold. The pool
allocates an extent when it needs one of its pages. When it is built it
allocates the runway, the extents of the first 64 pages, which startup warmup
runs on; nothing but the pool allocates or releases an extent. An extent whose
last page is free stays allocated until a reclaim releases it, at once and only
between commands: memory pressure, an admission the budget denies, the
publication of a replay point in use or a junction, or startup cleanup. Kernels
reach a page through the GPU address in its request's page table, so no command
binds KV; the command queue's residency set makes extents resident for every
command buffer. The host reaches the same memory (`PageStorage::spans`), which
is how the disk tier moves pages. A reclaim returns free pages before it evicts
anything: an empty extent as it is, and the free pages scattered over the others
of its form as soon as they cover the extent of that form that holds the fewest
pages, whose pages the pool copies to them (`KvPool::compactExtent`): a Full
extent before a Slim one and a Slim one before an Index one, unless the reclaim
needs the memory of one form, as a shortfall of Index pages needs Full memory.
The blocks and requests on those pages follow them, a page a disk transfer reads
or writes stays where it is, and a pass that evicts everything copies only what
is left afterwards. Every extent a pass empties is released, except that a
warning pass, like startup cleanup, keeps one empty extent as the runway the next
request starts from, and a pool that spills keeps an extent's worth of free
Index pages, which spilling draws on first, so that a page spills at the memory
wall.
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

## Admission and eviction

`/status.admission` distinguishes memory and concurrency waits, counts the
requests held back behind one refused memory (`held_behind_refusal`; during
recovery only the suspended ones and those of a strictly higher priority, and
the refused request itself while a pass defers it, even one that recovery does
not try, which still waits for memory) and those waiting for a disk restore
(`restoring`), reports suspended requests, recovery draining and the oldest
current wait age, which for a request holding admission closed runs from when
its wait began. Memory transitions also appear in the console. Growth that no
request in service needs, and so the cache, leaves other applications their
memory: it keeps a spare reserve of 7% of the Mac's memory (2 to 7 GiB) and a
1 GiB margin free of the host's spare memory, the free, file-backed and
purgeable pages and half of what compressing the anonymous pages would free at
the compressor's present ratio, at most a quarter of them. When the spare memory
runs short of that, such growth pauses and the cache gives memory back, a paced
pass at a time, down to one lane's state buffers and one KV extent. A pass that
no request waits for keeps the state a follow-up resumes from and takes nothing
in use (below), such as the replay point a request in service has just
published: what it would free there would only go back to that request's
growth. So it does, however much spare memory there is, while macOS reports
warning memory pressure, its signal for caches to give memory back: each pass
then returns up to 1 GiB until the level is normal again, still taking nothing
in use unless a request waits, and keeps the state a follow-up resumes from,
even while a request waits, whose growth the warning refuses as well (`/status`:
`memory_governor.system_pressure`, and `growth_allowed` is false meanwhile). A
pass counts only memory that leaves the engine; a cached state it evicts first
refills the lane's state buffers it keeps. Where routed experts stream, a pass
the host is really short for, its headroom measured short of the pass's margin
or macOS reporting critical pressure, also takes the [expert
store](server.md#streamed-experts)'s chunks above its floor, a chunk at a time
while its target is unmet, once nothing outside what is in use is left to free
and before anything in use; the experts a chunk held move to the chunks below it
or are evicted, and later steps read them from the SSD again. Memory on its way
back holds the store back: while cache nothing in use holds waits for a
transfer, a KV demotion or a state write, the pass ends pending, and the pass
that continues it once the transfer landed takes that cache first. macOS's
warning alone never takes the store. Once memory allows again the control
pass takes one chunk back per pass, its slots empty: when macOS reports no
memory pressure, no request waits for memory, the host lets growth go on, the
weights are neither released nor being restored, and 30 s have passed since a
chunk last went. The chunk takes from the host only its headroom beyond the
2 GiB recovery margin, and from the engine only room `--max-memory` leaves
free. While the engine holds any request, resident, queued or draining, it
takes only that. With none the chunk is an exchange for cache nothing in use
holds, which ranks below the store, for what the chunk lacks of either: the
engine first returns that cache to the system, one reclaim step at a time,
each releasing the extent it empties, until the bytes that left the engine
cover what the chunk lacks, and keeps the resume point of the newest
conversation. The governor's growth admission then counts those bytes as the
host's already (`MemoryGovernor::setGrowthCredit`): its own sample of the host
may not show them yet, and the pass checked the rest against the sample it
took before the exchange. Cache that cannot pay for the chunk in that pass
stays. The exchange starts no copy to the SSD cache, whose memory would come
back only after the pass: a state or page that needs one stays, and so do the
pages above that page and the newer states of that state's class (checkpoint
or ordinary). Unless the idle state buffers beyond a lane's and the extents
that what leaves at once would empty cover what the chunk lacks
(`Engine::exchangeableBytes`, `Cache::exchangeable`), the exchange takes
nothing and the chunk waits; it waits too while a transfer is in flight. With
the SSD cache on, a cached conversation buys a chunk only once its state and
pages have copies there, as `--persistent-cache` writes them. A start
needs only what macOS can
hand out at all beyond a reserve of a tenth of a small Mac's memory and 2 GiB of
a larger one, counting what compressing the anonymous pages would free at the
compressor's present ratio, at most half of them, short of critical pressure:
what it allocates, the KV runway, arenas and warmup included, is memory a
request in service needs until serving begins (`/status`:
`memory_governor.host_available_bytes` and `host_reserve_bytes`; the spare
memory and its reserve are `host_spare_bytes` and `host_spare_reserve_bytes`,
and `host_headroom_bytes` is the room above them). A request in service keeps
growing through host pressure and macOS's warning, short of critical pressure,
within `--max-memory`, first into cached pages no request holds:
it takes cached KV, and a cached state only where it sits on the KV that goes
next, since a state's buffers give it no page. An allocation the host refuses
takes none of the caches the model can rebuild (the vision encoder's scratch,
images' encoded rows, the readers of spilled KV), which would not get it past
the host; the paced passes take them. A new request waits while another
is in service unless it can start from what the engine already holds, and with
none in service it starts. A request whose start was refused memory, under host
pressure or at `--max-memory`, holds back the requests that arrived after it
until it starts, fails or is cancelled, so the lanes that finish leave their
memory to it; a higher priority is not held back. So does a start that waits
for the SSD cache's room for its KV, which other long requests hold. It keeps
holding them back while it waits for a prefix another lane is computing.

A prompt near the context limit can hold back every other start of its priority
or below until it ends, its prefill alone taking many minutes: in service, its
KV grows into the host's spare memory, which pauses other growth; a start beside
it takes only what the engine already holds, such as a lane's state buffers that
a finished request left in the pool; and the first start refused holds back the
rest, its wait's limit restarting while the long request works (below).
`/status` shows it: `admission.held_behind_refusal` counts the requests held
back, `memory_governor.growth_allowed` and `host_headroom_bytes` show whether
and how far the host lets the engine grow, and `state.idle_gdn_cells`,
`idle_draft_rings` and `idle_context_windows` count the pooled lane buffers a
start can take. A request's `timeout`, or `--request-timeout`, bounds the wait.

After a suspension, suspended requests resume before new work of their priority
or below, one at a time, higher priority first and then in the order they
arrived; one refused memory holds back the suspended requests after it, and
their resource waits do not run out while it does. When every lane is taken,
or a new request's start is refused memory that no copy being written to disk
is about to free, the resident of the lowest priority below it is suspended for
it (`cache.priority_suspensions` in `/status`) and resumes once a lane and its
memory are free. A new request below the priority of a running lane neither
starts nor takes a lane until no lane of that priority runs, since it could not
run before. Critical pressure evicts every unpinned cache entry and stops all
growth; it takes the expert store down to its floor after what nothing in use
holds and before what is in use, in one compaction per format. Refused at
`--max-memory`, a lane's state, or the model's memory for reading KV from the
SSD cache, takes cached KV only while evicting it can empty an extent, and is
tried again only once an extent or a state has gone. Any growth `--max-memory`
refuses, a start's beside a resident request too, takes a chunk of the expert
store above its floor once nothing outside what is in use is left and before
anything in use, and is tried again; so does the prompt's replay point before
it recycles another request's state in use. Growth that only shortens a command
(the KV a prefill captures past its first state boundary) or leaves memory for
later takes neither the store nor anything in use, a junction or checkpoint
capture never takes the store, and neither does a refusal of the host, which
the paced passes answer.

A request keeps its reusable model state at the last whole 32-token page before
its generation prompt, the text a chat template appends to open the reply: the
next turn may render it differently, so a follow-up resumes from there.

Until the request ends, suspended or not, that replay point is in use, and so is
the KV it restores through. When the last request using it ends, the point
becomes the newest ordinary state; the older points on its chain keep their age.
Cache victims come in three classes: checkpoints, then ordinary states and KV
(first the KV no state restores through, which saves no prefill), then what is
in use. No work displaces anything of a class above its own. The expert store's
chunks above its floor rank between the last two: they go once nothing ordinary
is left, before anything in use, for work that may take what is in use and for
needed growth whose class stops at ordinary, but never for growth that only
shortens a command or leaves memory for later, which takes nothing in use
either. One exception is kept: a paced pass a request waits for under macOS's
warning alone, the host not short, takes what is in use while the store keeps
its chunks, since the warning alone never shrinks the store. Memory for running
requests takes what is in use after everything else, and so does a paced pass
that a request waits for. Three kinds of work take nothing in use: a paced pass
that nothing waits for, under macOS's warning too, and the two that follow. A
start that a resident lane holds back waits for that lane. And when no lane's
growth fits, the lane that yields first, such as one just started beside a
decoding lane, pays for the memory with its own suspension; while other lanes
fit, it waits for them, resident, as any lane refused its KV does, since they
grow too and are refused in turn. Where KV spills, a refusal they do not share,
of the model's memory for reading KV from the SSD cache (a decoding lane's share
of the hot block cache, a prompt's staging), which no transfer in flight
returns, suspends the lane that yields first if that lane yields before it and
is resident outside its command. A prefill command captures the state of
each boundary it reaches, in buffers reserved before it runs. A junction or a
checkpoint retires its lane's checkpoint once published, and a checkpoint the
same command would capture before it is not captured at all; the prompt's
replay point keeps the lane's checkpoint cached. Until a state retires it, that
checkpoint is the lane's recovery point, which a command that is cancelled,
expires or fails leaves to a retry, so no state takes its buffers, or room a
reclaim could make of it, before its command. Refused the pool and an admission
while its lane holds a checkpoint no reader pins, a state ends its command at
its boundary and is taken from the lane once the command has completed: a
junction or a checkpoint in that checkpoint's buffers, which it takes first,
the replay point in them only when refused again. A checkpoint only on disk
gives no buffers, and stays until the state is published. A checkpoint whose
predecessor a reader pins ends its command too, and is published only once the
pin has gone. Refused, the replay point, which is in use, takes cached KV and
states in the same order, then the oldest state in use; of the KV it takes only
leaves whose page frees at once, and only while an extent can be emptied, which
is released at once; it drops a state whose write must wait for the write in
flight. Refused then by the host alone, it takes its buffers as memory the
request in service needs, as that request's KV does: a headroom between the
1 GiB margin and the margin plus one state would otherwise refuse every replay
point once no cached state is left to recycle. A junction makes room the same
way short of what is in use, and an optional checkpoint takes only
checkpoints; neither drops a state whose write must wait, and the host's margin
refuses both. A state whose lane holds no such checkpoint makes that room before
its command, and still refused ends its command at its boundary. With the SSD
cache, a state its command could not capture and memory still refuses is
written from the lane unless another write is in flight. A disk copy in use may
displace the oldest copy in use, and ordinary or optional work never displaces anything in use. Nothing
in use is pinned, so running work that needs the memory still takes it once
nothing else is left. A resumed lane that lost its prompt's replay point
rebuilds it on the way; the point its generated history reaches is a disposable
checkpoint.
`/status` reports under `state` the replay points unfinished requests hold
(`in_use`, zero when idle) and those evicted all the same (`in_use_evictions`).

Requests sharing a cold prefix can wait for a resident request's planned recovery
point, including one whose prefix is still being restored from disk, then enter
through the ordinary cache restore path. Waiting requests hold no lane or KV
pages and return to ordinary admission when no useful producer remains. A
producer's command ends at the boundary a waiting request resumes from. Late
arrivals can extend the plan where a whole draft window can still be captured
for the new boundary.
Higher-priority work does not wait for a lower-priority producer. `/status` exposes
`scheduler.waiting_prefix` separately from resource waits.

Greedy, sampled and constrained requests share decode batches; each lane keeps
its own sampling policy and RNG, and a greedy lane takes the argmax path in any
batch. A batch with a constrained lane runs its cycle as three commands around
the host mask exchange: the draft, the target forward while the server computes
each constrained lane's masks from its proposals, and the selection, in which
only the constrained lanes read masks. A mask request the server leaves
unanswered for 5 s fails that request with the retryable `mask_timeout` (HTTP
503), so a stalled grammar cannot hold the batch's command.

Long prefill uses disposable rolling checkpoints every 4096 tokens; none is
planned within one prefill chunk (2048 tokens) of where the request resumes or
of its replay boundary. Each retires only once the next one or a junction is
published, which takes its buffers, when memory refuses others, only after its
command has completed: a request whose command is cancelled, expires or fails
keeps it, and a retry resumes there. One only on the SSD retires when that
state lands.
The last stays cached once the replay point is published: a later request that
shares the prompt up to it, such as another conversation with the same long
system prompt, resumes there instead of from the start. It stays only if the
replay state fits beside it; refused memory, the replay state takes its buffers
rather than another lane's checkpoint. With the SSD cache it costs a write as
well: reclaim takes checkpoints first and writes the states it evicts to the
SSD, this one too (153 MiB for 27B), and one published straight to the SSD,
when no RAM slot took it, stays there. These writes draw on the SSD cache's
write budget when one is set, which drops them while it is spent, a suspended
request's own too, and a full quota makes room for one by dropping the oldest
copy, perhaps an older conversation's restore point.
Contended prefill adapts toward a 500 ms slice, keeping
2048-token chunks for long unopposed work. A command's estimated cost is a fixed
part, the time its GPU work waited on host steps (a streamed target's expert
reads), and a part per row; the slice grows to twice the fixed part, so a
command's rows take at least as long as its reads. Unopposed, a target whose
experts stream runs its remaining chunks in as few prefill groups of up to four
as hold them, evenly split, and so does a request whose KV spills to the SSD
cache, which reads its history once a group; one whose KV stays in memory on a
target whose experts load whole runs chunks, so a request that arrives beside it
waits for a chunk. On a target whose experts stream a request keeps its groups
whatever contends, and a command whose estimate exceeds the slice runs alone in
passes over the model's layers, as many as keep each within the slice: its
later passes come before any other prefill or admission, decode commands run
between them as the decode share owes them, only the last commits its rows,
and a request that ends between passes leaves the rest unrun. A lane whose history holds KV only the SSD cache has
reads all of that history in each command, whatever its rows: it runs alone,
and contended it takes the fewest rows, halving from its group's, whose time
still covers its own running estimate of that wait, and never fewer than another
contended command; the wait stays out of the command estimate. Without
contention, a prompt that can
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
cached prefix was evicted and the expert store shrank to its floor, fails with
400 `capacity_exhausted`, naming
`--max-memory` and `--max-context`; retrying it fails the same way.

A cached state holds the target's GDN cell and the draft's context window, not
the draft's rings: the context row of each of the 2048 positions the rings
cover, from which every draft layer computes its keys and values, in 4-bit
codes over groups of 64 values with each group's fp16 scale and minimum
(`DraftStateLayout::windowBytes`: 5.6 MiB for 27B and 2.25 MiB for 35B, where
the rings take 40 and 48 MiB). Prefill keeps a lane's window beside its rings;
a restore that continues the rings computes them from the window again at the
start of the request's first prefill chunk (`DFlashDraft::addContextRebuild`).
Decode writes the rings alone, so states are published only in prefill. A
prefill chunk writes each state it captures inside its rows
(`ModelBatchItem::stateCaptures`) into buffers reserved before it runs: every
GDN layer's dispatches stop there to store the state and go on from it, and
the state's window is a copy of the lane's as the chunk finds it plus the
chunk's rows before the boundary, since the chunk's later rows take the slots
of the state's oldest rows in the lane's window. A state at the chunk's end is
copied from the lane after it. A target without GDN layers keeps no GDN state:
its cells have no bytes, every lane and state shares one empty cell that holds
no memory, so a lane holds the draft's rings and window and a cached state the
window alone, and `state.idle_gdn_cells` stays 0.

## Sliding windows

A target whose layers include sliding-window ones (`kv::Layout::windowTokens`,
over the layers `windowedLayers` names) pages every layer, and a row of a
windowed layer reads only the `windowTokens` keys before its own. Once every
row that will run has passed a page by a window, its windowed layers' K/V is
read no more, and the page gives it up for the Slim form, which keeps the
layers no window bounds (`kv::PageForm::Slim`): for Flint 14 of 49 layers, 462
KiB of a page's 1.58 MiB in int8. A cached state needs its tail, the
`windowPages` blocks that end at its block (`Layout::windowPages`, 16 for a
512-token window), Full, since a request resuming there reads them whole: so a
cached prefix restores at a state as ever (`Cache::restoreRequest`), and states
closer than a window share the pages of their tails. The cache keeps it so
(`CacheWindowPolicy`):

- Between commands, before any lane grows, `Cache::trimWindows` moves to the
  Slim form every Full page that every request holding it has published
  `windowPages` blocks past (its next command's rows, at or past its published
  blocks, read no key on it), that no state's tail holds and that no transfer
  moves, the cached pages no request holds and the pages requests keep beside
  their blocks among them. The Slim runway's free pages take them first,
  then extents memory allows: while a request is in service, memory it
  needs, as its KV is, so the host's pause does not hold them back and a long
  prompt's passed pages trim as it prefills. A refusal leaves pages Full for
  the next pass: memory, never correctness. A multi-pass prefill publishes
  only once its last pass ends, so its rows' windows stay Full between
  passes.
- Every state, in RAM or on the SSD, pins its tail (`KvCache::countState`, a
  bounded walk beside the counts of the states below); only its removal
  unpins the tail, which the next pass trims: a state written to the SSD
  keeps its entry and its tail Full, on a request's own chain too, where the
  plan's floor (`kvWindowFloorPages`) does not count it.
- A writer that computes a block again keeps its own page, which the block
  takes once no transfer moves the block's (`KvCache::insert`,
  `kv.window.upgrades`), rather than moving onto the block's page as a
  writer of another layout does: its window reads the page whole, which a
  Slim page no longer is, and were it to move onto a Full one, what it reads
  would depend on whether trimming had taken the block's windowed layers.
  Requests that hold the old page keep it, as a page of their own the next
  passes trim once their windows have passed it. A copy of a Slim page goes,
  there and where a disk-only block takes a writer's page: a block's copy
  holds at least what its page holds, should the page leave.
- A state is published only on a Full tail (`Cache::stateTailFull`), which a
  command leaves where it computed or held the tail in its window; the engine
  counts one it refuses as a failed publication. A start takes back a
  persistent restore point only with Full copies of its tail.
- Each cut of a request's page table checks that every page from the window
  before its published blocks on is Full, always (`Cache::cutPageTable`).
- A KV slot holds what its page holds when it is written, and its record
  names the form (`KvSlotLabel`); a restore allocates that form. A restore
  point persists its tail Full, as RAM holds it.

So no byte a kernel reads changes: a run with trimming off
(`--kv-window-trim off`, the native argument validation passes) emits the
same tokens and logits wherever memory lets it cache what a run with
trimming on caches, only taking the memory of Full pages. The memory plan
counts on trimming, with it off too, so a validation run keeps within what
Full pages hold: one request's KV capacity is the Full pages its lane keeps,
its window's, the pages a command writes (a chunk, or a group of four where
the routed experts stream) and one more (`kvWindowFloorPages`), in whole Full
extents, and the pages of as many whole Slim extents as the rest holds, since an
extent holds pages of one form; the pool's page ids cover the Slim pages the
hard budget could hold. A Full shortfall takes trims first, then, while the Slim
pool has room for their tails, states, a checkpoint first, each followed by a
trim of the tail it unpinned, while those trims free pages, before any KV leaf
goes (`Cache::reclaimForPages`). At the memory wall, where the Slim runway
cannot refill, no state goes for its tail, and the victims go as without a
window, dead leaves before ordinary states. A state the SSD cache keeps unpins
nothing: each step that keeps extents (`CacheReclaimMode::KeepExtents`) writes
one more state there before any leaf goes. Empty Slim extents beyond the runway
go for the retry to grow by.

## Disk tier

The tier holds cached request states (GDN cell plus the draft's context
window) and KV pages.
RAM and disk copies share the same block tree and recency order. Restoring a
prefix keeps its disk copy, so its next eviction needs no write while that copy
remains cached. The KV tier takes no Metal memory. The state staging buffer,
one state, is a buffer of the backend's like every other: resident, and set
aside by the memory plan within `--max-memory` when the tier starts. A quota too
small for one state leaves the tier disabled and sets nothing aside, and so
does the automatic quota where the plan has no room for the staging beside the
model and the requested context, so a start the staging would fail goes on
without the tier, and where the plan refuses the Neural Engine split beside the
staging but not without it (`aneFfnRefused`): the split wins. The disk plays no
part in a dense target's plan, so it changes no start's context; a block-sparse
target's first start takes its room for active KV into its context
([memory and context](server.md#memory-and-context)). The server's relaunch of the
engine decides the staging as its first engine did and keeps that engine's
context ([Neural Engine prefill](#neural-engine-prefill)). The tier's files
open after the plan, where the disk takes them; where none open, the tier turns
off and the staging the plan set aside stays unused. The automatic quota opens
them only on a volume with room for the tier (`automaticDiskRefusal`, reading
volumes through `RuntimeResourcesConfig::diskVolume`): the temporary files',
where it goes without a persistent cache or whenever the root cannot be used,
and a persistent tier's root, with the root's caches nobody holds counted as
the tier's, since the tier may take their place. A root that refuses, or whose
open fails, leaves the tier temporary files if their volume takes it. A
relaunch whose plan spills takes temporary files on any local volume it can
measure, whatever room it has: the first engine's context it keeps may rest
on the tier, and each request's commitment checks the room
(`Engine::admitSpill`).

The volume sizes the tier only once its files are open, before a persistent
tier takes anything back, and again at every control pass and whenever a write
finds the volume full. `DiskSpace` reads the volume with `fstatfs` on the first
file's descriptor, which names the right volume however the file was made, and
sets the budget's limit to min(quota, max(0, available + held - floor) / 4) for
the automatic quota and to min(quota, max(0, available + held - floor)) for a
size, held being the blocks the tier's files take (`st_blocks`). The KV of
requests in service, the copies of the KV they hold and the commitments of
spilling requests, may use the latter under either
(`DiskBudget::activeLimitBytes`, [KV on the SSD](server.md#kv-on-the-ssd)). The tier's
own writes and punches move bytes between available and held, so they leave the
limit where it is, while space anything else takes or frees moves it by a
quarter, or for a size by as much. Several automatic tiers on one volume each
settle at (available - floor) / (tiers + 3) of what was available before them,
and blocks a local snapshot keeps when such a tier frees them cost it at most a
third more than it freed. Tiers with sizes on one volume take its room in turn,
down to the floor, with no margin above it: while a local snapshot keeps the
blocks such a tier frees, each copy it writes in place of another takes a slot
of free space, and the copies a measure then has it give up free nothing, so
free space can fall below the floor, as far as a full volume, before the tier
has given up every copy it holds or the snapshot goes. A limit below what the
tier holds is met at the next control pass by giving copies up in the order a
full quota replaces them, never a copy in transfer or the only copy of a state
in use, and a limit below one slot makes no room at all. Off a local volume, or
one that cannot be measured, the quota is the limit. A persistent tier's limit
bounds its whole root, and the tier may use all of it: the other caches nobody
holds, looked at once at start, count among what it holds, and while the root
holds more than the limit, as the tier grows or the volume shrinks, they go,
oldest first, before any copy of the tier's. A write that finds the volume
full (`ENOSPC`, `EDQUOT`) only reports it, and the engine measures the volume
at its next poll of transfers, once the writes that landed or failed are
collected. A volume that shows room above the floor, or one the tier does not
follow, met a limit the measure cannot see, or filled and freed again before
the measure. The first such write changes nothing; from the second on, the
limit is capped at what the tier holds then, its live slots, for the rest of
the process, and each such write lowers the cap again. The files go on taking
writes, so the tier keeps serving and replaces copies within what it has; only
a write that fails for another reason stops a file.

The write budget, set only by `--max-cache-disk-writes`
(`RuntimeResourcesConfig::cacheDiskWritesPerDay`), is a token bucket in the
disk budget, counted at submission in whole slots by every write of either
file. Without it the bucket has no rate and admits every write, so nothing
below is ever held back. It is the process's: each start begins with a
quarter of a day's. It decides which states are written, and nothing else: one gate on the engine thread, the state file writable, its
limit holding a slot and the budget admitting the state's class (a state in
use passes), is read only by the state's admission probes
(`CompositeState::canOffload`, `Model::canSnapshotToDisk`), never by a
reservation, an acquire or a write, so a write the engine admitted is never
refused for it. KV is demoted only for a chain a state on the SSD needs, and
copied only for a persistent point whose state was written, so those writes
pass the budget: a state the budget let through keeps its chain, and what a
chain costs comes out of later states' share. The budget therefore paces the
writes rather than bounding them: writes of states in use and of KV pass it,
and every start begins with a quarter of a day's.
While the budget is spent an evicted state is dropped, and `Cache::persist`
reports the tier busy before it writes anything of a point; once the budget
admits the point's state, the state is written first and the pages of its
chain after it. It refuses a point for a file that stopped taking writes, or
one larger than the tier's limit, its quota or what the volume leaves it. The
budget refills on the awake clock the native loop schedules its wake-ups on,
and write-behind wakes when the budget has room again.

Writes happen when RAM reclamation selects a victim. States copy through one
staging buffer, freeing their RAM immediately. KV leaves needed by a state
on them or below them are written straight from their extents and released
after the write succeeds; only then is the block above such a leaf a leaf. A
chain whose newest blocks have no copy, as a finished request's prompt past the
chain it restored, so leaves RAM one write at a time, and an allocation refused
meanwhile waits for each write. Unneeded
tails are dropped without writing, before any state, together with any disk
copies below them. Writes run on their file's IO worker. A KV page's slot
holds what the page holds as it is written: a Slim page's is the first part of a
Full page's payload, its regions in their order, and the block keeps the form of
its copy (`KvCache::copyForm`), which a restore of the block allocates its page
in. A KV page is restored by the engine's SSD readers (`SsdReaders`), which read
what the page's form holds, its index alone, a Slim page's payload or the whole
page, with one aligned read into memory of their own, and scatter it into the
page; the restores asked for between two submissions (`KvTier::submitRestores`)
go out together, index restores on every reader and whole ones on at most
`KvPageTier::kWholeRestoreReaders`. A state is restored on its file's worker.
Either way the transfer runs beside whatever command the model runs: a cached
page is never written by a command, and no command uses a page before its read
has landed. While a step waits for SSD
reads (streamed experts, PLE rows or spilled KV), each 1 MiB of a write waits
for them, at most 2 ms, unless a read queued behind the write on its file's
worker would wait as long (`SlotFile::Options::deferWrites`). At most 64 KV
writes are in flight (`KvPageTier::kWrites`): a persistent point's copies a
quarter of them and spilling requests' write-through half, so that neither
starves the demotions a reclaim needs, which may take the rest. When the tier
takes no more, leaves that need a write stay until a transfer lands, while
leaves whose page frees without one still go.

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
that no other state needs go first in turn. A state in RAM restores through the
K/V RAM holds, not its copies: a copy of K/V that RAM holds too goes once no
state on disk restores through it. The only copy of a state in use makes room
only for a copy in use, and last. Either way an ordinary state that
finds no other room is dropped. Before reclaim takes a state in RAM, only a
spilling request's page takes the spilled KV under it, the state first,
once nothing else is left. A freed slot returns its quota at once and its
blocks to the volume (`F_PUNCHHOLE`) once its file's IO worker finishes the
transfer it is in, so the two files together occupy about the live slots instead
of each keeping its high-water mark; `/status` reports that as `disk.file_bytes`
beside `disk.used_bytes`. Until the punch the other file can take that quota, so
allocated blocks can briefly exceed the limit by the slots freed but not yet
punched, bounded by the transfer each file's worker is in: about 2% of a 10G
tier in a 25-minute churn run on an M5 Max. A write that finds the volume full
anyway only reports it: its file goes on, and the engine measures the volume at
once and shrinks to the new limit. If the volume still shows room above the
floor, the write met a limit the measure cannot see, such as a volume quota or
a disk image whose host is full, or the volume filled and freed again; after a
second such write the tier keeps within what it holds, with one log line, while
serving continues. On a volume that cannot punch holes the files keep the
blocks of freed slots, which the runtime reports once; the limit counts them.

Transfers use `pread`/`pwrite` with `F_NOCACHE`, every one an aligned range.
Those of the files' IO workers, every write and a state's reads, move whole
1 MiB chunks of 16 KiB-aligned memory that start at an aligned offset of the
slot (a state's staging buffer and lane buffers) directly, everything else
through the file's own 1 MiB buffer; a KV read on the SSD readers is one read
of whole 4 KiB pages around the bytes it needs, into the readers' own memory
(`SlotFile::readAt`). A write that fails for another reason
than a full volume disables further writes to that file, with one log line.
Failed KV writes retain RAM pages; failed state writes invalidate the disk
copy. A failed read invalidates its cached data, allowing lookup to fall back
to the surviving prefix.

With `--persistent-cache`, each slot's record carries a CRC-32C of its payload,
or, in a KV file whose pages hold block records ([KV on the SSD](server.md#kv-on-the-ssd)),
one of each segment: the page's index, then each block's record. A state that
fails it on its first read is a miss. Every read of a KV copy a restart took
back checks the segments it covers: a whole or index restore that fails is a
miss, and a block or history read that fails is `kv_unreadable` for the
requests that read it. A KV record of a layout with a sliding window also names
the form of the page its slot holds (`KvSlotLabel`), and a start takes back
only the forms it restores.

`/status` reports the shared quota, its limit, the write budget and KV
transfers under `disk`. Its cumulative `read_bytes` and `written_bytes` count
bytes transferred by file IO across KV and state files, including partial or
cancelled transfers. They exclude
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

## Native protocol

The engine receives `ignore_eos` as bit 0 of the request frame's flags word and
`return_progress` as bit 1; the word rejects undefined bits.

A request that generates in a thinking block may carry a thinking budget: its
tokens, the end tokens the engine selects where it runs out, the last of which
closes thinking, and a second token that also ends it, a call's opening where
the output may call a tool. The engine counts the thinking tokens it selects,
and then the end tokens. In the selection where the budget runs out, one small
dispatch gives that row and those after it the end's tokens, unless a draft
token before the row ended thinking, and gives a sampled lane's drafted row the
target probability 1 for a drafted end token and 0 otherwise, so acceptance
keeps the draft only where it drafted the end. Once no thinking token is left,
the end starts at a selection's row 0, before which no draft token can end
thinking, and a second small dispatch replaces the lane's draft proposals with
the end tokens left before the verify, so a decode cycle accepts as many of
them as it verifies; the request's speculation counts leave those proposals
out. A constrained request's end waits for its grammar: it starts at the first
row from the cut whose mask admits its first token, the rows before it keeping
their own selections, which the engine counts as thinking past the budget, and
it replaces no proposals before it has started, as only the verify reads the
masks. A selection where no budget runs out dispatches nothing more.

Requests carry the score-token IDs and Done events the selected f32 logits; the
server and the engine must speak the same native wire version. Scoring requires
2–255 distinct, in-vocabulary tokens, no images or generation constraints, and a
zero output budget. It may use the full context window because no generated
token needs a reserved position. The final prefill chunk runs the target head
but no sampling policy or DFlash decode. Successful scoring emits no Tokens
event, finishes with Stop, and reports zero decode time. Cancelled requests
carry no logits.

A request's image spans, at most 1024, images and PDF pages at most 64 of
them, each name one frame, an image, or two, a video's temporal patch: its
pixels hold the patch's two frames one after the other, which `vision_patchify`
reads into the patch's two temporal slots, where an image fills both with its
one frame. A span's grid, M-RoPE positions and encode are an image's; its
digest covers both frames. Version 12 of the wire added the frame count.

An engine sends its Ready event first. One whose bootstrap fails, as when
`--max-context` exceeds what its budget holds, writes its report to stderr,
sends the report's reason as an engine failure in its place and exits; the
server ends the start with that reason (`Error · …`, or
`Engine restart failed · …` for a relaunch). An engine that ends before either,
as a crashed one or one refusing its arguments does, ends it with
`native protocol reached EOF`.

## Failures and crash traces

A row that is not finite is a per-request failure, not an engine fault. Every
prefill chunk checks the last row of each lane's final hidden, which a value
that is not finite anywhere in the chunk, or in the state it computed from,
reaches through the recurrent state, and through attention on a GPU whose tile
weighs non-finite keys and values (some families' tiles give them zero weight,
for every reader); a token selection finds one as
a logit row with no selectable token, its admitted logits all NaN or -inf
(sampling penalties leave those as they are), and a score request as a score
logit. The engine reports `model_result_invalid` for that request alone, a
selection's message counting the row's NaN, +inf and -inf logits, before it
publishes the failing step's cache state or any output, and the rest of the
batch finishes normally. Since every step that
publishes was checked, what earlier steps published stays, as it does for a
cancelled request. A damaged state is one way to such a result: a request
whose first step from the state it restored is not finite takes that state out
of the cache, and the next request with the prompt computes it again. Values
that are not finite cannot reach the next holder of a request's memory either:
a page a request takes comes with zeros in the rows its work does not write,
and the lane of a request that found one returns its draft rings cleared, since
attention weighs rows past a request's end by zero and a BF16 value is read as
stored (0 × NaN is NaN). GPU faults and broken engine invariants stay fatal and
still mark the runtime unhealthy.

Request logs omit bodies; full crash traces require explicit
`SPLASH_CRASH_TRACE=1` and can contain private conversation data. With it, a
failed engine's last frames, at most 512 and 16 MiB, are written to
`~/Library/Logs/Splash/crash/`, which keeps the newest four traces within
64 MiB. A frame over 16 MiB, such as a request with large images, is kept only
as a marker with its size and SHA-256 (`omitted_frames` counts them), and a
trace missing engine input that way cannot be replayed.
`python -m server.crash_trace <trace>`, run from the checkout root, replays a
trace.
