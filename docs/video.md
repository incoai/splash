# Video

`splash serve --model MiniMaxAI/MiniMax-H3` serves MiniMax-H3, a text to
video and audio model, as its own instance: its server answers
[OpenAI's video API](https://platform.openai.com/docs/api-reference/videos)
and no language model's routes, which answer 404; chat and video together
are two instances on two ports. The two share the model layer, the
installer's inspection, publication and assemblies, and the server's HTTP
handling; the video model has an engine process of its own
([figure](video-serving.svg)). It runs on an M5 (Apple10
GPU) with 48 GB.

## Video API

| Route | Answer |
| --- | --- |
| `POST /v1/videos` | a queued video object |
| `GET /v1/videos/{id}` | the video object |
| `GET /v1/videos/{id}/content` | the mp4; `?variant=preview` the TAEH3 preview once decoded |
| `GET /v1/videos` | a list (`limit` up to 100, `order`, `after`) |
| `DELETE /v1/videos/{id}` | forgets the job, cancelling it at its next step |

`POST` takes JSON, or the multipart form OpenAI's SDK sends: `prompt`,
`model`, `seconds` (a whole number from `4` to `15`, `5` the default:
Pika's `--duration` choices and `4`; the smallest 17 n + 5 frames at least
seconds x 24 at 24 fps, Pika's `align_num_frames`, such as 107, 124, 175, 243
and 362 frames for 4, 5, 7, 10 and 15 s; the golden-clip tests' server also
takes `3`, 73 frames, under `SPLASH_GOLDEN_CLIPS=1`) and `size` (`1344x768`, the
default, or another canvas of
[the 768p and 480p tiers](#video-canvases), such as `832x480`), and
Splash's `variant` (the installed one: `vdn-8`, `vdn-50`, `fasth3-8`, `fasth3-4`, `pika-6` or `pika-ref2va-6`), `seed`, `steps` (0 or the variant's: its adaLN tables are those of
its schedule), `preview` (false) and `audio` (true). OpenAI's
`input_reference`, a PNG or JPEG file field, is the clip's first frame, and
Splash's `last_frame`, another (with `input_reference`), its last; the server
puts each on the request's canvas as the model's MLX port does
(`images.keyframe`: an image of that size as it is, otherwise the first frame
stretched and the last scaled to cover and centre-cropped, LANCZOS; stored
RGB, alpha dropped, no EXIF rotation). The video object has OpenAI's fields,
and `phase`, `variant`, `seed`, `preview` (ready) and `phases`, each completed
phase's seconds and the engine's peak footprint. Errors have the OpenAI
shape: `video_queue_full` (503) past four queued jobs, `video_not_ready`
(409) for content not yet decoded, `not_found` and `model_not_found` (404).

A Ref2VA job (Pika H3 Flex's reference-conditioned video and audio) takes
`references` in place of keyframes: a file field of the multipart body,
repeated, in the order the prompt numbers them (`<Picture i>`, `<Video k>`,
`<Audio j>`, each kind counted on its own). Its content type gives its kind:
`image/png` or `image/jpeg` an image, `video/mp4`, `video/quicktime` or
`video/webm` a video with its soundtrack, `audio/mpeg`, `audio/wav`,
`audio/mp4` (M4A) or `audio/aac` an audio clip (a voice or music reference);
any other type is a 400. JSON bodies take no references (the keyframes too
are multipart files only). The server decodes and normalizes them as Pika's
`pika_h3.core` does (`server/references.py`): PyAV (`av`, the library Pika
uses, pinned to its version) decodes a video's rgb24 frames for at most 16 s
at its average rate and resamples a soundtrack or an audio clip to 32 kHz
stereo; PIL decodes an image (RGB, no EXIF rotation). Reference video and
audio need PyAV, which Splash does not ship; install it with
`<python> -m pip install av==19.0.2`, `<python>` the interpreter Splash runs
(without it such a request is a 400 naming that command). A video goes onto the
24 fps grid (each frame held until the next one's slot), is truncated to the
clip's frame count and LANCZOS-resized to the canvas its own aspect ratio
resolves to on the clip's tier (480p: short edge 480 within 832 x 480 pixels;
768p: 768 within 1344 x 768); an image is LANCZOS-resized to the tier's
reference short edge (1024 at 480p, 2048 at 768p, upscaled, no area cap,
both sides rounded to 32); a soundtrack or an audio clip is truncated to
round(frames / 24 x 32000) samples. Pika's limits hold: at most 9 images, 3
videos and 3 audio clips, 12 in all; each video and each audio clip 2 to 15 s
of decoded source, and the videos 15 s at most in all, the audio clips
likewise; a video at least 22 frames once normalized; an image or a video
among them; aspect ratios within 1:4 and 4:1. References are refused with
`input_reference` or `last_frame`, and condition the `pika-ref2va-6` variant
alone, which a request with references takes when it names none (another
variant is a 400 naming it, and `pika-ref2va-6` refuses a request without
references). The presentation (`ref2va_presentation`) holds at most 2,048
text rows (Pika's `MAX_TEXT_TOKENS`) and, with its vision rows, at most the
bound the engine's Ready announces (32,768); a request without references
keeps its whole presentation within 2,048. An audio clip's samples and an
audio reference's labels are those of Pika's `decode_audio_stereo` and
diffusers' `_build_presentation`.

Jobs run one at a time in their order of arrival; the mp4s and previews of
finished jobs stay in a 4 GB store, the oldest evicted first.

`splash generate --model MiniMaxAI/MiniMax-H3 --prompt ... --out clip.mp4`
installs the model as `serve` does and runs one clip on its engine without a
server (`python -m server.videos`): its flags are the `POST` fields, checked
by the same `parse_request`, `--first-frame` and `--last-frame` the image
files of `input_reference` and `last_frame`, `--reference FILE`, repeated in
presentation order, the files of `references` (each one's kind from its
leading bytes, else its extension); progress goes to stderr, each line with
the engine's peak footprint,
`--preview` writes `clip.preview.mp4` once decoded, and Ctrl+C cancels at the
next step.

## Video canvases

A canvas belongs to a tier by its short edge: 768p or 480p. Within a tier
any width and height that are multiples of 32 (one DiT token is 2 × 2
latent pixels of 16 × 16 px), with the long edge at most twice the short,
render; the server and the engine refuse others (400; the engine's
`ops::canvasTier`). These are the named canvases:

| `size` | Tier | Token grid | Rows per latent frame | VSA tiles, 5 s (partial) | Video VAE tiles |
| --- | --- | --- | ---: | ---: | ---: |
| `1344x768` (16:9, default) | 768p | 42 × 24 | 1,008 | 660 (120) | 4 × 7 |
| `768x1344` (9:16) | 768p | 24 × 42 | 1,008 | 660 (120) | 7 × 4 |
| `832x480` (FastVideo's) | 480p | 26 × 15 | 390 | 280 (118) | 3 × 4 |
| `864x480` (mlx-serve's) | 480p | 27 × 15 | 405 | 280 (118) | 3 × 5 |
| `480x832` | 480p | 15 × 26 | 390 | 280 (118) | 4 × 3 |

Everything a request derives from its canvas comes from the token grid,
per request: the packed rows (text, keyframes of one latent frame's rows
each, audio, latent frames × grid), the (t, h, w) positions and RoPE, the
window softmax's key-range table (its window of 5-frame chunks, radius 1,
is temporal), FastH3's tile map (4 × 4 × 4 cubes, partial on any axis: a
480p grid's 15 rows end in cubes of 3), the activation plan, the video
VAE's 256 px tiles on both axes (encoder and decoder), the TAEH3 preview
and the mp4's bit rate (0.8 bits per pixel and frame). No weight or adaLN
table depends on the canvas: the tables depend on the variant's timesteps
alone, so one conversion serves every canvas. The device's execution
constants (`runtime/video/metal/DeviceProfile.hpp`) were tuned at 768p and run 480p
unchanged; those a 480p tuning pass should revisit are marked there. A
480p clip's DiT has 0.39-0.40 of the rows of the 768p clip of its length:
5 s at 832×480 plans 1.7 GB of activations (VDN-H3) against 3.8 GB at
1344×768.

## Video installation

The model's assembly is of target format `h3-video` (`install/video.py`).
The family's constants, which the installer and the server read, are one
row, `H3`, in `install/families.py`.
Splash installs MiniMax-H3 from MiniMax's release, `MiniMaxAI/MiniMax-H3`,
one variant per installation (`H3.variants`): `:vdn-8`, OpenVDN's 8-step
VDN-H3 stage, and `:vdn-50`, its
50-step stage, are models of their own, each with its own selection link,
entry and assembly, and so are FastVideo's FastH3 students, `:fasth3-8` and
`:fasth3-4`, and Pika H3 Flex's `:pika-6` and `:pika-ref2va-6`. The model ID
without a variant installs `H3.default_variant`, `:pika-6`, unless its
selection link holds an installation, which keeps its variant. A server
serves the variant it installed.
A request without `variant` is served `H3.default_variant` by an assembly
of several that holds it, else the first.
Installation and `splash generate` print the license's name, its excluded
territories and its URL; the assembly links the release's `LICENSE`.

The installer converts the variant into the engine's images (the int8
format `h3-w8a8-vdn-v1`, [Kernels](internals.md#kernels)) as it reads its sources at
their default branches' commits: the release's FL2VA transformer, VAEs and
tokenizer, and `OpenVDN/vdn-minimax-h3` (the stage's linear branch and LoRA
adapters); TAEH3, the preview decoder, is read from
`github.com/madebyollin/taehv` at a pinned commit and checked by its SHA-256.
`install/conversion.py` runs a recipe's units, `install/h3.py` is
MiniMax-H3's: each unit names the tensors it reads, which the driver reads by
HTTP range requests of the safetensors files (`hub.Repository.open`, the
header first, then each run of adjacent tensors in chunks of 64 MB, eight at
a time), so no source file lands on disk, while the unit before it converts.
The adaLN tables of the variant's schedule are tabulated by the engine
(`splash video-adaln`, on the GPU) from the blocks' adaLN bases, read into
scratch first. About 83 GB are read; the disk holds at most about 29 GB for
`vdn-8`.

FastH3's variants have their own recipe, `install/fasth3.py` (int8 format
`h3-w8a8-fasth3-v1`). `:fasth3-8` reads the student
`FastVideo/FastVideo-FastH3-8-Step-V2` (bf16 shards, 69 GB) and its contract
(`fastvideo_inference.json` and the scheduler configs). `:fasth3-4` reads the
release's transformer shards with the LoRA bundle
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` (`vsa-datafree`, 5.3 GB),
refusing a release whose shards are not the bundle's base's, and, at a pinned
commit, the contract of the student the bundle was taken from. A block holds
its int8 linears with the VSA compression gate and its adaLN table of the
trained schedule; its keyframe-timestep row goes to scratch for
`splash video-adaln`. Every file but the manifest is byte for byte a
reference conversion's of the same sources.

The images are a derived entry, `.converted/<key>`, keyed by the SHA-256 of
the content IDs the Hub reports for every file the recipe reads, the recipe
and the DiT, as a GGUF's metadata is a `.metadata` entry. A conversion writes
into `.converted/.loading-<key>`, holding its `.lock`: each unit's files are
fsynced and published by rename, so an interrupted conversion resumes at the
first unit not written, and a second installation of the key waits for the
first. Once every unit is written, its `files.json` lists the files and
their sizes, and the publication renames the staging into the entry under
the installation lock, as it publishes the assembly that links it. Garbage
collection removes an entry no assembly links, and a conversion's staging
that no process holds and that has not changed for a week. The record's
sources are the release and the text encoder; `video` adds the variants, the
entry's key, the commits of the companion repository and of the files the
recipe pins, and the text encoder's package (`encoder`: its entry's key and
the shards it links, trims and leaves out). A start
finds the entry of the files it lists and converts nothing, so a commit that
changes none of them installs at once; without the Hub, it starts from the
entry the installation records.

| Path | Holds |
| --- | --- |
| `weights/dit-<dit>/` | the DiT (`turbo` for `vdn-8`, `default` for `vdn-50`, the variant's name for FastH3): 50 int8 blocks, 3 QKFIX blocks, globals, patch projections, prep inputs, adaLN tables and `manifest.json`; 24.8 or 26.1 GB |
| `weights/vae/` | the video VAE's int8 decoder and its encoder of a clip's keyframes, 2.8 GB |
| `weights/audio/`, `weights/preview/` | the audio VAE's decoder and TAEH3's, 0.3 GB |
| `tokenizer/`, `LICENSE` | FL2VA's tokenizer, which the server encodes prompts with, and the license, from the release's snapshot |
| `encoder/` | the text encoder's package of `mlx-community/Qwen3-VL-32B-Instruct-8bit` at its default branch's commit, as a draft is: only what the engine reads, 27.92 GB of the repository's 36.00 GB (below) |

The text encoder's package holds what `model/TextEncoder` and
`model/VisionTowerWeights` read of the checkpoint, 1,604 tensors: decoder
layers 0-49 (1,250), the token table (`embed_tokens` codes, scales and
biases) and the vision tower (351 bf16 tensors), with `config.json`; layers
50-63, the final norm and `lm_head` (354 tensors) are not installed. The
installer reads each shard's header (range requests, or the Hub cache) and
classifies it by its tensors' names, at any width: a shard of only read
tensors is downloaded and linked as it is, a shard of none is never
downloaded, and a shard of both is rewritten with only the read tensors
(names, dtypes, shapes and bytes as they are, in the shard's order, under its
`__metadata__`, by range reads) into a derived entry of its own,
`.converted/<package key>`, under the same file name, with a
`model.safetensors.index.json` of the package. The key is the SHA-256 of the
content IDs of the shards the package holds, the trimming's name
(`qwen3-vl-encoder-package-v1`) and the layer count, so a start whose shards
did not change writes nothing, and one without the Hub uses the package its
installation records. The engine scans the shards' headers
(`SafetensorsCheckpoint`) and reads no index; the repositories' own index
names a bf16 checkpoint's 14 shards, none of which they hold, and is not
linked. The layer 49 / 50 boundary falls inside a shard at every width, and
the final norm and `lm_head` share the last shard with layers past 49 only:

| Width | Repository | Package | Linked shards | Trimmed shard (tensors kept / held) | Left out |
| --- | ---: | ---: | --- | --- | --- |
| 4-bit | 19.62 GB | 15.34 GB | 1-2, 10.68 GB | 3: 4.66 GB (425 / 494) | 4 |
| 6-bit | 27.81 GB | 21.63 GB | 1-4, 21.31 GB | 5: 0.32 GB (9 / 331) | 6 |
| 8-bit | 36.00 GB | 27.92 GB | 1-5, 26.75 GB | 6: 1.18 GB (53 / 256) | 7 |

A Hub cache that already holds a repository's every shard keeps them (the
installation pins the commit's snapshot); the package then adds its trimmed
shard beside them, and the saving is the left-out and trimmed shards' once
the cache holds only what the package links.

`prepare_local_model` takes a directory of images converted before
(`dit-<dit>/` of one variant or more, `vae/`, `audio/`, `preview/`, and
`tokenizer/` with `LICENSE`), and links its files as they are; the encoder's
package comes from the Hugging Face cache: the one an assembly of the models
root records for that commit, or one planned from the cached shards. A local
copy of the release is refused.

## Video engine

The engine process of an h3-video assembly (`runtime/video/VideoRuntime.hpp`)
speaks the native protocol and announces `FeatureVideo` and, as its prompt
bound, `TextEncoder::kMaximumPresentationTokens` (32,768); a start that fails
sends its reason in place of Ready, as a language model's engine does. Its
Ready leaves a language model's fields (the vocabulary size, the stop tokens
and the mask rows) zero or empty, which the engine's and the server's
validation of Ready accept only with `FeatureVideo` set. A
`VideoRequest` (the clip's DiT, frames, size, steps, seed, outputs, the
prompt's token ids and, for a clip from its first and last frames, their
pixels on the canvas, for a Ref2VA clip its references' normalized pixels
and samples) runs every phase in this process, one request at a
time, on Splash's Metal backend and weight readers. The `VideoRuntime`
composes the phases as SGLang Diffusion composes a pipeline, one stage each
(`runtime/video/stages`): the text encoder, Qwen3-VL-32B through decoder
layer 50 (`TextEncodingStage`); request preparation, with the video VAE's
encoder for the keyframes (`PrepStage`); the DiT (`DenoisingStage`); the
TAEH3 preview (`PreviewStage`); the video VAE (`DecodingStage`); the audio VAE
(`AudioDecodingStage`); and the mp4 (`MuxStage`), each a `PipelineStage`
(`pipeline/`). Under the stages, `model/` (weights, request preparation, the DiT's activation plan and weight
stream), `ops/` (the operators) and `metal/` (the dispatch recorder and the
device's constants) keep Splash's layering, each part on its own lower parts and on the layers
of `runtime/` its namesake may use, checked by
`dev/tools/check_architecture.py`; the kernels are under
`runtime/metal/kernels/video`. The kernels and the host code were ported
from an earlier standalone engine by the same author. The DiT's
activations share slabs by lifetime within a block (`model/MemoryPlan`), and
its int8 block weights stream from their files through a ring of 8 slots,
each filled by a background batch of Splash's SSD readers ahead of the GPU
(`model/WeightStream`). `VideoProgress` reports each phase's start and end
and each denoising step, `VideoOutput` delivers the preview and then the
clip; a `Cancel` stops the request at the next step or phase. A language
model's engine refuses a `VideoRequest`.

The text encoder (`model/TextEncoder`) streams its weights a layer at a
time: each decoder layer's image is written from the checkpoint's shards
(`pread`, then the GPU repack) just before that layer's command and released
once the command has run, and the token table is released after the gather.
The encode holds one layer (0.27 GB at 4 bits, 0.40 GB at 6), the
activations and, during layer 0, the table, instead of all 50 layers (13.7
GB at 4 bits, 19.8 GB at 6); the commands and their order are the resident
encode's, so the hidden states are the same bits and the time is the same
sum of loads and layers. The shards are read, not mapped, so the page cache
they leave is not the engine's footprint.

A presentation with keyframes also runs the checkpoint's vision tower
(Qwen3-VL-32B's 27-block ViT, `ops/VisionTower`, `model/VisionTowerWeights`,
`kernels/video/vision_tower.metal`) on each keyframe before the text layers:
patchify (the pixels rounded to bf16, as the reference casts them), the
patch embedding and the bilinear resample of the 48 x 48 position table, 27
blocks of 2D-RoPE attention over the image and GELU MLP, the merger and the
three deepstack mergers after blocks 8, 16 and 24. Its weights are the
checkpoint's own, unquantized bf16 in mlx-community's 4- and 6-bit
repositories alike (the same bytes, 1.19 GB), read a stage at a time and
converted to fp16 (183,501 of 595 M values change, all below fp16's normal
range); weights and activations are fp16, accumulation, the residual stream
and the LayerNorm statistics fp32, a port of the language models' bf16 tower
kernels (`kernels/shared/vision.metal`). The merger's rows replace the image
pads' embeddings, the deepstack mergers' rows are added to them after
decoder layers 0, 1 and 2, and the rotation is Qwen3-VL's interleaved M-RoPE
of the presentation (`text_encoder_qk_norm_mrope`: an image's pads at (s,
s + row, s + column) of its merged grid); a text-only prompt runs the
text-only commands.

Ref2VA's presentation (diffusers `MiniMaxH3Ref2VATextEncoderStep`, built by
`server/videos.ref2va_presentation`) adds reference videos: per video
`<Video k>: ` (after `<Audio j>: ` when it carries sound) and, per pair of
its 2 fps samples, `<t seconds>` and a block of `<|video_pad|>`, one per
merged token of a frame. `TextEncoder::videoBlockFrames` (and the server's
`video_blocks`) pick the pairs as `_sample_video_condition_frames` does: of a
reference normalized to 24 fps, every 12th frame, the last repeated to an
even count; 124 frames give (0, 12), (24, 36), ..., (120, 120) at 0.25, 1.25,
..., 5.0 s, rendered `<0.2 seconds>` ... `<5.0 seconds>` (Python's `{:.1f}`).
A video block is an `Image` whose `rgb2` is the pair's second frame: the
tower patchifies the two frames over the temporal patch
(`h3_vision_patchify_pair`) and otherwise runs it as an image, which is what
Qwen3-VL does for a video (attention within each frame pair, the position
embedding and 2D RoPE repeated per pair); its M-RoPE positions are an
image block's (`get_rope_index` splits a video grid per merged frame, the
timestamps between are text). Past 2,048 rows (one prefill pass) the
linears run a pass of 2,048 rows at a time, the norms, rotation and
attention over every row (`kMaximumPresentationTokens`, 32,768); a prompt
within one pass runs the one-pass commands.

A first frame, and a last frame, condition the clip as the model's FL2VA
checkpoint does: the video VAE's encoder turns each into 1,008 latent rows,
noised to t = 0.999, which join the DiT's sequence after the text, stay fixed
through the denoising and take a fourth adaLN slot per block. The text
encoder sees MiniMax-H3's presentation of the request, as the release's
FL2VA pipeline and Pika H3 Flex build it (`server/videos.presentation`): per
keyframe, the first frame's and then the last's, the label `<Picture i>: `
and a vision block (`<|vision_start|>`, one `<|image_pad|>` per 2 x 2 merged
16-pixel patch of the keyframe on the canvas, 390 at 832x480 and 1,008 at
1344x768, `<|vision_end|>`), then the prompt, with no chat template and no
special tokens; the block's rows are tagged as video (tag 0) for the DiT.
`SPLASH_VIDEO_PROMPT_ONLY=1` keeps the earlier presentation, the prompt
alone, which the reference first-frame clip was made with
(`test-video-first-frame-real` sets it). A presentation is at most 2,048
tokens, so a 1344x768 clip from a first and a last frame (2 x 1,010 vision
rows and two labels) leaves 16 tokens for the prompt.

The FastH3 variants (`runtime/video/model/FastH3.hpp`) run the same stages.
Their DiT replaces VDN-H3's window softmax and linear branch by FastVideo's
Video Sparse Attention on 64-token tiles (`ops/VsaTable`, `ops/VsaAttention`,
`kernels/video/dit/vsa.metal`): the video rows in 4 x 4 x 4 cubes of latent
tokens, the text, keyframe and audio rows in 64-row chunks. A video tile
attends to every chunk and to its top k = ceil((1 - s) V) video tiles by the
scores of the tiles' mean Q and K (s 0.8 for `fasth3-8`, 0.9 for
`fasth3-4`), plus the tiles' compressed output gated by `to_gate_compress`;
the other rows attend densely. Inside the blocks the rows are in tile order.
The schedule is the student's trained DMD rungs (999 to 125 in eight, shifts
10 / 3; 999, 749, 500, 250, shifts 12 / 3), one forward each. The students
were distilled for text-to-video; a first and last frame condition them as
FastVideo's pipeline conditions any MiniMax-H3 DiT, which they were not
trained for: the keyframe rows, fixed at timestep 0.999, are a dense prefix
segment of the tile map.

A FastH3-layout DiT whose manifest says `"vsa": {"attention": "dense"}`
(sparsity 0, no compression gate: `pika-6`'s base, written so by
`install/fasth3.py` for a spec) runs dense attention on the same kernels
(`ops::VsaSpec::dense`): the QKV GEMM without the gate rows (the QKFIX
blocks skip the int8 GEMM and its producer altogether), no tile means,
routing or compressed output, and every 64-row chunk of the tile order
attends to all rows with the int8 QK^T / fp16 PV fine stage (VSA's flags
bit 1), QKFIX's fp16 QK and split term unchanged. The plan drops the gate,
coarse-stage and selection tensors (0.5 GB at 1344x768 5 s).

## Runtime adapters

A DiT can run a LoRA adapter at run time instead of having it merged at
conversion (`runtime/video/model/Adapter.hpp`): for every adapted linear
y = W x + scale B (A x). The down-projection T = scale A x is one MPP
`matmul2d` of the GEMM's own int8 input rows (A rotated by the same Hadamard
transform as the int8 weights, so the int8 rows and their scales are its
input) into fp16 T (`kernels/video/dit/lora.metal`); the up-projection
T B^T runs inside the GEMM's `_lora` kernel (`dit_qkv_lora_post`,
`dit_gated_residual_lora_post`, `dit_fc1_swiglu_lora_post`,
`dit_qkfix_qkv_lora`) and is added to the dequantized output before the epilogue: QK-norm and RoPE,
SwiGLU, the gated residual. q, k and v are three segments of one rank-128
T, the VSA compression gate has no LoRA, and the QKFIX blocks' fp16 q / k / v
GEMM reads an unrotated copy of A. The token refiner adds its adapter in the
prep as PEFT does (bf16 linears, then the bf16 add). The adapter's A and B
stay resident in bf16 (Pika H3 Flex FL2VA: 1.28 GB). Without an adapter the
plain kernels run, unchanged: the golden tests' latents stay byte-identical.

An adapter is converted once, at installation (`install/fasth3.py`
`lora_units`, format `h3-lora-v1`, into the DiT's `adapter/`), from a spec's
PEFT safetensors, zero-padded to rank 128.
The DiT's `manifest.json` names its adapters, `"adapters": [{"path": P,
"scale": S}]` (one at a time; entries that are not objects, VDN-H3's names of
the adapters merged at conversion, are none), and `video-golden` takes
`--adapter PATH[:SCALE]` or `--no-adapter` in their place.

`pika-6` is MiniMax-H3's base DiT with Pika H3 Flex's FL2VA adapter
(`Pika-Labs/Pika-H3-Flex`): the base converted in FastH3's block layout
with a zero compression gate at VSA sparsity 0, i.e. dense attention
(`install/fasth3.py`, its spec in `install/families.py`), and Pika's released sampler: six forwards
at the knots 1000, 750, 500, 250, 100, 40 (shifts 12 / 3, sigmas in float64
then fp32), one forward each (CFG 1), x0 = x + sigma v and, but after the
last forward, x' = (1 - sigma') x0 + sigma' eps with fresh noise per step,
video then audio, from the request's second stream (seed + 1, MLX draws:
Pika's are CUDA Philox, so only the first forward matches a reference
bit for bit in its inputs; `video-golden --renoise NPZ` takes another
sampler's noise). Its contract lives in the DiT's manifest
(`schedule.sampler: renoise`, `sigma_precision: float64`). The base keeps
the release's fp32 modules in fp32 (time embedder, patch projections, output
heads; `LoraStudent(keep_f32=True)`), as Pika runs them: rounded to bf16, as a
FastVideo student holds them, they move half the adaLN table entries by a
bf16 ulp and the step-0 velocity by 4-8 points of relRMS against a reference
run of Pika's pipeline. Its attention is dense (`vsa.attention`, [Video engine](#video-engine)).
The installer converts it into `dit-pika-6/`, its adapter in
`dit-pika-6/adapter/` (the manifest's relative path), with the release's
`vae/`, `audio/`, `preview/`, `tokenizer/`, `LICENSE` and `NOTICE`. Its golden test is `dev/tests/video_real.py engine
--golden-final FINAL --golden-step0 STEP0` on the 3 s golden clip (the first
forward's velocity and the six-step latents byte for byte).

`splash install MiniMaxAI/MiniMax-H3:NAME --spec FILE` installs a checkpoint
of MiniMax-H3's DiT that Splash does not ship as the variant NAME, by the
same recipe, from a JSON spec of the form `pika-6`'s in
`install/families.py`: `checkpoint` (`repo`, `prefix`, and a `revision`
unless it is the adapter's repository; native or diffusers names), an
optional `adapter` (`file`, `name`, `alpha`, and `repo` where it is not the
checkpoint's), `schedule` (`rungs`, `video_shift`, `audio_shift`,
`sampler`, `sigma_precision`, optionally `renoise_seed_offset`),
`references` (false by default) and `qkfix` (45, 48, 49 by default). The
installer refuses a NAME a shipped variant has, a field it does not know,
and what the recipe refuses (`fasth3.check_spec`, the checkpoint's shapes,
the adapter's rank and targets, the QK gains), each in one line; the
installation records the spec, so the model ID installs again without the
file. The installer checks structure only and has no quality threshold: a
spec that passes every check can still give poor clips.

The `_lora_post` GEMMs compute the up-projection tile after the K loop, so
its accumulators are not live through it (fc1 + SwiGLU keeps the gate
half's result in its int32 accumulator's registers, bit-cast, while it
computes the value half's).

## Reference rows

The packed sequence is a list of row classes (`runtime/video/ops/RowLayout.hpp`):
one table, `kRowClasses`, gives each class its input stream, modality tag,
adaLN slot, timestep, update rule and attention role, and
`RowLayout::build` its (t, h, w) positions, float64 as the pipelines compute
them. Text-to-video is the text and target classes, FL2VA adds keyframe rows,
and Ref2VA (Pika H3 Flex's reference-conditioned task) the reference classes,
in the order `text | per reference [soundtrack, video] / [image] / [audio] |
target audio | target video`:

| Class | Rows | Timestep | Position (t, h, w) | Update | Attention |
| --- | --- | --- | --- | --- | --- |
| text (tag 1) | prompt and presentation labels | target video t | (i, 0, 0) | fixed | dense prefix |
| vision (tag 0) | Qwen3-VL vision rows in the text | target video t | (i, 0, 0) | fixed | dense prefix |
| keyframe (tag 0, slot 3) | one target frame | max(t, 0.999) | first / last target frame | augmented | dense prefix |
| ref_soundtrack (tag 2, slot 4) | 2 × its latents, channel-major | 1.0 | clock + k; w: its video's width grid ends | fixed | dense prefix |
| ref_video (tag 0, slot 3) | F × (h / 2)(w / 2) | max(t, 0.999) | clock + temporal grid; its own grid | augmented | dense prefix |
| ref_image (tag 0, slot 3) | (h / 2)(w / 2) | max(t, 0.999) | clock; its own grid | augmented | dense prefix |
| ref_audio (tag 2, slot 4) | 2 × its latents | 1.0 | clock + k; w: the target grid's ends | fixed | dense prefix |
| target_audio (tag 2) | 2 × round(frames / 24 × 40) | audio t | clock + k | noised | dense prefix |
| target_video (tag 0) | F × G_H × G_W | target video t | clock + temporal grid | noised | VSA video region |

The rotary clock starts at the text length; an image advances it by 1, an
audio reference by its latents, a video by max(soundtrack latents, its
temporal span 5/3 × (1, 4, 4, 4, 4) summed sequentially); keyframes do not.
Augmented rows start at 0.999 x + (1 - 0.999) eps (fp32, as the keyframes),
clean rows at their latents, targets at the request's noise. A forward's
timesteps are the sorted distinct values of its rows (`RowLayout::timesteps`;
Ref2VA's step 0: {0, 0.999, 1}). `ranges` and `fixedRanges` give the
attention metadata: the dense prefix, the video region and the rows the
sampler never updates.

The DiT takes the rows in its engine order (`RowLayout::engineOrder`):
`text | video conditions C | audio conditions Ca | target audio | target
video` (`ops::SequenceLayout`, `condAudio`), the identity for text-to-video
and FL2VA. Every DiT operation but the VSA tile map works per row or over
all rows, so the order changes no result; the tile map cuts each prefix
segment (`SequenceLayout::segments`: the text, each run of one conditioning
class, the audio conditions, the target audio) into 64-row chunks of its
own and tiles the target video alone. adaLN slot 3 is the video modulation
at max(t, 0.999), slot 4 the audio modulation at t = 1 (5 slots per block,
and 5 in the final layer, which also runs the audio condition rows and
drops their output). The embedding reads the condition rows ahead of the
target rows (`videoIn` [C + Nv][96], `audioIn` [Ca + Na][32]); the output
heads and the sampler see the target rows only.

`Prep::referenceRows` lays out a Ref2VA request on the CPU from precomputed
reference latents (`PrepInputs::references`: video VAE latents [24][F][h][w],
audio latents [T][64]; `referenceNoise` the augmentation noise) and fills
`Prepared::sequence`, `cond` and `condAudio`. The engine computes those latents
itself (`runtime/video/model/References.hpp` `encodeReferences`): a reference video through
`VaeEncoder::encodeVideoRgb` (trimmed down to 17k + 5 frames, independent
17-frame clips tiled as the keyframes, the last 3 latent frames dropped:
F = 5k + 2), an image through `encodeRgb`, both sampled as
`encode_vae_condition` does (`runtime/video/model/ReferenceLatents.hpp`: torch's CPU
`randn` of seed 42 in (1, 24, F, h, w) order, float16 round trip,
normalized); a soundtrack through `runtime/video/ops/AudioEncoder.hpp` (DAC encoder +
pre_block, posterior mean, fp16 hi / lo convolutions as the audio decoder,
pre_block in fp32 on the host; weights `audio/audio_enc_f16.bin`). Ref2VA's
presentation (labels, timestamps, a video's per frame pair as a video block)
is the server's `videos.ref2va_presentation` and `TextEncoder::encode` with
every block's frames (`referenceVisionImages`). On a Ref2VA example (a video
and two images) the reference latents are at relRMS 4.4e-4 (video, the saved
frames), 5e-4 (images) and 1.7e-3 (soundtrack) against a reference run's saved
conditioning, the token ids exact.
A request's references reach the engine in its `VideoRequest`
([Native protocol](internals.md#native-protocol)); `TextEncodingStage` gives the text
encoder their vision blocks (`referenceVisionImages`) and `PrepStage` runs
`encodeReferences` (an audio reference through the audio VAE's encoder alone,
its rows `ref_audio`) before `Prep::run`, then releases their pixels; the
variant is `pika-ref2va-6` (its DiT `dit-pika-ref2va-6`), and the engine
refuses references with keyframes or on a DiT without the references'
adaLN tables. `Prep::run` takes the same references: the
video condition rows are augmented with the request's first MLX draw (as the
keyframes' noise) unless `referenceNoise` is given, the schedule gains t = 1,
and the adaLN tables are `adaln.references.*` of `adaln.safetensors`, which
`Prep::tabulateAdaLn` writes for a FastH3-layout DiT whose transformer
directory holds `blocks.N.adaln.clean`. The token refiner's two blocks share
their scratch (0.28 MB a text row). `build/engine-tests/video-ref2va-layout DUMP` checks it
against a reference run's saved tensors (DUMP, `dev/tools/video_ref2va_layout.mm`): positions,
tags, step-0 timestep indices, index arrays and condition-row counts bit for
bit, the step-0 target and audio condition rows bit for bit; the video
condition rows' noise comes from Pika's CUDA generator, so the tool reports
them clean (relRMS 1.3e-3) with the noise they imply (std 1.00). It also
prints the dense DiT's activation plan for the sequence: 3.7 GB for the
example's 39,729 rows at 832x480 and 10.4 GB for 111,057 at 1344x768, against
1.4 GB and 3.3 GB for text-to-video, and the refiner's scratch (1.8 and
5.9 GB, released before the DiT).

The installer converts the Ref2VA DiT as a spec of `install/fasth3.py`'s
recipe: `MiniMaxAI/MiniMax-H3:pika-ref2va-6` reads Pika's `transformer_ref/`
(diffusers names) and its ref2va adapter from the Hub, writes each block's
adaLN row of t = 1 to scratch, from which `splash video-adaln` writes the
references' tables, and converts the audio VAE's encoder
(`audio/audio_enc_f16.bin`). The layout check runs on its assembly
(ASSEMBLY below).

```sh
build/engine-tests/video-ref2va-layout DUMP --forward ASSEMBLY --metallib LIB --out DIR
```

`--forward` runs `Prep::run` on the dump's `prompt_embeds`, tags and
reference latents, puts the dumped condition rows (`step0.npz`, augmented
with Pika's noise) and initial noise in place of the engine's draws, and
compares one DiT forward's target velocities with `step0_velocity.npz`.

## Video tests

`make test-engine-metal` runs the encoder's attention kernels against a CPU
reference and one DiT block on synthetic weights against a reference implementation's hashes;
`make test-engine-cpu` checks FastH3's tile map and schedules
(`video_vsa_table_test.mm`) and, for every named canvas and clip length,
the geometry, key-range table, tile map, video VAE plan and activation
plan derived from its grid (`video_canvas_test.mm`), and the adaLN tables' host time
embedder against an fp64 reference (`video_temb_test.mm`).
`dev/tests/server/test_video_references.py` checks the server's references
(kinds, limits, normalization, the variant, the multipart field, the
presentation and the wire frame) on synthetic media; with
`SPLASH_REF2VA_RECORD_SHOP` (a Ref2VA example's inputs, a video, two images
and a prompt, with a reference run's saved `conditioning.npz` of them) and
`SPLASH_REF2VA_TOKENIZER` it requires the server's token ids identical to the
saved ones.

The real-model tests compare the engine with a reference directory,
`VIDEO_REFERENCES`: a reference implementation's run of the same prompt,
seed and canvas. The reference directory and the Ref2VA example's inputs are
not part of the repository; `dev/tests/video_real.py` requires
`--references`, and the targets below refuse to run unless
`VIDEO_REFERENCES` names a directory. The tests read from it:

| File | Holds | Read by |
| --- | --- | --- |
| `prompts/golden.npz` | the golden prompt's token ids (`ids`) and hidden states (`hidden`) | every target |
| `golden_3s_c5r1_both/final.npz` | the 3 s golden clip's latents | `test-video-real`, `test-video-http-real` |
| `golden_ref/frames.npy` | the 3 s golden clip's RGB frames | `test-video-real`, `test-video-http-real`; frame 0 for `test-video-first-frame-real` |
| `golden_i2v_3s/final.npz` | the latents of the 3 s golden clip from frame 0 | `test-video-first-frame-real` (`VIDEO_FIRST_FRAME_FINAL`) |

With the weights, `VIDEO_REFERENCES`, and `VIDEO_ROOT` a prepared model (by
default the selection link of `VIDEO_MODEL`, `MiniMaxAI/MiniMax-H3:vdn-8`,
not the `:pika-6` a plain `splash serve --model MiniMaxAI/MiniMax-H3`
installs) or `VIDEO_CONVERTED` a directory of images converted before:

```sh
make test-video-real VIDEO_REFERENCES=... VIDEO_CONVERTED=...
make test-video-http-real VIDEO_REFERENCES=... VIDEO_CONVERTED=... VIDEO_SECONDS=15
make test-video-first-frame-real VIDEO_REFERENCES=... VIDEO_CONVERTED=...
make test-video-canvas-real VIDEO_REFERENCES=... VIDEO_CONVERTED=... VIDEO_SIZE=832x480 VIDEO_SECONDS=5
```

The tests run the golden-clip engine (`dev/tools/video_golden.mm`): the
model's runtime with the text encoding replaced by the golden prompt's saved
hidden states and the denoising and decoding followed by writing their
latents and frames. `test-video-real` requires the 3 s golden clip's latents
and frames byte-identical to the reference directory's, then cancels a clip;
for a FastH3 variant, `dev/tests/video_real.py engine --golden-final FINAL`
requires the latents of a reference clip of the same variant.
`test-video-http-real` does the same through `/v1/videos`, deletes a running
job, then runs a clip of `VIDEO_SECONDS` from the server's tokenizer and the
engine's encoder. `test-video-first-frame-real` sends the golden frames'
first frame as a PNG `input_reference` and requires the clip's latents
byte-identical to `VIDEO_FIRST_FRAME_FINAL` (default
`VIDEO_REFERENCES/golden_i2v_3s/final.npz`). `test-video-canvas-real` makes
a clip of the golden prompt on `VIDEO_SIZE` for `VIDEO_SECONDS` and reports
each phase's time, the wall time and the engine's peak footprint;
`VIDEO_KEEP` keeps its latents and frames for a comparison with another
engine's clip of the same canvas, seed and noise. The reports go to
`build/release/video-*.json`.
