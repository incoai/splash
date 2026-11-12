# Video

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

## Video tests

`make test-engine-metal` runs the encoder's attention kernels against a CPU
reference and one DiT block on synthetic weights against a reference implementation's hashes;
`make test-engine-cpu` checks FastH3's tile map and schedules
(`video_vsa_table_test.mm`) and, for every named canvas and clip length,
the geometry, key-range table, tile map, video VAE plan and activation
plan derived from its grid (`video_canvas_test.mm`), and the adaLN tables' host time
embedder against an fp64 reference (`video_temb_test.mm`).
