# Models

## Upstream model loading

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
tensors of types the loader cannot read or with every projection and expert
unquantized (`gguf.require_loadable`, [GGUF targets](#gguf-targets)); no
weight file is downloaded for either. Tensor shapes, and an MLX vision tower's
weights, which must not be quantized (`VisionLoader.cpp`), are checked only
when the model loads.
`families.FAMILIES` names the draft trained for each family; repository names
and model-card `base_model` fields play no part. Families of one model type,
Qwen3.6-35B-A3B and Qwen3.5-122B-A10B (`qwen3_5_moe_text`, GGUF `qwen35moe`),
are told apart by their sizes, and a fine-tune of either, such as
Ornith-1.5-35B-A3B or Ornith-1.5-122B, is its family. A layout type
(`Qwen3_8.hpp`, `Qwen3_6Moe.hpp`, `Qwen4Exp.hpp`, `Flint.hpp` in
`runtime/model/`) is an architecture, whose code its families share, and each
family is a value of it in one list, `model::kFamilies` (`ModelDescriptor.hpp`):
another size of an architecture is one more value there, its draft in
`families.FAMILIES`, and, for a GDN or attention shape no family had, its kernel
instances and a row of the compiled shapes in `ops/GDN.cpp` or
`ops/PagedAttention.cpp`. The kernel tests take every family's shapes from that
list (`dev/tests/engine/TestFamilies.hpp`). A Qwen config holds its text model
under `text_config`; Flint's is flat, its keys at the top level, which the check
reads only for a model type whose configs are flat (a layout type's
`flatConfig`). Flint loads from a GGUF only, and its configuration is held to
Flint's values key by key (its widths, router, window, `layer_types`,
`no_rope_layer_indices`, `mlp_only_layers` and `eos_token_id`; `softmax_type`,
`rope_theta` and the rest where it states them).
An MLX target must declare `quantization` in `config.json` in formats it loads
([MLX targets](#mlx-targets)). Every start checks the configuration again.
Remote Python code is not loaded.

```bash
splash serve --model mlx-community/Qwen3.6-35B-A3B-4bit
splash serve --model unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M
splash serve --model mlx-community/Qwen3.8-27B-4bit --language-only
```

A model ID with `--revision`, `--language-only` or `--draft-model` is a
separate installation from the same ID without them.

## MLX targets

An MLX target's `quantization` gives a mode, bits and group size, and may give
a module its own, as mlx-lm writes it; a module's entry wins, and its mode
defaults to affine. The engine loads affine 2, 3, 4, 5, 6 or 8 bits in groups
of 32, 64 or 128, mxfp4 (4 bits in groups of 32) and nvfp4 (4 bits in groups
of 16), mixed in any way across modules, and refuses any other format, naming
the module (mxfp8). Every projection, the experts' too, and the head load only
quantized (`safetensors::quantizedModules`): an entry `false`, which mlx-lm writes for
a module it leaves unquantized, is refused for any of them before any weight
download, naming it, and the loader refuses one the checkpoint holds
unquantized. The router, the shared-expert gate, GDN alpha and beta and the
token table may be unquantized. An affine weight is `s * q + z` for its group's
bf16 `.scales` and `.biases` and its code `q`, packed little-endian in
`.weight`'s 32-bit words at every width; an mxfp4 weight is an E2M1 code times
its group's power of two, a uint8 scale, which is GGUF's MXFP4; an nvfp4 weight
is an E2M1 code times its 16 elements' E4M3 scale, a uint8, which is NVFP4
with a tensor scale of 1.

An MLX target loads into the `MDGG0001` images a GGUF target's do
(`SafetensorsTargetLoader`, [Weight loading](internals.md#weight-loading)):
each quantized tensor in the format mlx-lm infers from its tensors, an affine
`af<bits>g<group>` format of `runtime/metal/abi/QuantFormat.h`, MXFP4 or
NVFP4, at the checkpoint's bits per weight (NVFP4's at 4.625, its tensor scale
stored once per 256 weights); projections and experts as block planes, a
quantized token table as native rows and an unquantized one as its bf16 rows.
The MoE router keeps its format too, a quantized one as native rows, an
unquantized one as its bf16 or F32 rows, which the float projection runs
([Kernels](internals.md#kernels)). The MoE selection reads the shared-expert gate in F32,
so it loads as the F32 values of its quantization, and so do GDN alpha and
beta unless both are in one format. Norms stay bf16 and the GDN value heads in
MLX's grouped order.

## Model Optimizer targets

NVIDIA's Model Optimizer checkpoints (`quantization_config` with
`quant_method` `modelopt`), such as `nvidia/Qwen3.8-27B-NVFP4` and
`nvidia/Qwen3.6-35B-A3B-NVFP4`, load as safetensors targets beside MLX's. Each
layer they quantize, by `quantized_layers` or else by `quant_algo` for every
layer the `ignore` patterns leave, must hold NVFP4 in groups of 16 (`NVFP4`,
`W4A16_NVFP4`) or FP8 with one scale per tensor (`FP8`), and every projection
and the head must be such a layer; any other algorithm is refused before any
weight download, naming the layer. An NVFP4 weight is `g * e4m3 * e2m1`: its
E2M1 code (two per byte of `.weight`, U8), its 16 elements' E4M3 scale
(`.weight_scale`, F8_E4M3) and the tensor's FP32 `.weight_scale_2`; an FP8
weight is `g * e4m3`, its E4M3 value (`.weight`, F8_E4M3) and the tensor's
FP32 `.weight_scale`. The kernels read them weights-only, as NVFP4 and FP8 rows
of `runtime/metal/abi/QuantFormat.h` whose meta unit holds `g` beside the
E4M3 scales, so every kernel decodes them through the format table alone;
`.input_scale` and the KV cache quantization, which scale activations on
NVIDIA GPUs, play no part. The checkpoints keep transformers' module names:
`model.language_model.*`, `lm_head`, each routed expert a module
`mlp.experts.<e>` of its own (stacked into one tensor's planes, each expert
with its own `g`), and RMSNorm weights stored 1 below the weight the norm
multiplies by, which load as the F32 `1 + w` (the GDN's gated norm as stored,
widened to F32). Their unquantized tensors (GDN alpha and beta, the router,
the shared-expert gate, the bf16 token table) load as an MLX target's do, and
their MTP modules are not read.

## Compressed-tensors targets

llm-compressor's checkpoints (`quantization_config` with `quant_method`
`compressed-tensors`), such as `unsloth/Qwen3.8-27B-NVFP4` and
`unsloth/Qwen3.6-35B-A3B-NVFP4`, load as Model Optimizer's do. Each config
group's weights must be NVFP4 (`nvfp4-pack-quantized`: float 4-bit by
`tensor_group` in groups of 16) or FP8 (`float-quantized`: float 8-bit by
`channel` or `tensor`), symmetric and in their stored column order, and every
projection and the head, each routed expert's by its own name (the first and
the last expert's before the download, every expert's when its tensors load),
must be a group's target that no `ignore` entry names; a target or entry is a
module's name, a regular expression after `re:` matched from the start of the
name (at most 512 bytes), or the class `Linear`. Anything else is refused before any weight download,
naming the group or the module, so checkpoints that keep the head or a
projection in BF16 (RedHatAI's) are refused. An NVFP4 weight's codes are
`.weight_packed`, its `g` the reciprocal of the F32 `.weight_global_scale`;
an FP8 weight's F32 or BF16 `.weight_scale` is the tensor's `g` or, `[rows,
1]`, each row's, which that row's meta units hold.

## GGUF targets

`--model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` selects the repository's
root-level `.gguf` file (a lower-case extension, as the native loader requires)
for that variant: the file named with the model name its GGUFs share, then
`-UD-Q4_K_M`, or else the only one whose name ends in `-UD-Q4_K_M`
(`upstream.select_gguf`). Before any weight download, its header must list
every tensor the loader reads with a type it accepts for that tensor
(`gguf.loaded_tensors`, checked by `gguf.require_loadable`; a test holds its
format types to `runtime/metal/abi/QuantFormat.h`). The native loader checks again
and lists every unsupported tensor in one error:

- linears and experts: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q4_1,
  Q5_0, Q5_1, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
  IQ4_NL, MXFP4, NVFP4, PQ2_0, Q2_0, BF16 or F16;
- token embeddings: any of the linears' formats;
- norms: F32, BF16 or F16, loaded as the F32 values they hold;
- the MoE router: F32, BF16 or any of the linears' quantized formats (not
  F16), which the float projection runs as stored, a quantized one as its
  native rows (as an MLX target's quantized router loads);
- the shared-expert scalar gate: F32, BF16 or Q8_0, loaded as the F32 values
  it holds;
- the GDN convolution, decay and time-step bias: F32;
- GDN alpha and beta: each any of the linears' formats or F32. Two of one
  quantized type other than BF16 and F16 load as one segment of the GDN input
  projection in that format; any other pair loads as one F32 tensor of the
  values each holds, every type but F32 decoded on the GPU by its format's
  gather (`gguf_embed_f32_<format>`), GGML's own F32 values. A BF16 or F16
  pair stays such a float segment: it reads the input unrotated, as Prism's
  BF16 alpha and beta need, and leaves the GDN input projection on Apple9's
  register tile, which takes no F16;
- Flint's attention sinks: F32, as llama.cpp writes them.

Flint's GGUFs (`flint`, as llama.cpp writes them from release b11803) differ
from Qwen's beyond their tensor types: every layer attends, its norm before the
FFN is `ffn_norm`, `leading_dense_block_count` leading layers have a dense FFN
and the others a MoE whose shared expert has no gate, and each layer carries
its attention sinks, `attn_sinks.weight`. Its derived configuration is flat,
as Flint's config.json is, and holds the keys of that file the header states
(`gguf.flint_text_config`): every size, the experts each token takes, the
router (`router_score_function` from `expert_gating_func`, `norm_topk_prob`),
`mlp_only_layers`, `sliding_window`, `layer_types` from
`attention.sliding_window_pattern`, `no_rope_layer_indices` from
`attention.rope_pattern`, and `eos_token_id` from
`tokenizer.ggml.eog_token_ids`; the engine's check holds each to the family's,
the two layer patterns included, which the scalar metadata leaves out. The
token table must be of a format: RecursiveSI's int4 GGUF, whose table is F32,
is refused before its download.

A variant no root file names may be a folder of a split GGUF's parts, as
repositories publish files too large for one
(`VARIANT/MODEL-VARIANT-00001-of-0000N.gguf`, `upstream.split_gguf`): every
part from 1 to N must be present. The parts install side by side in
`target/`, and the native loader reads them as one GGUF (`findTargetGgufs`,
`OpenGguf`): the first part's metadata, whose `split.count` and
`split.tensors.count` must describe the parts, and every part's tensors. The
installer screens the merged header (`gguf.merged_split`).

BF16 and F16 are formats of the image like the quantized ones, for the tensors
a quantized file keeps unquantized: UD-Q8_K_XL keeps the 27B's attention
projections and LM head and some of the 35B-A3B's projections and experts in
BF16, and other publishers keep alpha and beta, GDN projections, experts or
the token table in F16. Their planes hold the values as stored, with an FP32
scale of 1 per 256 values; the staged tiles round each value once to half, the
Apple9 register tile reads BF16 exactly and takes no F16, so a projection with
an F16 weight stages there, as do all of a model's experts when any expert
weight is F16. Whole-BF16 and F16 models are not a target: before any weight
download, the installer refuses a GGUF whose linears and experts, the LM head
among them, are all BF16, F16 or F32 (`gguf.require_loadable`), with one
message that asks for a quantized variant, not a list of tensors. The token
table, alpha and beta do not count, and one quantized linear or expert is
enough, so these files pass; the native loader does not check this.

Of Unsloth's files in September 2026 that covers every file of Qwen3.8-27B and
Qwen3.6-35B-A3B from UD-IQ1_S to UD-Q8_K_XL. llama-quantize's default type
selection stores alpha and beta in the file type's format (Q4_K in a Q4_K_M),
as in lmstudio-community's files, so those load too. A format's image takes
the bits per weight of its GGUF blocks, but for Q3_K's and Q6_K's padded meta
units (1/16 bit more), IQ3_S's chunk words (4.06 bits for its 3.44) and BF16's
and F16's scales (1/8 bit more). Qwen3.8-Flash-Next's hyper-connection
projections are the exception to loading a GGUF's formats as stored: those of
BF16 or F16 (as some GGUFs keep them) load as Q8_0, the type other GGUFs store
them in, quantized from their values by the reference GGUF quantization of Q8_0:
every decode cycle reads all of them, and Q8_0 holds a value in little more
than half the bytes. The gates they make move by less than one bf16 step.

A MoE layer may hold its routed experts' gate and up weights in llama.cpp's
fused `ffn_gate_up_exps` (`create_tensor_gate_up_exps`, as byteshape's files
do), each expert's gate rows then its up rows. The planner reads each expert's
rows of it into the gate and up tensors, so the images are those of the two
tensors.

A MoE GGUF may keep fewer routed experts than its family, as
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` keeps 256 of
Qwen3.8-Flash-Next's 512. Its derived configuration states the header's
`expert_count` as `num_experts` (`gguf.model_config`), and the engine serves
any count of at most the family's that the MoE kernels route (`ops::MoeShape`,
at least the experts each token takes) and whose router's float projection
fills whole 8-column tiles (`ggufRoutedExperts`): the model check takes that
count, and every start plans, sizes and loads the model by it. Every other
dimension is the family's, and every start's planner holds the header's
`expert_count` and each router and expert tensor to the count. A configuration
without it, which an earlier release derived, states the family's; an MLX
target's count must be the family's.

llama.cpp's NVFP4 (type 40, as in `cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF`)
is `block_nvfp4`: 64 elements, a UE4M3 scale per 16 (an E4M3 byte whose bit 7
llama.cpp ignores, reading 0x7F as 0) and E2M1 codes, beside an optional F32
`.scale` tensor, one value for a projection or one per expert for an experts
tensor (`_exps`), which llama.cpp multiplies the products by; a `.scale` beside
a weight of another type is refused, as only NVFP4's rows hold one. It loads as
Model Optimizer's NVFP4 does
([Model Optimizer targets](#model-optimizer-targets)): four blocks make one
256-element NVFP4 block, whose meta unit holds the `.scale` value as its `g`
(1 without one), at 4.625 bits per weight for the file's 4.5.

Q2_0 is GGUF tensor type 42, `block_q2_0`, as in the GSQ-RCO GGUFs:
2-bit codes q worth d (q - 1) with one half d per 64 weights. Its planes are
PQ2_0's ([PQ2_0](#pq2_0)), with a meta unit per 64 weights.

## PQ2_0

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

## Drafts

Every model decodes with a DFlash2 draft. Each family names the repository
of the DFlash2 checkpoint trained for it (`families.FAMILIES`), which holds
it as the release publishes it: `config.json` and BF16 `model.safetensors`
(or the shards its index names), and `mask_embedding.safetensors` when the
release ships its mask token's embedding; a family whose draft is not released
yet names none, and its installation names one with `--draft-model`.
Installation downloads only those files and follows the repository's default
branch as it follows the target's ([Revisions](server.md#revisions)); `--draft-model`
accepts another repository, followed the same way, or a local directory that
holds them. A checkpoint is installed only when the engine accepts its
configuration with the target's (its `model-check`), so a draft of another
architecture never replaces one that loads.

A draft's `config.json` states its layers, intermediate size, rotary base and
the target layers whose outputs it reads (`dflash_config.target_layer_ids`,
ascending, at most eight); its hidden size, vocabulary and mask token are its
target's (Flint's vocabulary keeps no mask token: a Flint draft chooses one of
its reserved special tokens, 150019 to 150143, as its
`dflash_config.mask_token_id`, which the target's layout then holds,
`QwenTargetDimensions::firstDraftMaskToken`), and its attention (32 query and
8 KV heads of 128), selector (rank 256, 16 candidates) and 2048-token window are
those the kernels are built for, and its block at least their 8 rows, of which a
cycle drafts the first (the Qwen3.5-122B-A10B draft was trained to fill 16;
`readDraftLayout` in `ModelDescriptor.mm`); the kernels take any hidden width of
whole 16-channel convolution groups. A draft's rows after its anchor read the
target's token table row of the mask token, unless its release ships that
token's BF16 embedding, which they read instead (`DFlashDraft::addMaskRows`):
`mask_embedding.safetensors`, one tensor `mask_embedding` of the hidden size.
A quantized table can lose the row: Qwen3.5-122B-A10B's is of order 1e-6,
below what the fp16 block scales of Q8_0 and the K-quants hold, so its GGUFs
read zeros there: on an M5 Max serving unsloth/Ornith-1.5-122B-GGUF:UD-Q4_K_XL
with every row verified, its draft accepted 2.3 tokens a cycle on code
without the embedding, 4.7 with it. Every family loads and runs its draft
alike: a decode cycle drafts eight rows, the anchor and its
seven proposals, and the target verifies their first rows, as many as the cycle
chose for the lane (its verify rows, `ops::DraftSelector::addVerifyRows`), never
more than the lane may still retain: the tokens its output budget leaves, at
most all eight. A dense target verifies every row up to that limit: its
projections read their weights once for all of a cycle's rows, so a row costs it
little. Each row a MoE target verifies routes its own experts, which a rejected
row reads for nothing (a Qwen3.8-Flash-Next cycle of one lane routes about 43
experts a layer with all eight rows and 17 with two), from the SSD where they
stream ([Streamed experts](server.md#streamed-experts)), so by default (`--verify-rows
auto`) a MoE target's cycles choose their rows by expected accepted tokens per
unit of cost (the verify budget, `model::VerifyBudget`), whether its experts
stream or not: on an M5 Max serving Qwen3.6-35B-A3B with its experts all loaded,
one, two and four concurrent agentic requests of 1000 tokens decoded 8.7%, 8.5%
and 2.1% faster under the budget than with every row verified in GGUF UD-Q4_K_M,
and 2.1%, 0.9% and 6.0% faster in MLX 4-bit. `--verify-rows all` has every
target's cycles verify every row. The startup log says which rows the cycles
verify and why, and `/status` reports it as `verify_rows`. The selector writes
each lane's confidence at each position, the largest of its candidates'
probabilities (the softmax at temperature 1 of their scores), which the lane's
draw there does not change, so a sampled lane's acceptance stays exact. An
online model, logit(p) = slope * logit(confidence) + intercept[position], gives
the probability that the target accepts a proposal once it accepted those before
it; the engine refits it every 64 cycles on the last 4096 proposals the target
accepted or rejected (each lane's up to its first rejection, but none whose
confidence is not a probability, which a draft row that is not finite gives), by
damped Newton steps under a prior fit to Qwen3.8-Flash-Next's and
Qwen3.6-35B-A3B's cycles, which also stands before any label. A lane's survival
at a position, the product of p up to it, is what verifying one more row adds to
its expected tokens; one GPU step after the selector (`draft_verify_rows`) gives
every lane its anchor and the rows of the largest survivals over the batch, as
many as maximize the expected tokens over the cycle's cost at that many rows in
all. The engine prices those rows from the wall times of the lane count's latest
cycles, which for streamed experts include the store's reads of its misses: a
least-squares line in the rows over about the last 30 cycles, its slope shrunk
toward the shape the measured targets share, since what a row costs follows the
requests served (a Qwen3.6-35B-A3B cycle with 4 GB of experts adds about 3 ms a
row on prose and 8 ms on reasoning); constrained cycles are fit apart, and every
32nd cycle of a lane count verifies every row, which keeps the rows of the fit
spread. A fixed shorter block would do worse: in agentic use about one cycle in
six accepts all seven proposals and one in five accepts none. Rows past a lane's
verify rows still run the target's layers, whose kernels take eight rows a lane,
but their MoE steps route no expert (`moe_group_routes_verified`) and acceptance
retains none of them. The budget's costs follow wall time and its acceptance
model every request's labels, so the rows a cycle verifies depend on the
machine's timing and on what the server served before, and a MoE row's result
depends on which rows share its expert tiles ([Kernels](internals.md#kernels)): under the
budget the same greedy (temperature 0) or seeded request can give different text
from run to run, even alone on an idle server ([Sampling](api.md#sampling)); sampled
outputs keep their exact distribution. With every row verified (a dense target,
or `--verify-rows all`), the rows a cycle verifies follow its lanes' output
budgets alone. The speculation counts ([Output limits and
streaming](api.md#output-limits-and-streaming)) count the proposals a cycle verified.
The target writes the outputs of the layers the draft reads for each row it
reads; Qwen3.8-Flash-Next's, whose layers carry four hyper-connection streams,
as the streams' mean, summed in fp32 and stored as bf16, as the draft was
trained on it. `dflash_config.target_hidden_reduction` states that reduction
(`stream_mean` for a hyper-connected target, `none` for another); a config
without it takes its target's, and one that names another is refused. Native
loading validates the configuration against the target again and loads the draft
like a target ([Weight loading](internals.md#weight-loading)): `DraftCheckpointLoader`
(`DraftCheckpoint.cpp`) plans the draft's `MDGG0001` images, one per layer and
one for the rest, and `GgufPreparation` quantizes each projection into af4g64
planes, 4 bits in groups of 64 as MLX's affine quantization rounds them, and
copies every other tensor as stored. The draft runs on the block kernels its
target's projections run on.

## Vision

Vision comes from the target repository: MLX's `vision_tower.*` tensors (or
transformers' `model.visual.*`, as Model Optimizer keeps them, whose patch
embedding is already in the image's order), linking
only `config.json` and the shards holding them, or the GGUF repository's root
projector, a GGUF whose name holds `mmproj` (as `mmproj-BF16.gguf` or
`MODEL-mmproj-BF16.gguf`), chosen by its header: a `clip` projector whose
weights are BF16, F32 or F16, preferred in that order. An F16 projector made
from a BF16 tower, llama.cpp's default output, loads exactly: F16 rounds the
smallest BF16 weights onto its subnormal grid, and what that keeps is still a
BF16. The processor configuration (MLX `preprocessor_config.json`, or the
`image_processor` object of `processor_config.json`, where newer Transformers
releases save it; the GGUF's `clip.vision` metadata) must describe the one
preprocessing Splash implements (`server/images.py`); it is checked before any
weight download and not installed. The video processor's (MLX
`video_preprocessor_config.json`, or the `video_processor` object of
`processor_config.json`) is checked too, its preprocessing, and its frame rate,
frame range and pixel budget within what Splash serves (`server/videos.py`),
and installed: the server samples a model's videos by it, with the keys it
leaves out at transformers' Qwen3VLVideoProcessor defaults. A GGUF repository
has none, and its videos are sampled as Qwen publishes for these models: 2
frames per second, 4–768 frames, 4,096–25,165,824 pixels. The models of a family without a vision
tower (Flint; `vision` in `install/families.py`) serve text only: their
installation reads no projector, with or without `--language-only`.

Both sources are written into one image, which the one BF16 vision operator
reads: BF16 tensors are
copied, and F32 or F16 tensors are converted under the exact-BF16 rule of
[weight loading](internals.md#weight-loading).
Unsloth's mmproj stores its 1-D tensors, patch embedding and position table as
F32, all of them BF16-exact, and loads byte-identical to the MLX tower.
Quantized MLX towers, deepstack projectors and mmproj tensors the tower does not
use are rejected.

A prefill command encodes the images whose rows begin in it, from no cached
rows, in a command of its own, and is submitted once that command has
completed: an encode takes about 0.3 s for a 1000 x 1400 page on an M5 Max
and seconds for a 4-megapixel image on a small GPU, which Metal would count
against the 5 s it gives the prefill command's first host step or Neural
Engine wait ([Host steps](internals.md#host-steps)). The scheduler counts the encodes'
time in what the prefill owes decoding requests, and leaves it out of its
prefill estimate.

`--language-only` links and loads no vision weights and removes them from
memory accounting. It skips a GGUF's mmproj download and an MLX shard that
holds only the tower (OptiQ keeps the tower in a subdirectory shard of its own);
vision tensors in the language model's shards download with them. The native
Ready event
announces vision only when the model loaded it. Without it, image, PDF and
video input fails with a 400 naming the modality. Every API shape converts its media to
image and file parts, and message normalization, the one place that accepts or
rejects them, checks before any image is decoded or PDF rendered, in user
turns, tool results and stored Responses history alike. `/status` and
`/v1/models` report `vision: false` and `input_modalities: ["text"]`, and the
launchers configure OpenCode, Hermes and Pi without attachments.

## Tokenizer and chat templates

An MLX target's configuration, tokenizer files and chat template come from its
resolved snapshot. For GGUF, `install/gguf.py` reads the selected file's
metadata without mapping or decoding weight tensors. Vocabulary IDs, BPE merge
ranks, control/user-defined token types, BOS/EOS/padding IDs and template text
come from that file; the supported profiles supply the pre-tokenization
algorithms: `gpt2/qwen35` NFC and Qwen's split, `gpt2/flint` Flint's three
splits (digits in runs of up to three, runs of Han and kana, then o200k's
pattern) without a normalizer, each before byte-level BPE, as the release's
own tokenizer.json has them. Unknown profiles, malformed metadata
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

## Chat formats

Each family's model frames its reasoning and lays out its tool calls in a
format of its own, a value of `server/chat_format.py` that `families.FAMILIES`
names for the family. The server reads and constrains a model's output in the
format of the family its installation found, which the assembly records:

- Qwen's (`QWEN`): the generation prompt opens a think block where the request
  thinks, calls are Qwen3-Coder's XML (`QWEN_XML`), and `<|endoftext|>` and
  `<|im_end|>` stop.
- Flint's (`FLINT`): the generation prompt is `<|assistant|>` alone, and the
  model opens a think block itself, as its first token, when it reasons; calls
  are GLM-4.7's argument tags (`GLM_ARGS`),
  `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`,
  each tag a token; `<|end_of_text|>`, `<|user|>`, `<|observation|>` and
  `<|developer|>` stop.

A format holds the model's vocabulary and the engine's stop tokens, which are
its target layout's; the tokens of `</think>`, of `<think>` where the model
opens its reasoning itself, and of `<tool_call>`; the call syntax; and the
special token after which prompt tokenization reuses a cached prompt's tokens,
`<|im_end|>` or `<|assistant|>`. Startup checks that the tokenizer encodes each
as the format has it. Grammars write reasoning and calls as the format lays
them out: Flint's optional think block may come before any constrained output,
and its tags, which the grammar compiler takes as tokens rather than text, are
written as their tokens. A model that opens its reasoning itself closes it
before a call, so a `<tool_call>` inside its think block is reasoning, as
Flint's own parser reads it.

Flint's template renders as shipped: its roles, the tools block it adds before
the first message that is not a system message, each assistant turn's
reasoning and calls, and tool results as `<|observation|>` turns. Request
preparation merges leading system and developer messages into one system
message for every family, so a developer message renders as `<|system|>`.

## Local model files

`splash serve --model-dir PATH` prepares the target at `PATH` with
`install.local.prepare_local_model` and starts the server on the directory it
returns. `PATH` names a GGUF file, any part of a split GGUF, a directory
holding one GGUF target, or an MLX model directory. It names the target
directly, rather than a library to search by repository ID:

```sh
splash serve --model local/qwen \
  --model-dir /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --draft-model /models/Qwen3.8-27B-DFlash2 \
  --mmproj /models/Qwen3.8-27B-GGUF/mmproj-BF16.gguf
```

`--model` remains required in `OWNER/REPO[:VARIANT]` form as the API's model
name; it selects no files and makes no Hub request in this mode. Preparation
never contacts the Hub, whether or not `--offline` is set. `--draft-model`
must name a local compatible DFlash2 draft directory; without one, the
family's released draft must already be in the Hub cache at the commit its
default branch records, as `hf download REPO` or `splash serve` leaves it. A
missing draft is refused with its download instruction; it is never downloaded
implicitly.

`--mmproj` supplies a local BF16, F32 or F16 GGUF vision projector. Without
one, a GGUF serves text only. An MLX target uses its own vision tower unless
`--language-only` is set. `--revision` cannot be combined with `--model-dir`,
and `--mmproj` requires `--model-dir` and cannot be combined with
`--language-only`. Files are linked, and each start prepares them again under
the same validation, publication and lifetime rules described below.

An application that downloads models itself prepares them for the server
with `install.local.prepare_local_model`, the installer's one public
interface, and starts the server on the directory it returns. It runs in
Splash's own Python (`.venv/bin/python` in a checkout, `python/bin/python3`
in a release) with the checkout or release root on `PYTHONPATH`, and makes no
network request:

```python
from install.local import ModelError, prepare_local_model

model = prepare_local_model(
    "/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf",
    draft="/models/Qwen3.8-27B-DFlash2",
    mmproj="/models/Qwen3.8-27B-GGUF/mmproj-BF16.gguf",
)
```

```sh
PYTHONPATH="$ROOT" "$ROOT/python/bin/python3" -P -m server.server "$MODEL" \
    --model unsloth/Qwen3.8-27B-GGUF --binary "$ROOT/engine/splash" --port 8000
```

`--model` is the ID the API reports, in `OWNER/REPO` form, and `--binary` the
engine of the same checkout or release (`build/splash` in a checkout). The
server takes the [server options](server.md#server-options) of `splash serve`; the
model's source options (`--revision`, `--draft-model`, `--language-only`,
`--offline`) are `prepare_local_model`'s arguments instead, or have no use.

- `target`: a GGUF file, any part of a split GGUF naming all its parts,
  which lie beside it; or a directory whose root holds an MLX model or else
  a single GGUF target, whole or split, beside any projectors named `mmproj`.
- `draft`: the directory of a DFlash2 draft ([Drafts](#drafts)). Without
  one, the family's released draft is taken from the Hugging Face cache, at
  the commit last recorded there for its default branch, as `hf download
  incoai/Qwen3.8-27B-DFlash2` and `splash serve` record it.
- `mmproj`: a GGUF target's vision projector ([Vision](#vision)); without
  one, a GGUF target serves text only, and a target of a family without a
  vision tower refuses one. An MLX target serves images with the tower in
  its checkpoint unless `language_only=True`.
- `models_dir`: where prepared models are kept, Splash's own models
  directory by default ([files Splash writes](server.md#files-splash-writes)).

The files are checked as a download is, by the engine's `model-check` and the
GGUF screening, on their metadata ([Upstream model
loading](#upstream-model-loading)). When Splash cannot serve them without a
download (a missing file or split part, an unsupported format, a draft of
another model), a `ModelError` names the file and the reason. The directory
returned is an assembly of links to the files ([Installation](internals.md#installation)),
which never changes. Preparing the same files again returns it, reading only
their metadata, sizes and times; the first preparation reads each file
outside the Hugging Face cache once, for its digest. Once a file changes, or
a download moves the draft's default branch, preparing again returns another
directory: restart the server on it. The server holds the directory it
serves until it exits; one that no preparation returns any more is removed
only once no server holds it. The server takes the installation lock of
`models_dir` to hold it when it starts, so it needs write access there.

`prepare_local_model` and `ModelError` keep their names, parameters,
defaults, results and errors across the releases of one major version; a
release may add keyword parameters whose defaults keep that behavior.
Nothing else in `install/` is public, nor is the layout of the prepared
directory, which is the server's and the engine's of the release that
prepared it: after an upgrade, prepare again, as `splash serve` installs at
every start.

## Splash packages

Splash packages, such as `incoai/Qwen3.8-27B-Splash`, are the prebuilt format
of earlier releases, which Splash no longer loads. A `--model` that names one,
installed or on the Hub, stops and names the MLX model of its family to serve
instead: `mlx-community/Qwen3.8-27B-4bit` or
`mlx-community/Qwen3.6-35B-A3B-4bit`, which load the packages' weights byte for
byte but for the 27B's GDN decay vectors, each within a float ULP. For an
installed package it also names the Hub cache folder holding the package's
files, which nothing reads any more and can be deleted. A package is
recognized by its `manifest.json`, which names the package format.
