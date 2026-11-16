# Models

## Upstream model loading

A target is identified by its own configuration: an MLX `config.json`, or the
one `gguf.model_config` derives from the selected GGUF's header, read with a
few HTTP range requests. Before any weight file is downloaded, the installer
runs the engine's own check on it, `model-check` (`build/splash model-check`
in a checkout, `engine/splash model-check` in a packaged installation), which
holds it to the rules every start applies to the installation
(`inspectSourceConfiguration` in `ModelDescriptor.mm`) and names the family
it describes. For a GGUF the check also takes the headers of its files,
which the installer copies as it reads them, and plans the target from them
as every start plans the file (`gguf::planImages`): its metadata (sizes, RoPE
base, dimensions and scaling, RMS epsilon, GDN sizes, FFN widths, experts and
a family's arrays) and each tensor's presence, type and shape. The target's
text configuration is checked before its vision tower (a GGUF's projector, an
MLX target's processor and the shards holding its tower) and the tower's
configuration, so a model of no family Splash serves is told so whatever its
tower, and the check runs again with the draft's `config.json` once the
family's draft is chosen.
A GGUF that holds most of its values unquantized is refused first
(`gguf.require_quantized`, [GGUF targets](#gguf-targets)), and a model the
engine refuses with its reason; no weight file is downloaded for either. An MLX
target's tensor shapes, and an MLX vision tower's weights, which must not be
quantized (`VisionLoader.cpp`), are checked only when the model loads.
`families.FAMILIES` names the draft trained for each family; repository names
and model-card `base_model` fields play no part. Families of one model type,
Qwen3.6-35B-A3B and Qwen3.5-122B-A10B (`qwen3_5_moe_text`, GGUF `qwen35moe`),
are told apart by their sizes, and a fine-tune of either, such as
Ornith-1.5-35B-A3B or Ornith-1.5-122B, is its family. A layout type
(`Qwen3_8.hpp`, `Qwen3_6Moe.hpp`, `Qwen4Exp.hpp`, `Flint.hpp` in
`runtime/model/`) is an architecture, whose code its families share, and each
family is a value of it in one list, `model::kFamilies` (`ModelDescriptor.hpp`):
another size of an architecture is one more value there, built by the
constructor from a `QwenHybridLayout` its layout type gains with its second
family (as `Qwen3_6MoeLayout`'s builds `kQwen3_5_122BLayout`), its draft in
`families.FAMILIES`, and, for a GDN or attention shape no family had, its kernel
instances (of bf16 norms only where a family of it loads safetensors) and a row
of the compiled shapes in `ops/GDN.cpp` or `ops/PagedAttention.cpp`. The kernel
tests take every family's shapes from that list
(`dev/tests/engine/TestFamilies.hpp`). A Qwen config holds its text model
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

`--model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` selects the repository's GGUF
target for that variant (`upstream.select_gguf`). A target is a root-level
`.gguf` file (a lower-case extension, as the native loader requires), the
parts of a split GGUF at the root named by their model
(`MODEL-00001-of-0000N.gguf ...`, `upstream.root_targets`), or the parts of a
split GGUF in a folder, as repositories publish files too large for one
(`VARIANT/MODEL-VARIANT-00001-of-0000N.gguf`, `upstream.folder_targets`). The
variant names, in this order: the root target named with the model name the
root targets share, then `-UD-Q4_K_M`; the folder named `UD-Q4_K_M`, in any
case; the only root target whose name ends in `-UD-Q4_K_M`, which the
installer reports when the repository holds several targets. A whole file
beside split parts of its name, such as their merge, is a target of its own,
which the name selects before the parts, by the variant or, as a repository's
or a local directory's only name, without one (`upstream.named_target`).
Before any weight download, the
engine's check plans the target from its header, which must list every tensor
the loader reads, of its shape and of a type it accepts for that tensor; one
error names each missing or unsupported tensor once, with the layers that
have it (`ffn_gate_exps.weight (Q8_1) in 40 layers`). The types it accepts:

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
`tokenizer.ggml.eog_token_ids`, or, in a header without it (any but
unsloth's), the IDs of Flint's stop tokens in its vocabulary, each a control
token, one of them its EOS; the engine's check holds each to the family's,
and the header's layer patterns to them. The token table must be of a format:
RecursiveSI's int4 GGUF, whose table is F32, is refused before its download.

A split GGUF is the parts of one name, `NAME-0000K-of-0000N.gguf`: every part
from 1 to N must be present, at the root or in its folder, and a folder that
mixes the parts of several names is refused (`upstream.split_order`, and
`findTargetGgufs` in the engine). The parts install side by side in
`target/`, and the native loader reads them as one GGUF (`findTargetGgufs`,
`OpenGguf`): the first part's metadata, whose `split.count` and
`split.tensors.count` must describe the parts, and every part's tensors. The
installer copies each part's header under the part's name, and the engine's
check reads them so.

BF16 and F16 are formats of the image like the quantized ones, for the tensors
a quantized file keeps unquantized: UD-Q8_K_XL keeps the 27B's attention
projections and LM head and some of the 35B-A3B's projections and experts in
BF16, and other publishers keep alpha and beta, GDN projections, experts or
the token table in F16. Their planes hold the values as stored, with an FP32
scale of 1 per 256 values; the staged tiles round each value once to half, the
Apple9 register tile reads BF16 exactly and takes no F16, so a projection with
an F16 weight stages there. Whole-BF16 and F16 models are not a target: before
any weight download, and before the engine's model check, the installer
refuses a GGUF whose BF16, F16 and F32 tensors hold more of its values than
its quantized ones (`gguf.require_quantized`), with one message that asks for
a quantized variant. A whole-BF16 file with its token table quantized is still
refused; these files pass, and so does a MoE whose routed experts alone are
quantized. The native loader does not check this.

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
a weight of another type is refused, as only NVFP4's rows hold one. An F32
`.input_scale` beside an NVFP4 weight (NVIDIA's activation scale, as many
values as `.scale`) plays no part, as in Model Optimizer's targets; beside any
other weight, or alone, it is refused. It loads as Model Optimizer's NVFP4 does
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
PQ2_0 token table, and GDN value heads in grouped order (`GgufFile` checks the
parameters and the planner the rest, both before any download through the
engine's model check).

On M5 and later Macs, whose GPUs multiply int8 at twice their bf16 rate, such a
model decodes in int8 activations, one scale per row and block of 1024 rotated
inputs (next-token KL 2.8e-4 against bf16, 13-16% less time a decode cycle on a
20-core M5 Pro), which `--activations bf16` turns off
([server options](server.md#server-options)).

## Drafts

Every model decodes with a DFlash2 draft. Each family names the repository
of the DFlash2 checkpoint trained for it (`families.FAMILIES`), which holds
it as the release publishes it: `config.json` and BF16 `model.safetensors`
(or the shards its index names), and `mask_embedding.safetensors` when the
release ships its mask token's embedding. Installation downloads only those files and follows the repository's default
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
alike: a decode cycle drafts eight rows, the anchor and seven proposal
positions, and the target verifies a tree of the draft's candidates at those
positions and of its host chain (below), as many of its rows as the cycle's
step gives the lane, up to sixteen (`ops::DraftSelector::addVerifyRows`), no
more of them, and none deeper, than the lane may still retain: the tokens its
output budget leaves. The host chooses a cycle's step
before the cycle runs (`model::StepCosts::choose`): the rows, from one a
lane to sixteen, with the most expected accepted tokens over the cost of a
step of that many rows, from each lane's chances, no more of them than the
tokens it may still retain past its anchor (its draft's rows' estimates in
its latest tree with a draft child of its anchor, which grows to the tokens
the lane may retain whatever its step, every row's 1 before its first, a
cycle of a forced chain, beside whose tokens the draft's rows are dead,
leaving them as they were; and its host chain's, or without one those of its
last chain's path that its draft joined), and a table of what a step costs
by its class: the steps whose kernels are the same (`ops::ExecutionPlans::
decodeRowClasses`: a projection's tile and staged height, a float tile, a MoE
target's expert tile), priced by a step of the class's last rows, up to the
rows the target runs in one pass (a streamed-expert store's round of 32, else
sixteen a lane). The start measures the table on the Mac before it serves
(in the decode warmups, three passes over a step of each class at each lane
count, each step's median wall time: a first start's decode warmups of
Qwen3.8-27B on a 48 GB M5 Pro take 11.1 s, a later start's 1.0 s), later
starts of the same build, models and formats on the Mac read it back from
beside the Neural Engine's calibrations instead, and no cycle changes it. The
table's steps ran at a few tokens of history, and each attention pass, one
for a lane's first eight rows and one more for its rows past them, reads the
lane's whole history (on Qwen3.6-35B-A3B on a 48 GB M5 Pro a step at 32K
takes 2.2 to 3.6 ms a pass more than at 2K), so a step costs its table's
price plus k times its passes
over history: over its lanes, a lane's history in thousands of tokens times
the passes its rows take. k is one number for the model, none at each start,
which serving fits: the slope of its cycles' time less their table's price on
their passes over history, the latest cycles weighing most, from the cycles
the costs chose that held no constrained lane
(`model::StepCosts::observeContext`). A
lane count whose lanes' state a start cannot hold beside the rest of its
memory (under `--max-memory`, or where the warmup's allocation fails) is not
measured: its cycles take a draft's block a lane, `--verify-rows all`'s step,
as far as the lanes' chances fill it,
until a start that holds them measures it. A lane holds up to eight rows
unless the step's rows give one lane more of the lanes' chances, sixteen
then, at what the 16-row lane cap's GDN kernels and attention tiles add to a
step of eight rows a lane, which the start also measures. Every target
verifies every row of its
step that its lanes' trees fill: a dense target's projections read their weights once for
all of a step's rows, so a row costs it little, and a MoE target's step,
whose rows each route their own experts, which a rejected row reads for
nothing (a Qwen3.8-Flash-Next cycle of one lane routes about 43 experts a
layer with all eight rows and 17 with two), from the SSD where they stream
([Streamed experts](server.md#streamed-experts)), is priced by the same
costs of its steps' rows. `--verify-rows all` has every cycle's step verify
every row of a draft's block a lane, so that greedy and seeded outputs
repeat. The startup log says which rows the cycles
verify and why, and `/status` reports it as `verify_rows`. The selector grows
each lane's verify rows into a tree over its candidates (`draft_select_tree`):
from the anchor, best first by the estimate that the target accepts a row's
path, the product of its steps' estimates, each row taking its most likely
candidates at the next position as children, and a lane whose thinking end starts at
its anchor takes the end as a chain of its own, whose rows the target selects
whatever its logits. A lane not thinking under a budget also proposes a
prompt-lookup chain beside them:
the tokens that followed the latest earlier occurrence of its last three or
more tokens (`model::LookupIndex`), each estimated by the match's length, its
depth and whether the draft's candidates there hold it
(`model::LookupAcceptance`, fit on traced cycles and refined by every
unconstrained cycle's outcome under `--verify-rows auto`, greedy and sampled
lanes apart: a sampled lane accepts a copied token only as often as it draws
it); a lookup token that is
a draft row's joins that row. A
constrained lane whose grammar forces the tokens after its anchor proposes
those instead, each estimated at 1: the server finds, with a cycle's masks,
the tokens the grammar forces after each row's path, and the next cycle takes
those of the path the lane retained where its anchor begins them (on 15
structured outputs of Qwen3.8-27B on a 48 GB M5 Pro, the masks admitted the
forced token alone at every one of 847 forced positions, and the chains
would add 1.4% to the tokens a cycle accepts, 16% on an answer whose long
keys only the schema names). A step to a candidate of probability q (the softmax at
temperature 1 of its position's candidates' scores after its parent's) is
estimated by an online model, logit(p) = slope * logit(q) +
intercept[position], the probability that the target accepts it once it
accepted its parent, at most what the parent's host chain child (at its
chain's rate) and the candidates the parent drew before it left of 1 (its
cap), since the target selects one token after the parent: the estimates of a
row's children sum to at most its own. Before the chain's child counted, on
Qwen3.6-35B-A3B's 31K-token code prompts, a chain's first rows of rate 0.9 or
more were retained 0.95-0.99 of the time and the draft rows beside them
estimated 0.19-0.20 and retained 0.02-0.09, the anchor's children summing to
1.14-1.16. There is one model for greedy lanes and one for sampled lanes,
whose drawn children their verify rows' rejection sampling accepts. The
engine refits each every 64
cycles on its last 4096 labels: each row of a lane's tree whose parent the
cycle's walk accepted, labeled with whether the target selected the row's
token there and carrying the row's cap (a label its cap bounds tells the fit
nothing), a row its lane did not verify too (its parent's selection
says whether the target would have accepted it), but none whose probability
is not one (a lookup chain's row that no draft step joined, a row of a chain
the target selects whatever its logits, or a draft row that is not finite) and
none under a selection a thinking cutoff made; by damped Newton steps under a
prior fit to Qwen3.8-Flash-Next's and Qwen3.6-35B-A3B's cycles, which also
stands before any label. Labels of a tree's rows, not only of its likeliest
chain, keep the estimates of a row's siblings true: over 7,309 greedy traced
cycles of Qwen3.8-27B, each workload under the fit of the others', trees of
16 rows grown with the fit of their rows' labels had those rows' estimates
within 0.012 of the rate the target accepted them at (the mean over tenths of
the estimate), against 0.021 with the fit of their chains' labels, under
which rows estimated at 0.15 to 0.45 were accepted at 0.10 to 0.38. A row's
estimate is what verifying it adds to its lane's expected tokens; one GPU
step after the selector (`draft_verify_rows`) lays out the step's rows: every
lane its anchor and the rows of the largest estimates over the batch, a
lane's in the order its tree took them and none past a lane's cap, as many as
the cycle's rows hold, every one of which its lane verifies. A row's
admission so reads only the rows ranked above it, never its own descendants,
whose estimates follow its draws, which keeps a sampled lane's selections
draws from the target. It numbers a lane's rows in preorder, its likeliest
path on its first rows, packs the lanes' rows in lane order, and writes each
lane's rows (`DecodeTensor::LaneLayout`), which the cycle's later kernels
read on the GPU. Where the trees hold fewer rows than the step (a sampled
lane at a low temperature draws fewer children than its last tree's chances
promised), the step's rows past the lanes' belong to no lane: the target's
projections compute them unread, as a tile's spare rows, each holding lane
0's anchor, but no attention, GDN recurrence, MoE routing
(`moe_group_routes_verified`), selection, penalty or acceptance reads them. A sampled lane's draft draws a
row's children without replacement from its candidates at the lane's
temperature (their Gumbel-top-k), which its tree takes in the order drawn, and
each of its verify rows selects by recursive rejection sampling over them: it
keeps child i, of probability p under the target and q under the draft after
the children before it, with probability min(1, p / q), the target's
distribution becoming its residual max(p - q, 0) and the draft's losing the
child after each rejection, and draws from the last residual where it keeps
none (`metal/abi/Sampling.h`). Its cycle keeps a proposal only where the row
before it selected that token, so its acceptance stays exact; with one child
a row this is the rejection sampling of a chain of draft draws, and every
child more accepts more. A greedy lane's argmax is the same rule at
temperature 0, its children its draft's most likely. On Qwen3.8-27B's own
sampled cycles at temperature 1, top_k 20 and top_p 0.95, a row that selected
by its draw alone, of its draft's most likely child, kept 0.671 of the
proposals at each position, where this rule keeps 0.701, and 0.785 and 0.841
of its proposals with two and four children. A fixed shorter block would do
worse: in agentic use about one cycle in six accepts all seven proposals and
one in five accepts none.
The step costs are those the Mac measured at a start, the cost of history
they add follows the cycles served, and its acceptance
model follows every request's labels, so the rows a cycle verifies depend on
the machine and on what the server served before, and a MoE row's result
depends on which rows share its expert tiles ([Kernels](internals.md#kernels)): under
`--verify-rows auto` the same greedy (temperature 0) or seeded request can give
different text from run to run, even alone on an idle server
([Sampling](api.md#sampling)); sampled outputs keep their exact distribution. Under `--verify-rows all`, the rows a
cycle verifies follow its lanes' output budgets alone. The speculation counts ([Output limits and
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
weight download and not installed. The models of a family without a vision
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
and seconds for a 4-megapixel image on a small GPU, which the scheduler,
timing it apart, counts in what the prefill owes decoding requests and
leaves out of the measured time it sizes later prefill commands by, and
which Metal would count against the 5 s it gives the prefill command's first
Neural Engine wait ([Host steps](internals.md#host-steps)).

`--language-only` links and loads no vision weights and removes them from
memory accounting. It skips a GGUF's mmproj download and an MLX shard that
holds only the tower (OptiQ keeps the tower in a subdirectory shard of its own);
vision tensors in the language model's shards download with them. The native
Ready event
announces vision only when the model loaded it. Without it, image and PDF input
fails with a 400 naming the modality. Every API shape converts its media to
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

Only the chat template writes control tokens. A tokenizer finds the tokens it
adds beside its vocabulary wherever a prompt spells them: the turn and role
tokens, `<tool_call>`, `<tool_response>`, `<think>`, an image's
`<|image_pad|>`. Text that only quotes one, as a tool result showing a
parser's `tool_call_start = "<tool_call>"` does, would reach the model as the
token itself, and a `<|im_end|>` in a fetched page as the end of its turn. So
request preparation marks every added token of the tokenizer, special or not,
where it begins in all that a request gives the template: the content of
messages of every role, an assistant's reasoning and calls, tool results, tool
definitions, `chat_template_kwargs`, a PDF's text and a judgment's row. It
marks one a text spells whole, and one the text's end cuts off, which the
characters the template writes next could complete: Qwen's templates write `>`
after a call's name and after each of its argument's names, so a name ending
in `<|im_end|` would end the turn, and a message's text parts follow one
another. The prompt encodes each marked one as its characters, as a tokenizer
that had not added it reads them, so the tags the template writes itself are
the prompt's only control tokens, and each `<|image_pad|>` among them is one
the template wrote for an image. A prompt that quotes nothing encodes as the
tokenizer encodes it, token for token.

The template renders marked text as any other text, so it finds no tag in a
request either. Three templates read tags there, and none does once the tag
is quoted: Qwen's take a user message that is one
`<tool_response>…</tool_response>` block for a tool's result where they look
for the last question, Qwen3.6's split an assistant's content at `</think>`,
and Flint's deletes `<think>` and `</think>` from a message and rewrites its
role tokens. Each stays the text the client sent. A client sends an
assistant's reasoning as `reasoning_content` and a tool's result as a `tool`
message, which the template frames itself. Flint's template also deletes
U+0000 from a message, which would write whole a token the text spelled around
one, where nothing was marked: a U+0000 in a request's text reaches every
template as U+FFFD. With the templates of the families Splash serves, no text
of a request is then a token. What remains is a
[custom template](#custom-chat-templates)'s own doing: one that deletes or
rewrites characters of a text it is given, or that itself writes the first
characters of a token before a text, can still put a token together. A text
completion's prompt and `/tokenize` are raw text, in which a special-token
string is its token and a U+0000 stays.

Startup refuses a tokenizer whose added tokens it could not keep out of a
request's text, before the native runtime starts: one without a
`tokenizer.json`, one that finds an added token other than as its literal text
(normalized, stripping the space around it or matching whole words only) or
adds one of a single character, and one whose pipeline is not byte-level BPE
behind regex splits with NFC or no normalizer, where the text between two
added tokens might not encode on its own. The tokenizer Splash derives from a
GGUF is always one it serves.

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

## Custom chat templates

`--chat-template-file FILE`, of `splash serve` or the server, names a UTF-8
Jinja file that every request renders with, in place of every template the
tokenizer defines, `tool_use` included; without it, the tokenizer's templates
serve as before. The server reads the file once at startup, probes it as it
probes the tokenizer's, and logs its path (`Chat template file · ...`); a
later change to the file takes effect at the next start. A file that cannot be
read, is not UTF-8 or is blank stops startup, as does a template that would
serve no request, before any weights load: one transformers cannot compile
(the error names the line), one that cannot render a question, a follow-up or
a tool result (the error names the first of them it cannot, with the
template's own error), and one whose prompts for the three do not end with one
generation prompt, which every request's prompt must end with. Splash never
falls back to the model's templates.

The template supplies formatting alone: the vocabulary, special tokens and the
[chat format](#chat-formats) the output is read in remain the model's.
Whether a request thinks, and what `/v1/models` reports of reasoning, follow
the generation prompt the template renders for the request's effort, as with
the model's own. Startup does not check that the template writes the family's
format, so that is the template's to keep: the server reads tool calls only in
the family's layout, whatever layout the template shows the model, and serves
an image only where the template writes the family's image placeholder
(`<|image_pad|>`) once for each image of the request, in their order. The
server takes each placeholder among the prompt's tokens for an image's, a
request's own text being no token, and widens it to that image's tokens; a
request whose placeholders and images differ in number fails with a 400.
Preparation is unchanged: a GGUF still needs a chat template of its own.

The template must also write each text of a request as it is given. Request
preparation sets a mark inside every added token that a text begins and takes
the marks out of the rendered prompt
([tokenizer and chat templates](#tokenizer-and-chat-templates)), so a template
that changes a text's case, cuts it or replaces characters in it leaves a
broken mark in the prompt: `{{ message.content | upper }}` writes
`<__SPLASH_QUOTED_…__TOOL_CALL>` where a message quotes `<tool_call>`. One
that deletes characters inside a text, trims a text before characters of its
own that would complete a token, or writes a token's first characters itself
before a text can put together a token the request only quoted. Writing a
text as JSON (`tojson`), joining it to others, trimming it before a token or a
line break of the template's own and testing it leave it as it is; a test for
a tag in a text finds none, as in the model's own templates.

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
as the format has it, and that it can keep its added tokens out of a request's
text ([tokenizer and chat templates](#tokenizer-and-chat-templates)). Grammars
write reasoning and calls as the format lays
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
holding GGUF targets of one name, or an MLX model directory. It names the target
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
  GGUF targets of one name ([GGUF targets](#gguf-targets)), beside any
  projectors named `mmproj`.
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

The files are checked as a download is, by the engine's `model-check` on their
metadata and a GGUF's headers ([Upstream model
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
