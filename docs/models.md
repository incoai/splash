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
target's tensor shapes are checked only when the model loads; an MLX vision
tower's weights, which must not be quantized (`VisionLoader.cpp`), once its
files have downloaded ([Vision](#vision)) and at every load.
`families.FAMILIES` names the draft trained for each family; repository names
and model-card `base_model` fields play no part. Families of one model type,
Qwen3.6-35B-A3B and Qwen3.5-122B-A10B (`qwen3_5_moe_text`, GGUF `qwen35moe`),
are told apart by their sizes, and a fine-tune of either, such as
Ornith-1.5-35B-A3B or Ornith-1.5-122B, is its family. A layout type
(`Qwen3_8.hpp`, `Qwen3_6Moe.hpp`, `Qwen4Exp.hpp`, `Flint.hpp`,
`Inkling2Loop.hpp` in `runtime/model/`) is an architecture, whose code its
families share, and each family is a value of it in one list,
`model::kFamilies` (`ModelDescriptor.hpp`):
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
`flatConfig`). Flint loads from a GGUF or an MLX checkpoint, and its configuration is held to
Flint's values key by key (its widths, router, window, `layer_types`,
`no_rope_layer_indices`, `mlp_only_layers` and `eos_token_id`; `softmax_type`,
`rope_theta` and the rest where it states them).
Inkling 2's layout states the dimensions the shared runtime runs on as the
layers a step runs, one per layer application (48: its 24 weight layers, the
loop's twelve three times), and keeps its weight layers, its loop and the
mapping of an application to its weight layer in its own members
(`Inkling2LoopSchedule`); its layers attend through relative attention, not
the shared attention, in the two variants its layout states, a full layer's
and a sliding layer's (a layout's `relativeAttention`, none for a family of
the shared attention): its target plans them, the start warms their kernels
and the kernel tests take them from it, and the tests' shared-attention
shapes leave it out. It is served up to Splash's serving ceiling,
1,010,000 tokens, below its own window of 1,048,576. It loads from a GGUF or
from its release's checkpoint or an MLX conversion of it ([Inkling 2's
checkpoints](#inkling-2s-checkpoints)); its configuration, under
`text_config`, is held to its values key by key (its weight layers, its own
window at `model_max_length`,
`unpadded_vocab_size`, `local_layer_ids`, `loop_config`, its experts' width;
the relative bias, clips, softcaps, routing and the rest where it states
them), its clip and log-scaling alpha as F32 values. Its vocabulary keeps
no mask token: its draft reads its last unused special token,
`<|unused_200056|>` ([Drafts](#drafts)).
An MLX target must declare `quantization` in `config.json` in formats it loads
([MLX targets](#mlx-targets)). Every start checks the configuration again.
Remote Python code is not loaded.

```bash
splash serve --model mlx-community/Qwen3.6-35B-A3B-4bit
splash serve --model unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M
splash serve --model mlx-community/Qwen3.8-27B-4bit --language-only
```

A model ID with `--revision`, `--language-only`, `--draft-model` or
`--draft-revision` is a separate installation from the same ID without them.
The app downloads and updates a model with Text only on as its
`--language-only` installation, and
lists the model installed so alone as it lists one installed without it;
deleting it in the app removes every installation of the model.

## MLX targets

An MLX target's `quantization` gives a mode, bits and group size, and may give
a module its own, as mlx-lm writes it; a module's entry wins, and its mode
defaults to affine. The engine loads affine 2, 3, 4, 5, 6 or 8 bits in groups
of 32, 64 or 128, mxfp4 and mxfp8 (4 and 8 bits in groups of 32) and nvfp4 (4
bits in groups of 16), mixed in any way across modules, and refuses any other
format, naming the module. A projection, an expert or the head the checkpoint
holds unquantized (an entry `false`, as mlx-lm writes for a module it leaves
so), in BF16 or F16, loads as Q8_0, quantized from its values by the reference
GGUF quantization of Q8_0 (`quantize_row_q8_0_ref`), as a GGUF's BF16
hyper-connection projections do ([GGUF targets](#gguf-targets)): every decode
cycle reads every projection, and Q8_0 holds a value in little more than half
the bytes of a 16-bit float. Experts whose 16-bit values are exactly MXFP8's
load as MXFP8 instead, as a GGUF's do. The router, the shared-expert gate, GDN alpha and
beta and the token table load unquantized as stored. An affine weight is
`s * q + z` for its group's bf16 `.scales` and `.biases` and its code `q`,
packed little-endian in `.weight`'s 32-bit words at every width; an mxfp4
weight is an E2M1 code times
its group's power of two, a uint8 scale, which is GGUF's MXFP4; an mxfp8 weight
is an E4M3 value, a byte of `.weight`, times its group's power of two, the
same scale; an nvfp4 weight is an E2M1 code times its 16 elements' E4M3 scale,
a uint8, and the global scale its module holds where its converter kept
NVIDIA's (MLX's `global_scale` over the 6 × 448 MLX's dequantization divides
it by, or Model Optimizer's `weight_scale_2`), which is NVFP4 with that tensor
scale, 1 without one, in every family's checkpoints; an experts module holds
one per expert, its experts' scales. Rows of a
multiple of 64 inputs but not of 256 take NVFP4's 64-element form, as a
GGUF's do ([GGUF targets](#gguf-targets)).

An MLX target loads into the `MDGG0001` images a GGUF target's do
(`SafetensorsTargetLoader`, [Weight loading](internals.md#weight-loading)):
each quantized tensor in the format mlx-lm infers from its tensors, an affine
`af<bits>g<group>` format of `runtime/metal/abi/QuantFormat.h`, MXFP4, MXFP8
or NVFP4, at the checkpoint's bits per weight (NVFP4's at 4.625, its tensor scale
stored once per 256 weights); projections and experts as block planes, a
quantized token table as native rows and an unquantized one as its bf16 rows.
The MoE router keeps its format too, a quantized one as native rows, an
unquantized one as its bf16 or F32 rows, which the float projection runs
([Kernels](internals.md#kernels)). The MoE selection reads the shared-expert gate in F32,
so it loads as the F32 values of its quantization, and so do GDN alpha and
beta unless both are in one format. A Qwen hybrid's norms stay bf16 (Flint's
widen to the F32 its kernels are compiled for) and the GDN value heads in
MLX's grouped order. A checkpoint's names are mlx-vlm's (`language_model.model.*`,
as the Qwen releases') or mlx-lm's (`model.*` and `lm_head`, as
`mlx-community/flint-mixed-3-4bit`'s), told apart by where its token table is,
and each family's tensors are read in the order its GGUF images hold them
(`safetensors::planImages`), Inkling 2's by the sections its GGUF plan lays
out ([Inkling 2's checkpoints](#inkling-2s-checkpoints)). One planner serves
both, as `GgufImage.cpp`'s does a GGUF: the families' walks and Inkling 2's
sections plan through the same Builder's primitives (`SafetensorsImage.cpp`).
Flint's sinks (`self_attn.softmax_offset`) widen to the F32 its GGUF holds, its
leading layers' FFN is dense and its shared expert, two routed experts wide and
ungated, runs as two slices.

## Model Optimizer targets

NVIDIA's Model Optimizer checkpoints (`quantization_config` with
`quant_method` `modelopt`), such as `nvidia/Qwen3.8-27B-NVFP4` and
`nvidia/Qwen3.6-35B-A3B-NVFP4`, load as safetensors targets beside MLX's. Each
layer they quantize, by `quantized_layers` or else by `quant_algo` for every
layer the `ignore` patterns leave, must hold NVFP4 in groups of 16 (`NVFP4`,
`W4A16_NVFP4`) or FP8 with one scale per tensor (`FP8`); any other algorithm
is refused before any weight download, naming the layer, and a projection or
the head they leave unquantized loads as Q8_0, as an MLX target's does. An
NVFP4 weight is `g * e4m3 * e2m1`: its
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
`channel` or `tensor`), symmetric and in their stored column order; anything
else is refused before any weight download, naming the group. A projection no
group quantizes, or one an `ignore` entry names, such as the BF16 head of
RedHatAI's checkpoints, loads as Q8_0, as an MLX target's does. An NVFP4
weight's codes are
`.weight_packed`, its `g` the reciprocal of the F32 `.weight_global_scale`;
an FP8 weight's F32 or BF16 `.weight_scale` is the tensor's `g` or, `[rows,
1]`, each row's, which that row's meta units hold.

## Inkling 2's checkpoints

Inkling 2's release (`thinkingmachines/Inkling-2-Mini`, `model.llm.*`) and its
MLX conversions load into the images of its GGUFs
(`Inkling2LoopSafetensors.cpp`): each section its GGUF plan lays out
(`Inkling2LoopSections.hpp`) is read from the tensors a checkpoint names for
it, by the release's names or by those of its converters, which are aliases of
them: oMLX's (`model.*`, a sliding layer's queries, keys, values, relative
scores, gate and sinks fused into one `in_proj`, a full layer's queries,
relative scores, gate and sinks into its `q_proj` and its taps' projections
into its `k_proj` and `v_proj`, a loop layer's message gates transposed in two
tensors), mlx-vlm's (`model.language_model.*`) and mlx-lm's
(`language_model.model.*`, its experts' gate and up apart), told apart by the
token table's name. The release's routed experts are NVFP4 in layers 0-19, U8
codes and F8_E4M3 scales per 16 (`<module>_weight`, `_weight_scale`), each
expert's global scale its F32 `_weight_amax` over the
`_weight_outer_scale_denominator` (2688, `hf_quant_config.json`'s
`amax_denominator`), as llama.cpp's converter divides them into its GGUFs'
`.scale`, and MXFP8 in layers 20-23, F8_E4M3 values with a U8 E8M0 exponent per
32; their gate and up rows are interleaved, a row each, which the planner
takes as two windows of each row pair (oMLX's `w13` and mlx-vlm's
`gate_up_proj` keep them so). A conversion keeps those codes: nvfp4 experts with
MLX's `global_scale` (the release's amax) or mlx-lm's `weight_scale_2` (its
quotient), mxfp8 ones as they are ([MLX targets](#mlx-targets)). The release's
attention projections and head are BF16 and load as Q8_0, as an MLX target's
unquantized projections do, its token table as its bf16 rows, its routers as
their BF16 rows. With those projections on bf16 activations, as
`--activations bf16` runs them, its logits, teacher-forced over 1,933
positions of five prompts, are within a mean KL of 0.0024 nats of those of its
BF16 projections (UD-Q4_K_M's are as near), pick the same top-1 at 98.7% of
positions, and keep each prompt's NLL of the greedy continuation within 0.004
(0.1838 pooled, BF16's 0.1847). On M5 and later Macs, by default, a prefill
chunk of more than 64 rows runs those Q8_0 projections on int8 activations,
which adds its own error ([int8-activation
prefill](internals.md#int8-activation-prefill)). A full layer's static
convolution taps (`static_w`) and the relative bias
(`rel_logits_proj.proj`) are stored transposed, the taps last first, and the
release keeps its loop layers' message queries in two lists of every loop
layer's (`middle_cycle_cross_pass_queries`, its q and FFN rows, and
`_kv_queries`, its k and v rows), its block mix queries in lists of every
layer's (`attn_res.attn_queries`, `mlp_queries`). A loop layer's message
gates load as stored, F32 or BF16. The fused kernels take a call's gates (its
q, k and v gates, or its FFN gate) in one format, so a call that mixes the
two has its BF16 rows widened to F32, exactly: mlx-vlm's conversions keep a
gate's down projection BF16 and its forget and gain F32 (28 MB more for
`mlx-community/Inkling-2-Mini-4bit`). A checkpoint's MTP head is not loaded;
its vision tower is ([Vision](#vision)). Where a checkpoint and a GGUF hold
the same values, the images are the same bytes (`gguf-preparation`).

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
  IQ4_NL, MXFP4, NVFP4, PQ2_0, Q2_0, BF16 or F16; a tensor of a format of
  256-value blocks (the K-quants, BF16, F16) whose rows are not whole blocks
  of it is refused by name, but routed experts of BF16 or F16 values, which
  load as MXFP8 or Q8_0 (below);
- token embeddings: any of the linears' formats;
- norms: F32, BF16 or F16, loaded as the F32 values they hold;
- the MoE router: F32, BF16 or any of the linears' quantized formats (not
  F16), which the float projection runs as stored, a quantized one as its
  native rows (as an MLX target's quantized router loads);
- the shared-expert scalar gate: F32, BF16 or Q8_0, loaded as the F32 values
  it holds;
- the projections a family runs in fp32 (Qwen3.8-Flash-Next's QSA indexer
  projections, Inkling 2's message gates and sink projections): F32 or BF16
  as stored, or any of the linears' formats, loaded as the F32 values it
  holds (AtomicChat's Q8_0 and Q4_K/Q5_0 gates, the sink projections of
  lmstudio-community's and ggml-org's files in their queries' format); a
  call of Inkling 2's message gates that mixes BF16 with another format, as
  kaitchup's smallest quants keep some gates BF16 beside Q8_0 ones, has its
  BF16 gates widened to F32, exactly, as a checkpoint's are
  ([Inkling 2's checkpoints](#inkling-2s-checkpoints));
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

Inkling 2's GGUFs (`inkling2_loop`) state their head counts once per layer,
which every layer must share. Their derived configuration holds the text model
under `text_config`, as Inkling 2's config.json does, with the keys of that
`text_config` the header states (`gguf.inkling2_text_config`): every size, the
vocabulary padded to the token table's rows (`vocab_size`) and its tokens
(`unpadded_vocab_size`, which must be the tokenizer's), the relative bias
(`d_rel` from `attention.relative_length`, `rel_extent`), the full layers'
log scaling, the clips, `local_layer_ids` from
`attention.sliding_window_pattern` with the window and heads of those layers,
`sconv_kernel_size` from `shortconv.l_cache`, `n_layers_per_attn_res_block`
from `attn_res.block_size`, the logits' and the experts' softcaps
(`mlp_gate_softcap` and `mlp_up_softcap` from `activation.situ_beta` and
`situ_linear_beta`), the experts each token takes, their width and
`route_scale` from `expert_weights_scale`, and `loop_config` from the `loop`
keys; besides those, the sizes every family's configuration states
(`max_position_embeddings`, `num_experts`) and `eos_token_id`, which the
release states beside `text_config`. A float is the shortest decimal of the
header's float32 (`1e-06`), as the release writes it.
Unsloth's own layout of the same tensors (`inkling2`, the root files of
`unsloth/Inkling-2-Mini-GGUF`, which `--model
unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M` selects) states its layer counts, head
counts and sliding layers per layer application (48: the loop's layers once a
pass), its loop by `loop.core_start`, `loop.core_end` and `num_loops`, with no
`loop.varying`; it names a loop layer's message gates by the input they gate
(`loop_attn_`, `loop_k_`, `loop_v_` and `loop_mlp_` for llama.cpp's
`loop_q_ctrl_` to `loop_ffn_ctrl_`) and keeps every loop layer's message
queries in rows of two tensors of them all (`loop_res_query`, each layer's q
and FFN rows; `loop_kv_res_query`, its k and v rows). Its configuration is
derived by those keys (`num_hidden_layers` its weight layers, each layer's
values its first application's), and the engine plans it as the converter's,
into the same images (`Inkling2LoopGguf.cpp`). All ten of the repository's
quants load in either layout, from UD-IQ1_S to UD-Q8_K_XL (its BF16 experts
as MXFP8, below). The repository keeps the
converter's layout in `Mainline_Support/` with the extension `.gguf_file`,
which target selection does not read: such a file installs from a local copy
named `.gguf` ([Local model files](#local-model-files)). An installation of
one of the repository's quants holds its draft,
`incoai/Inkling-2-Mini-DFlash2` ([Drafts](#drafts)), and its vision tower, the
repository's `inkling2v` projector ([Vision](#vision)); a local copy takes a
projector with `--mmproj`. Inkling 2's decode captures a reply's state at
each page as the Qwen families' does, its convolutions' state folded at the
page ([Replay points and
checkpoints](internals.md#replay-points-and-checkpoints)), so a conversation's
next turn whose prompt renders the reply back resumes at the reply's last page
instead of prefilling the reply again.

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

Routed experts of BF16 or F16 values load as MXFP8 where every 32 of their
values are exactly an MXFP8 block's (an E8M0 power of two times E4M3 values),
as a release's MXFP8 experts widened hold them: Inkling 2's last four layers'
experts in Unsloth's UD-Q8_K_XL, BF16 of the release's MXFP8, whose down
projections of 1152 inputs BF16's blocks of 256 cannot hold. Lossless, at half
the bytes; each block takes the smallest exponent that holds its largest
magnitude, 2^-127 at least (`gguf::mxfp8Block`). Whether a tensor's values are
MXFP8's needs them, which a start reads once before it sizes and plans the
target (`exactlyMxfp8`, every value of each 16-bit expert tensor, on a thread
per core). On an M5 Pro (48 GB), UD-Q8_K_XL's 8.15 GB of them took 1.05 s
beside the weights' 3.88 s load from a fresh copy of the file, which no cache
held, and 0.43 s beside 1.93 s from the page cache. Experts of other values
load as stored where their rows are whole 256-value blocks, else as the Q8_0
the reference quantization makes of them, as Flash-Next's hyper-connection
projections do. The engine's model check, of headers alone, plans those
fallbacks. A Mac that streams the routed experts ([Streamed
experts](server.md#streamed-experts)) reads such an expert's 16-bit values at
every load and builds its MXFP8 blocks again: twice the SSD bytes of the MXFP8
it holds.

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
`.scale` tensor of one value for a projection, which llama.cpp multiplies the
products by; a `.scale` beside a projection of another type is refused, as
only NVFP4's rows hold one. An F32 `.input_scale` beside an NVFP4 weight
(NVIDIA's activation scale, as many values as `.scale`) plays no part, as in
Model Optimizer's targets; beside any other weight, or alone, it is refused.
It loads as Model Optimizer's NVFP4 does ([Model Optimizer
targets](#model-optimizer-targets)): four blocks make one 256-element NVFP4
block, whose meta unit holds a projection's `.scale` value as its `g` (1
without one), at 4.625 bits per weight for the file's 4.5. A tensor whose rows
are not whole 256-element blocks (llama.cpp writes any multiple of 64, an
expert down projection of 1152 inputs among them) loads in NVFP4's form of
64-element blocks instead (`nvfp464`): each block's four E4M3 scales and `g`
make its meta unit, which spans two 32-input groups, at 5 bits per weight;
every tile that takes the formats' list takes it.

An experts tensor (`_exps`) of any type may carry an F32 `.scale` of one value
per expert, which llama.cpp multiplies each expert's products by: NVFP4
releases have one, and Unsloth keeps it beside the experts it quantized from
NVFP4 to other types (IQ1_S to IQ3_XXS, Q3_K, Q2_0, MXFP4). It loads into the
layer's expert scales (`ops::BlockMoeWeights::expertScales`, F32 [3][experts]
after the routed experts in the layer's image, the fused `ffn_gate_up_exps`'s
for both gate and up, 1 without one), which the expert passes multiply their
fp32 sums by before rounding them: gate's and up's before the activation,
down's before the routing weight. An NVFP4 expert's `g` stays 1.

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
release ships its mask token's embedding. Installation downloads only those
files and follows the repository's default branch as it follows the target's
([Revisions](server.md#revisions)); `--draft-revision` names a branch, tag or
commit of that repository; and `--draft-model` accepts another repository,
followed the same way with its own `--draft-revision`, or a local directory
that holds them. A checkpoint is installed only when the engine accepts its
configuration with the target's (its `model-check`), so a draft of another
architecture never replaces one that loads.

A draft's `config.json` states its layers, intermediate size, rotary base and
the target layers whose rows it reads (`dflash_config.target_layer_ids`,
ascending, at most eight); its hidden size, vocabulary and mask token are its
target's (Flint's vocabulary keeps no mask token: a Flint draft chooses one of
its reserved special tokens, 150019 to 150143, as its
`dflash_config.mask_token_id`, which the target's layout then holds,
`QwenTargetDimensions::firstDraftMaskToken`), and its attention (32 query and
8 KV heads of 128), selector (rank 256, 16 candidates) and 2048-token window are
those the kernels are built for, and its block at least their 8 rows, of which a
cycle drafts the first (the Qwen3.5-122B-A10B draft was trained to fill 16;
`readDraftLayout` in `ModelDescriptor.mm`); the kernels take any hidden width of
whole 16-channel convolution groups. How a draft reads its target is its
target's family's, as its drafts' producer captured it
(`QwenTargetDimensions`): at each id the rows that target layer writes, or, for
a family whose drafts read layer inputs (`draftReadsLayerInputs`), the rows
passed into the layer, as the layer before wrote them, and at the id past its
layers the final norm's rows the head reads
(`dflash_config.capture_final_norm`); and its own rows, its anchor's and its
mask's, are the token table's rows as the target's first layer reads them,
through the table's norm where the family's input takes one (`draftEmbeddingNorm`,
`embedding_normalization` `folded_rmsnorm`). Inkling 2's drafts read layer
inputs, as its producer captures its block residual's partial sum before each
layer application, with no shift by one: its draft's ids 5, 23 and 41 read the
rows passed into those applications, the outputs of applications 4, 22 and 40
(not the mixes of the block residual their norms read), and 48 its final
norm's; their own rows take its table's norm. Its logits take the finish of its
target's head (`QwenTargetDimensions::logitSoftcap` and `vocabularyTokens`,
`DFlashDraft`): no padding row past the head's tokens is a candidate
(`dflash_config.valid_vocab_size`), and the head's softcap
(`final_logit_softcapping`) applies to each candidate's score after the
selector's edge (`selector_softcap_mode` `after_correction`, the one placement
the selector runs), before the probabilities its tree and a sampled verify
read; a config states the softcap where its target's head has one (Inkling
2's: 30). Three values a config may leave to its target, as
`incoai/Inkling-2-Mini-DFlash2`'s release leaves them to its serving adapter:
whether the draft reads the final norm (`capture_final_norm`, which its ids
say by naming the id past the target's layers), how its own rows are made
(`embedding_normalization`) and the head's tokens (`valid_vocab_size`,
Inkling 2's: 200,058 of 201,024 rows). One it states must be the one its
target makes, and one it leaves out is that one. A draft trained on others
would only be accepted less: the target verifies every token it proposes.
A draft's rows after its anchor read the
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
target's expert tile), priced by a step of the class's last rows, up to
sixteen rows a lane. Nothing the step weighs comes from the cycle's own
draft: a row verified for its own draws, or for its descendants' estimates,
which its draws condition, would leave a sampled lane's tokens off the
target's distribution (`VerifyRowsParams`). `inkling2-target` checks it on
greedy and sampled lanes: a cycle whose draft draws its tree anew takes the
same step and its lane verifies the same rows. The start measures the table
on the Mac before it serves (in the decode warmups, three passes over a step
of each class at each lane count, each step's median wall time: a first
start's decode warmups of
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
step of eight rows a lane, which the start also measures. Such a target
verifies every row of its step that its lanes' trees fill: a dense target's
projections read their weights once for all of a step's rows, so a row costs
it little, and a MoE target's step, whose rows each route their own experts,
which a rejected row reads for nothing (a Qwen3.8-Flash-Next cycle of one
lane routes about 43 experts a layer with all eight rows and 17 with two),
is priced by the same costs of its steps' rows. A target whose experts stream
from the SSD ([Streamed experts](server.md#streamed-experts)) measures no
table, as what its rows cost follows its store's state, which only serving
sees: its cycles fit 1.6.2's online cost line, a + b R ms for the R rows a
cycle's lanes verify at each lane count (`model::CostLine`), and verify a
row exactly when the estimate that the target accepts it is at least the
line's floor, b times the tokens per ms of the lane count's latest cycles:
the price at which a row's chance of a token is worth its time. A cycle
applies the floor twice. Before it runs, the host applies it to the lanes'
chances, which pick only the kernels that run: the step is the last rows of
the kernel class that holds the rows the floor passes, no fewer than a
draft's block in all and at most one a lane and the store's round of 32
rows. After its draft, the device applies it to the trees' own
estimates, which pick the rows: the lanes verify the step's rows at or above
the floor, and the others belong to no lane and route no expert
(`VerifyRowsParams`). Before the line's first sample the floor is its seed's,
and every 32nd cycle of a lane count verifies a draft's block a lane within
the round, which keeps the line's rows spread. `--verify-rows all` has every
cycle's step verify every row of a draft's block a lane, so that greedy and
seeded outputs repeat. The startup log says which rows the cycles
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
([Sampling](api.md#sampling)); sampled outputs keep their exact distribution. Under `--verify-rows all`, every
cycle verifies a draft's block a lane whatever output its lanes have left, which caps only what they
retain, so that an answer under a lower output limit starts the one under a higher. The speculation counts ([Output limits and
streaming](api.md#output-limits-and-streaming)) count the proposals a cycle verified.
The target writes the rows of the layers the draft reads, as its family names
them, for each row it reads; Qwen3.8-Flash-Next's, whose layers carry four
hyper-connection streams, as the streams' mean, summed in fp32 and stored as
bf16, as the draft was trained on it. `dflash_config.target_hidden_reduction` states that reduction
(`stream_mean` for a hyper-connected target, `none` for another); a config
without it takes its target's, and one that names another is refused. Native
loading validates the configuration against the target again and loads the draft
like a target ([Weight loading](internals.md#weight-loading)): `DraftCheckpointLoader`
(`DraftCheckpoint.cpp`) plans the draft's `MDGG0001` images, one per layer and
one for the rest, and `GgufPreparation` quantizes each projection into af4g64
planes, 4 bits in groups of 64 as MLX's affine quantization rounds them, and
copies every other tensor as stored. The draft runs on the block kernels its
target's projections run on.

The draft's logits are its target's vocabulary head's, which a cycle runs twice,
for the draft's proposals and for the target's verify; the draft's pass reads
the head's rows of the tokens it may propose (`DFlashDraft`), a padded head's
rows past its tokens too, which its logits leave out (Inkling 2's 966: its
200,058 tokens end inside a 64-row tile of the head, and the last tile's 58
tokens hold its chat format's message tags, which its drafts propose). The Qwen
families' vocabulary of 248,320 holds English and code first, Chinese from
95,726 to 148,274, then its other scripts, and its chat, tool and thinking
tokens from 248,044: a step's draft proposes from its first 148,288 tokens and
its last 320 (`kQwenDraftVocabulary`, 59.8% of the head's rows) while each of
its lanes' last 64 tokens, its anchor the last, lie there, and from the whole
vocabulary otherwise (`DFlashDraft::vocabulary`). The draft proposes no token
outside the subset (a lane's prompt-lookup chain alone may), so text that
leaves it, another script or a word of one, drafts over the whole vocabulary
until 64 tokens have passed inside it. The
target verifies over its whole head, so a greedy request's output stays the
target's and a sampled one keeps its exact distribution; only the proposals,
and with them the tokens a cycle accepts, can change: 99.98% of the tokens
Qwen3.8-27B and Qwen3.6-35B-A3B committed decoding agents, code and rewrites
greedily lie in the subset. On 1.6.2, whose cycles held at most eight rows a
lane, on an M5 Max the draft's head pass took 1.13 ms of a one-lane
Qwen3.8-27B MLX 4-bit cycle instead of 1.72 ms (a dispatch of the leading rows
and one of the last 320), and decode steps took less GPU time with the same
acceptance (`backend_regression`, ABBA): 1.5% at one lane and 0.6% at four on
that model, 1.5% and 1.0% on its UD-Q4_K_M GGUF, 2.3% and 1.7% on
Qwen3.6-35B-A3B UD-Q4_K_M. Flint's vocabulary mixes its scripts through all
its tokens (its first 98,304 hold 98.1% of English text's tokens and 88.5% of
Russian text's), so its drafts propose from all of them.

## Vision

Vision comes from the target repository: MLX's `vision_tower.*` tensors (or
transformers' `model.visual.*`, as Model Optimizer keeps them, whose patch
embedding is already in the image's order), linking
only `config.json` and the shards holding them, or the GGUF repository's root
projector, a GGUF whose name holds `mmproj` (as `mmproj-BF16.gguf` or
`MODEL-mmproj-BF16.gguf`), chosen by its header: a `clip` projector whose
weights are a type the family's tower takes, in its order of preference
(`projectors` in `install/families.py`): BF16, F32 or F16 for the Qwen
families. An F16 projector made
from a BF16 tower, llama.cpp's default output, loads exactly: F16 rounds the
smallest BF16 weights onto its subnormal grid, and what that keeps is still a
BF16. The processor configuration (MLX `preprocessor_config.json`, or the
`image_processor` object of `processor_config.json`, where newer Transformers
releases save it; the GGUF's `clip.vision` metadata) must describe the
preprocessing Splash implements for the family's tower (`server/images.py`);
it is checked before any weight download and not installed. The models of a
family without a vision tower (Flint; `vision` in `install/families.py`) serve
text only: their installation reads no projector, with or without
`--language-only`.

Inkling 2's tower loads from each of its sources: its release's checkpoint
(`model.visual.*`), an MLX conversion's (transformers' `model.vision_tower.*`,
unquantized, as `mlx-community/Inkling-2-Mini-4bit` keeps it), either under
`vision_tower.` in `model.visual.`'s place (oMLX's oQ conversions, as
`mlx-community/Inkling-2-Mini-oQ4e-mtp` keeps the release's names, and
thoddnn's nvfp4 conversion transformers' names beside the release's position
embedding and projector) and, beside its GGUFs, an `inkling2v` projector
(`clip.vision.projector_type`), which Unsloth
publishes as BF16, F16 and F32 and ggml-org as BF16 and Q8_0; the installer
prefers BF16, then F32 and F16, and takes Q8_0 last. A checkpoint's patch
embedding orders a patch's values by row, column and channel, which
preparation reorders to the image's channels first, so each source writes the
same image, byte for byte, from the same BF16 values. A Q8_0 matrix is the one
tensor preparation rounds: each value, its block's F16 scale times its int8
quant, becomes the nearest BF16, ties to even, the precision the tower's
kernels read every weight in (`model/InklingVision.hpp` lays out its image).
Its processor is transformers' `Inkling2ImageProcessor`, whose one
configuration is its class's defaults (a token of 40 x 40 pixels, mean and
deviation 0.5): an MLX conversion, which saves none, is installed with them.
Its `vision_config` may state its patches' side and merge as the release does
(`native_patch_size`, `adapter_merge_size`) or as transformers saves them
(`patch_size`, `spatial_merge_size`).

Inkling 2's images are not resized but padded right and bottom to whole tokens
of 40 x 40 pixels, four 20-pixel patches merged 2 x 2: an image of H x W pixels
takes ceil(H / 40) x ceil(W / 40) tokens (a 640 x 480 chart 192, a 1000 x 620
screenshot 400). A transparent image is composited onto its processor's 8-pixel
checkerboard of greys 204 and 153, after any resize, so its squares stay 8
pixels. The reference pads with -1 on the 0-255 scale, which no byte holds: a
request's padding is zero, and its image span gives the image's own height and
width, past which the encoder takes it as -1. Only an image past the patch
budget the image cap gives (`--max-image-pixels` / 256 patches, 4,096 of its
tokens at the default, llama.cpp's budget too) is resized: its sides scaled
alike, a pixel each at least, to the largest size whose padded tokens fit the
budget. One `<|unused_200054|>` per token stands for it in the prompt, between
`<|content_image|>` and `<|end_message|>`, and its rows take the projector's
embeddings as they are, without the norm the token embeddings get. Its encoder
(`ops::InklingVision`) appends a class token to the patches, runs 23 blocks of
16 heads of 64 with an axial 2D RoPE and LayerScale on the shared dense
operators Qwen's tower runs, and merges the patch rows 2 x 2 into a projector
without biases. Measured against an fp32 tower of the same BF16 weights
(`dev/tools/vision_reference.py`, which matched transformers 5.24's
`Inkling2VisionModel` to within 1e-5), its embeddings were within 0.7% to 1.7%
(relative error), the worst token within 2.0%, at grids of 2 x 2, 2 x 128,
128 x 2, 6 x 124, 20 x 200 and 60 x 60 patches, padded or not; `vision-encoder`
holds a 6 x 8 grid to within 3%.

A tower's sources are all written into one image, which its BF16 encoder
(`ops::ImageEncoder`) reads: BF16 tensors are
copied, and F32 or F16 tensors are converted under the exact-BF16 rule of
[weight loading](internals.md#weight-loading).
Unsloth's mmproj stores its 1-D tensors, patch embedding and position table as
F32, all of them BF16-exact, and loads byte-identical to the MLX tower.
Quantized MLX towers, deepstack projectors and mmproj tensors the tower does not
use are rejected. The installer has the engine prepare a tower once its files
are downloaded, as every start prepares it (the engine's `vision-check`, under
a second and the image's memory): a tower it refuses, of names it does not
bind or of a weight that is not exactly a BF16, which headers do not show, is
refused at installation, naming `--language-only`, not at every start.
Inkling 2, which 1.7.2 served as text only, serves text instead, as from a
source without a tower Splash serves (a GGUF repository without an `inkling2v`
projector, a checkpoint without its tower): the installation records why
(`vision_unavailable`, for the Hub's listing of its commit), which later starts
do not ask again until the commit moves. An Inkling 2 installation that 1.7.2
made, text only, takes its tower in at its next start that reaches the Hub (a
GGUF repository's projector downloads); offline, or where its tower cannot
download, it goes on serving text and is updated otherwise as before.

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
pattern) without a normalizer, `gpt2/inkling2` o200k's pattern alone without a
normalizer, its BPE taking a piece its vocabulary holds whole as that token
before any merge (`ignore_merges`), each before byte-level BPE, as the
release's own tokenizer.json has them. Unknown profiles, malformed metadata
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
- Inkling 2's (`INKLING2`): the template appends no generation prompt, and the
  model opens each of its messages itself, `<|message_model|>`, a content kind
  and its text, closed by `<|end_message|>`: reasoning in
  `<|content_thinking|>` blocks, which may come again after text, the answer in
  `<|content_text|>` blocks (a `<|content_tool_error|>` or
  `<|content_invoke_tool_text|>` block is the answer's too), every tag a
  special token; calls are a JSON object in the model's message named for the
  function (`INKLING2_JSON`),
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>`;
  `<|content_model_end_sampling|>` stops.

A format holds the model's vocabulary and the engine's stop tokens, which are
its target layout's; the tags that open and close its reasoning (`<think>` and
`</think>` for Qwen and Flint), with their tokens where the model opens its
reasoning itself, and those that frame its answer's text where it has any,
each frame and its close the format's visible tags alone, which grammars
write as their tokens (a format whose frames hold other text is refused where
it is defined);
the call syntax, with the tags a call opens and closes with and the token of
its opening, its arguments either tags around each argument or one JSON
object, after a header that names the function, whose keys for the name and
the arguments the syntax states (`"name"` and `"args"` for Inkling 2; a syntax
that lays a call out otherwise is refused where it is defined); whether the
model opens its own message; the special tokens decoding
keeps in the text it reads (`visible_tags`: Inkling 2's tags, every other
special token, its stop token among them, decoding to nothing); and the
special token after which prompt tokenization reuses a cached prompt's tokens,
`<|im_end|>`, `<|assistant|>` or `<|end_message|>`. Startup checks that the
tokenizer encodes each as the format has it, and that it can keep its added
tokens out of a request's text ([tokenizer and chat
templates](#tokenizer-and-chat-templates)). Grammars write reasoning and calls
as the format lays
them out: Flint's optional think block may come before any constrained output,
and its tags, which the grammar compiler takes as tokens rather than text, are
written as their tokens. A model that opens its reasoning itself closes it
before a call, so a `<tool_call>` inside its think block is reasoning, as
Flint's own parser reads it. A call of JSON arguments streams them as the
model writes them, after the object's name, and its grammar writes them to
the tool's schema as JSON; where the format frames the answer's text, a
grammar's text and JSON answer are written in that frame.

Inkling 2's template renders as shipped, and tells the model the request's
effort in a system message before the first message that is not one
(`Thinking effort level: 0.9` unless the request names one; none and
`enable_thinking: false` render 0): `/v1/models` lists its efforts, and
`--reasoning-effort` sets every request's. A thinking budget caps only
reasoning the generation prompt opens, so it caps nothing here, as for Flint.
Reasoning the model writes after text stays where it is in Messages and
Responses, and joins the rest in Chat Completions' `reasoning_content`.

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

`--mmproj` supplies a local BF16, F32 or F16 GGUF vision projector (or Q8_0,
for Inkling 2). Without
one, a GGUF serves text only. An MLX target uses its own vision tower unless
`--language-only` is set. `--revision` and `--draft-revision` cannot be
combined with `--model-dir`; `--draft-revision` cannot be combined with a
local `--draft-model` directory; and `--mmproj` requires `--model-dir` and
cannot be combined with `--language-only`. Files are linked, and each start
prepares them again under the same validation, publication and lifetime rules
described below.

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
model's source options (`--revision`, `--draft-model`, `--draft-revision`,
`--language-only`, `--offline`) are `prepare_local_model`'s arguments instead,
or have no use.

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
loading](#upstream-model-loading)) and its `vision-check` on a vision tower's
weights ([Vision](#vision)). When Splash cannot serve them without a
download (a missing file or split part, an unsupported format, a draft of
another model), a `ModelError` names the file and the reason. The directory
returned is an assembly of links to the files ([Installation](internals.md#installation)),
which never changes. Preparing the same files again returns it, reading only
their metadata, sizes and times and a vision tower's weights, which the engine
prepares at every preparation, in under a second; the first preparation reads
each file outside the Hugging Face cache once, for its digest. Once a file
changes, or a download moves the draft's default branch, preparing again
returns another directory: restart the server on it. The server holds the
directory it serves until it exits; one that no preparation returns any more
is removed only once no server holds it. The server takes the installation
lock of `models_dir` to hold it when it starts, so it needs write access
there.

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
