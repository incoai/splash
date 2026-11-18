"""A LiLiCorr draft released as a GGUF (Nemotron 3.5 Bolt's), converted
into the DFlash2 checkpoint the engine loads: config.json and BF16
model.safetensors, as a DFlash2 release holds them.

Its projections are NVFP4 or Q8_0, each with a per-tensor scale; the rest
F32 or BF16. Every weight is dequantized to float32, scaled, and rounded to
BF16 (the engine quantizes the draft as it loads it). Its two KV heads are
each repeated four times: the draft kernels take eight, and attention over
the repeats is the same. The LiLiCorr selector keeps its own names
(runtime/metal/abi/LiLiCorr.h). The conversion is deterministic, so a
reinstallation derives the same bytes (assembly.derived_draft)."""

import json
import struct

import numpy as np

from . import gguf
from .models import ModelError

# The engine's draft layout (runtime/metal/abi/DraftAttention.h): eight KV heads, and
# the selector's rank and candidates, which a LiLiCorr draft's selector
# replaces but its configuration still states.
KV_HEADS = 8
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16
# The released checkpoint's configuration (nvidia/NVIDIA-Nemotron-3.5-Bolt-
# 74B-A9B-24GB-LiLiCorr, config.json), which its GGUF does not state.
MASK_TOKEN_ID = 990
VOCABULARY = 131072
# The engine's draft context ring (ExecutionLimits::draftContextTokens): a
# draft trained over a wider window (Super's 8192) sees the ring's positions.
# Super's draft also states RMS epsilon 1e-5 and its own output head; it runs
# with the kernels' 1e-6 and the target's head. Its head is the target's,
# rounded to NVFP4 (correlation 0.995 over 256 sampled rows), and its norms'
# mean squares are 8 or more, where the epsilons' rsqrt differs by under 1e-6.
# ponytail: its window is 8192, and the ring holds 2048, a compile-time size
# of every draft kernel; acceptance at 33K tokens (0.50) matches short prompts
# (0.47-0.51), so a wider ring waits for a prompt where acceptance drops.
RING_WINDOW = 2048


def _ue4m3():
    """NVFP4's unsigned E4M3 block scales, halved as ggml halves them for its
    doubled E2M1 values (ggml_ue4m3_to_fp32)."""
    table = np.zeros(256, np.float32)
    for x in range(256):
        exponent, mantissa = (x >> 3) & 0xF, x & 7
        if x not in (0, 0x7F):
            value = (
                mantissa * 2.0**-9
                if exponent == 0
                else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
            )
            table[x] = value * 0.5
    return table


UE4M3 = _ue4m3()
E2M1 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)


def _dequantize(raw, kind, count):
    """count float32 values of ggml type kind from raw bytes."""
    name = gguf.TENSOR_TYPES.get(kind)
    if name == "F32":
        return raw.view(np.float32)[:count].copy()
    if name == "F16":
        return raw.view(np.float16)[:count].astype(np.float32)
    if name == "BF16":
        return (raw.view(np.uint16)[:count].astype(np.uint32) << 16).view(np.float32)
    if name == "Q8_0":
        blocks = raw.reshape(-1, 34)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        return (scales * blocks[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)
    if name == "NVFP4":
        # 64 values a block: four E4M3 scales, then 32 bytes, eight per
        # 16-value group, low nibbles its first half and high its second.
        blocks = raw.reshape(-1, 36)
        scales = UE4M3[blocks[:, :4]][:, :, None]
        packed = blocks[:, 4:].reshape(-1, 4, 8)
        nibbles = np.concatenate([packed & 0x0F, packed >> 4], axis=-1)
        return (scales * E2M1[nibbles]).reshape(-1)
    raise ModelError(f"unsupported draft tensor type {name or kind}")


def _repeat_heads(value, heads, times):
    rows = value.reshape(heads, -1, value.shape[-1])
    return np.repeat(rows, times, axis=0).reshape(-1, value.shape[-1])


def _tensors(layers, sinks=True):
    """(checkpoint name, GGUF tensor) of every weight, in the names Splash's
    DFlash2 loader and LiLiCorr selector read."""
    yield "fc.weight", "fc.weight"
    yield "hidden_norm.weight", "enc.output_norm.weight"
    yield "norm.weight", "output_norm.weight"
    for i in range(layers):
        b, p = f"blk.{i}.", f"layers.{i}."
        for name, source in (
            ("input_layernorm.weight", "attn_norm.weight"),
            ("attention_conv.base_kernel", "attn_conv_base"),
            ("attention_conv.kernel_projection.weight", "attn_conv_proj.weight"),
            ("self_attn.q_proj.weight", "attn_q.weight"),
            ("self_attn.k_proj.weight", "attn_k.weight"),
            ("self_attn.v_proj.weight", "attn_v.weight"),
            ("self_attn.q_norm.weight", "attn_q_norm.weight"),
            ("self_attn.k_norm.weight", "attn_k_norm.weight"),
            ("self_attn.o_proj.weight", "attn_output.weight"),
            ("post_attention_layernorm.weight", "ffn_norm.weight"),
            ("mlp_conv.base_kernel", "ffn_conv_base"),
            ("mlp_conv.kernel_projection.weight", "ffn_conv_proj.weight"),
            ("mlp.gate_proj.weight", "ffn_gate.weight"),
            ("mlp.up_proj.weight", "ffn_up.weight"),
            ("mlp.down_proj.weight", "ffn_down.weight"),
            ("self_attn.sinks", "attn_sinks.weight"),
        ):
            if sinks or name != "self_attn.sinks":
                yield p + name, b + source
    # The selector: its feature MLP is the block after its two layers.
    feature = f"blk.{layers + 2}."
    for name, source in (
        ("token_proj", "lilicorr.token_proj"),
        ("pass_proj", "lilicorr.pass_hidden_proj"),
        ("context_proj", "lilicorr.context_proj"),
        ("feature_up", feature + "ffn_up"),
        ("feature_down", feature + "ffn_down"),
        ("factor_proj", "lilicorr.factor_input_proj"),
        ("out_head", "lilicorr.out_head"),
        ("in_head", "lilicorr.in_head"),
        ("anchor_out_head", "lilicorr.anchor_out_head"),
    ):
        yield f"lilicorr.{name}.weight", source + ".weight"
        yield f"lilicorr.{name}.bias", source + ".bias"
    yield "lilicorr.feature_norm.weight", feature + "ffn_norm.weight"
    yield "lilicorr.feature_norm.bias", feature + "ffn_norm.bias"
    yield "lilicorr.slot_embedding", "lilicorr.slot_embedding.weight"
    yield "lilicorr.rank_embedding", "lilicorr.rank_embedding.weight"
    yield "lilicorr.attention_bias", "lilicorr.attention_bias.bias"
    yield "lilicorr.output_norm", "lilicorr.output_norm.weight"
    yield "lilicorr.anchor_norm", "lilicorr.anchor_norm.weight"
    for i in range(2):
        b, p = f"blk.{layers + i}.", f"lilicorr.layers.{i}."
        yield p + "attn_norm", b + "attn_norm.weight"
        yield p + "in_proj.weight", f"lilicorr.layers.{i}.attn.in_proj.weight"
        yield p + "in_proj.bias", f"lilicorr.layers.{i}.attn.in_proj.bias"
        yield p + "out_proj.weight", b + "attn_output.weight"
        yield p + "out_proj.bias", b + "attn_output.bias"
        yield p + "mlp_norm", b + "ffn_norm.weight"
        for name in ("up", "down"):
            yield p + f"{name}.weight", b + f"ffn_{name}.weight"
            yield p + f"{name}.bias", b + f"ffn_{name}.bias"


def is_lilicorr(header):
    """Whether a GGUF header is a LiLiCorr draft's."""
    values = header.values
    return (
        values.get("general.architecture") == "dflash"
        and "lilicorr.hidden_size" in values
    )


def config(header):
    """The DFlash2 config.json of the LiLiCorr draft GGUF header describes."""
    if not is_lilicorr(header):
        raise ModelError("it is not a LiLiCorr draft GGUF")
    values = header.values

    def value(key):
        if key not in values:
            raise ModelError(f"draft GGUF lacks {key}")
        return values[key]

    kv_heads = value("dflash.attention.head_count_kv")
    if not isinstance(kv_heads, int) or kv_heads <= 0 or KV_HEADS % kv_heads:
        raise ModelError(
            f"draft GGUF has {kv_heads} KV heads, not a divisor of {KV_HEADS}"
        )
    return {
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": False,
        "dflash_config": {
            "block_size": value("dflash.block_size"),
            "conv_group_size": value("dflash.conv_group_size"),
            "conv_kernel_size": value("dflash.conv_kernel_size"),
            "mask_token_id": MASK_TOKEN_ID,
            "selector_rank": SELECTOR_RANK,
            "selector_top_k": SELECTOR_TOP_K,
            # GGUF counts the embeddings as layer 0.
            "target_layer_ids": [i - 1 for i in value("dflash.target_layers")],
            "projector_type": "lilicorr",
            "lilicorr_candidate_topk": value("lilicorr.candidate_topk"),
            "lilicorr_hidden_size": value("lilicorr.hidden_size"),
            "lilicorr_num_heads": value("lilicorr.num_heads"),
            "lilicorr_num_layers": value("lilicorr.num_layers"),
        },
        "head_dim": value("dflash.attention.key_length"),
        "hidden_act": "silu",
        "hidden_size": value("dflash.embedding_length"),
        "intermediate_size": value("dflash.feed_forward_length"),
        "is_causal": False,
        "num_attention_heads": value("dflash.attention.head_count"),
        "num_hidden_layers": value("dflash.block_count"),
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "rope_theta": value("dflash.rope.freq_base"),
            "rope_type": "default",
        },
        "sliding_window": min(value("dflash.attention.sliding_window"), RING_WINDOW),
        # Bolt's draft adds a sink logit per query head; Super's has none.
        "attention_sink_bias": "blk.0.attn_sinks.weight" in header.tensors,
        "tie_word_embeddings": False,
        "vocab_size": VOCABULARY,
    }


def config_bytes(header):
    return json.dumps(config(header), indent=2, sort_keys=True).encode()


def _to_bf16(value):
    """Round to nearest even, as BF16 words."""
    bits = np.ascontiguousarray(value, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def write_checkpoint(path, stage):
    """Write the draft GGUF at path into stage as config.json and
    model.safetensors, one tensor at a time."""
    header = gguf.Metadata(path, tensors=True)
    (stage / "config.json").write_bytes(config_bytes(header))
    values = header.values
    layers = values["dflash.block_count"]
    kv_heads = values["dflash.attention.head_count_kv"]
    alignment = values.get("general.alignment", 32)
    data = (len(header.raw) + alignment - 1) // alignment * alignment
    sources = dict(_tensors(layers, "blk.0.attn_sinks.weight" in header.tensors))
    missing = sorted(set(sources.values()) - set(header.tensors))
    if missing:
        raise ModelError(f"draft GGUF lacks tensors: {', '.join(missing[:3])}")
    repeated = {
        f"layers.{i}.self_attn.{t}_proj.weight" for i in range(layers) for t in "kv"
    }

    def shape(name):
        dims = list(header.extents[sources[name]][::-1])
        if name in repeated:
            dims[0] *= KV_HEADS // kv_heads
        return dims

    def load(name):
        source = sources[name]
        dims = header.extents[source]
        count = int(np.prod(dims))
        kind = header.tensors[source]
        raw = np.fromfile(
            path,
            np.uint8,
            count=_bytes(kind, count),
            offset=data + header.offsets[source],
        )
        value = _dequantize(raw, kind, count).reshape(dims[::-1])
        scale = source.removesuffix(".weight") + ".scale"
        if scale in header.tensors:
            factor = np.fromfile(
                path, np.float32, count=1, offset=data + header.offsets[scale]
            )
            value = value * factor[0]
        if name in repeated:
            value = _repeat_heads(value, kv_heads, KV_HEADS // kv_heads)
        return value

    names = sorted(sources)
    table, offset = {}, 0
    for name in names:
        dims = shape(name)
        size = 2 * int(np.prod(dims))
        table[name] = {
            "dtype": "BF16",
            "shape": dims,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(table).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(stage / "model.safetensors", "wb") as file:
        file.write(struct.pack("<Q", len(encoded)) + encoded)
        for name in names:
            words = _to_bf16(load(name))
            if list(words.shape) != table[name]["shape"]:
                raise ModelError(f"draft tensor {name} has an unexpected shape")
            file.write(words.tobytes())


def _bytes(kind, count):
    """The bytes count values of ggml type kind take."""
    per_block = {"Q8_0": (32, 34), "NVFP4": (64, 36)}
    name = gguf.TENSOR_TYPES.get(kind)
    if name in per_block:
        values, size = per_block[name]
        if count % values:
            raise ModelError(
                f"draft tensor of {count} values is not whole {name} blocks"
            )
        return count // values * size
    return count * {"F32": 4, "F16": 2, "BF16": 2}.get(name, 0)
