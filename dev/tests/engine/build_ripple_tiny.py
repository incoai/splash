"""A tiny Ripple model's GGUF and draft checkpoint, at dimensions the
engine's pipeline loads end-to-end: 2 layers of the family's 4+1 pattern at
hidden 256, 8 query heads of 64 over 2 KV heads, a dense FFN of 512, and a
1-layer draft of the same hidden, whose convolutions carry the family's four
taps. The model root this script writes loads through the engine's
inspectModelRoot and loadModel with a descriptor built directly from the
layout (the family's own size check holds the release's dimensions, which a
test model does not carry)."""

import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# The tiny model's dimensions.
LAYERS = 2
HIDDEN = 256
VOCAB = 256
QUERY_HEADS = 80
KV_HEADS = 8
HEAD_DIM = 128
ATTENTION_WIDTH = QUERY_HEADS * HEAD_DIM  # 10240
KV_WIDTH = KV_HEADS * HEAD_DIM  # 1024
INTERMEDIATE = 512
WINDOW = 1024
EMBED_SCALE = 16.0
CLAMP = 6.0
# The 4+1 pattern: layer 0 slides, layer 1 is full (NoPE).
SLIDING_PATTERN = [True, False]
# The draft: 1 layer, reading target layer 0.
DRAFT_LAYERS = 1
DRAFT_HIDDEN = HIDDEN
DRAFT_INTERMEDIATE = 512
DRAFT_VOCAB = VOCAB
DRAFT_QUERY_HEADS = 32
DRAFT_KV_HEADS = 8
DRAFT_HEAD_DIM = 128
DRAFT_ATTENTION_WIDTH = DRAFT_QUERY_HEADS * DRAFT_HEAD_DIM
DRAFT_KV_WIDTH = DRAFT_KV_HEADS * DRAFT_HEAD_DIM
DRAFT_WINDOW = 2048
DRAFT_CONV_TAPS = 2
DRAFT_CONV_GROUP = 16
DRAFT_DYNAMIC = 2 * DRAFT_CONV_TAPS * (DRAFT_HIDDEN // DRAFT_CONV_GROUP)
DRAFT_RANK = 256
DRAFT_CANDIDATES = 16
TARGET_LAYERS = [0]
MASK_TOKEN = 200
EMBED_SCALE_F32 = struct.Struct("<f")


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def q8_0(values):
    """Q8_0's native blocks: per 32-value group, an fp16 scale and 32 int8
    codes."""
    blocks = b""
    for start in range(0, len(values), 32):
        group = values[start : start + 32]
        scale = max(abs(v) for v in group) / 127.0 or 1e-7
        half = struct.pack("<e", scale)
        codes = bytes(
            min(127, max(-128, round(v / scale))) & 0xFF for v in group
        )
        blocks += half + codes
    return blocks


def build_gguf(path):
    """The target GGUF: the ricursive metadata and the real tensor names at
    the tiny model's dimensions, Q8_0 weights."""
    tensors = []

    def tensor(name, dims, data, kind=8):
        tensors.append((name, dims, kind, data))

    def q(name, rows, cols):
        values = [((i * 37 + len(tensors) * 11) % 509 - 254) / 254 for i in range(rows * cols)]
        tensor(name, [cols, rows], q8_0(values))

    def norm(name, width):
        values = [0.5 + (i % 7) / 14 for i in range(width)]
        tensor(name, [width], struct.pack(f"<{width}f", *values), kind=0)  # F32

    def scalar(name, value):
        tensor(name, [1], struct.pack("<f", value), kind=0)  # F32

    q("token_embd.weight", HIDDEN, VOCAB)
    norm("output_norm.weight", HIDDEN)
    q("output.weight", VOCAB, HIDDEN)
    for layer in range(LAYERS):
        p = f"blk.{layer}."
        norm(p + "attn_norm.weight", HIDDEN)
        q(p + "attn_q.weight", ATTENTION_WIDTH, HIDDEN)
        q(p + "attn_gate.weight", ATTENTION_WIDTH, HIDDEN)
        q(p + "attn_k.weight", KV_WIDTH, HIDDEN)
        q(p + "attn_v.weight", KV_WIDTH, HIDDEN)
        norm(p + "attn_q_norm.weight", HEAD_DIM)
        norm(p + "attn_k_norm.weight", HEAD_DIM)
        q(p + "attn_output.weight", HIDDEN, ATTENTION_WIDTH)
        scalar(p + "attn_vres_lambda.weight", 0.5 + layer * 0.25)
        norm(p + "post_attention_norm.weight", HIDDEN)
        q(p + "ffn_gate.weight", INTERMEDIATE, HIDDEN)
        q(p + "ffn_up.weight", INTERMEDIATE, HIDDEN)
        q(p + "ffn_down.weight", HIDDEN, INTERMEDIATE)
        norm(p + "ffn_norm.weight", HIDDEN)
        norm(p + "post_ffw_norm.weight", HIDDEN)

    metadata = {
        "general.architecture": "ricursive",
        "general.name": "ripple-tiny",
        "ricursive.block_count": LAYERS,
        "ricursive.context_length": 4096,
        "ricursive.embedding_length": HIDDEN,
        "ricursive.feed_forward_length": INTERMEDIATE,
        "ricursive.attention.head_count": QUERY_HEADS,
        "ricursive.attention.head_count_kv": KV_HEADS,
        "ricursive.attention.key_length": HEAD_DIM,
        "ricursive.attention.value_length": HEAD_DIM,
        "ricursive.attention.sliding_window": WINDOW,
        "ricursive.attention.sliding_window_pattern": SLIDING_PATTERN,
        "ricursive.rope.freq_base": 10000.0,
        "ricursive.rope.dimension_count": HEAD_DIM,
        "ricursive.attention.layer_norm_rms_epsilon": 1e-6,
        "ricursive.vocab_size": VOCAB,
        "ricursive.embedding_scale": EMBED_SCALE,
        "ricursive.swiglu_clamp": CLAMP,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "dbrx",
        "tokenizer.ggml.tokens": [f"[tok{i}]" for i in range(VOCAB)],
        "tokenizer.ggml.token_type": [1] * VOCAB,
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    }
    from dev.tests.fixture_files import write_gguf
    write_gguf(path, metadata, tensors)


def build_draft_checkpoint(path):
    """The draft's BF16 safetensors: 1 layer at the tiny model's hidden,
    with the family's 4-tap convolutions and the selector's codebooks."""
    tensors = {}

    def add(name, shape):
        count = math.prod(shape)
        values = [((i * 37 + len(tensors) * 11) % 509 - 254) / 254 for i in range(count)]
        tensors[name] = (shape, "BF16", struct.pack(f"<{count}H", *[bf16(v) for v in values]))

    for layer in range(DRAFT_LAYERS):
        p = f"layers.{layer}."
        a = p + "self_attn."
        add(p + "input_layernorm.weight", [DRAFT_HIDDEN])
        add(p + "attention_conv.base_kernel", [2, DRAFT_CONV_TAPS, DRAFT_HIDDEN])
        add(p + "attention_conv.kernel_projection.weight", [DRAFT_DYNAMIC, DRAFT_HIDDEN])
        add(a + "q_proj.weight", [DRAFT_ATTENTION_WIDTH, DRAFT_HIDDEN])
        add(a + "k_proj.weight", [DRAFT_KV_WIDTH, DRAFT_HIDDEN])
        add(a + "v_proj.weight", [DRAFT_KV_WIDTH, DRAFT_HIDDEN])
        add(a + "q_norm.weight", [DRAFT_HEAD_DIM])
        add(a + "k_norm.weight", [DRAFT_HEAD_DIM])
        add(a + "o_proj.weight", [DRAFT_HIDDEN, DRAFT_ATTENTION_WIDTH])
        add(p + "post_attention_layernorm.weight", [DRAFT_HIDDEN])
        add(p + "mlp_conv.base_kernel", [2, DRAFT_CONV_TAPS, DRAFT_HIDDEN])
        add(p + "mlp_conv.kernel_projection.weight", [DRAFT_DYNAMIC, DRAFT_HIDDEN])
        add(p + "mlp.gate_proj.weight", [DRAFT_INTERMEDIATE, DRAFT_HIDDEN])
        add(p + "mlp.up_proj.weight", [DRAFT_INTERMEDIATE, DRAFT_HIDDEN])
        add(p + "mlp.down_proj.weight", [DRAFT_HIDDEN, DRAFT_INTERMEDIATE])
    add("fc.weight", [DRAFT_HIDDEN, DRAFT_HIDDEN * len(TARGET_LAYERS)])
    add("hidden_norm.weight", [DRAFT_HIDDEN])
    add("norm.weight", [DRAFT_HIDDEN])
    add("candidate_selector.hidden_projection.weight", [DRAFT_RANK, DRAFT_HIDDEN])
    add("candidate_selector.predecessor_codebook", [DRAFT_VOCAB, DRAFT_RANK])
    add("candidate_selector.successor_codebook", [DRAFT_VOCAB, DRAFT_RANK])

    from dev.tests.fixture_files import write_safetensors
    write_safetensors(path, tensors)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "dev/tests/fixtures/ripple-tiny")
    root.mkdir(parents=True, exist_ok=True)
    (root / "draft").mkdir(exist_ok=True)
    (root / "target").mkdir(exist_ok=True)

    build_gguf(root / "target" / "ripple.gguf")
    build_draft_checkpoint(root / "draft" / "model.safetensors")

    # The model root's config and metadata, at the tiny model's dims.
    config = {
        "model_type": "ricursive",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB,
        "max_position_embeddings": 4096,
        "num_attention_heads": QUERY_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTERMEDIATE,
        "sliding_window": WINDOW,
        "embed_scale": EMBED_SCALE,
        "ffn_clamp_threshold": CLAMP,
        "rope_theta": 10000.0,
        "layer_types": ["sliding_attention" if s else "full_attention" for s in SLIDING_PATTERN],
        "bos_token_id": 0,
        "pad_token_id": 2,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "use_valres": True,
        "use_gated_attention": True,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2))
    draft_config = {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": DRAFT_HIDDEN,
        "vocab_size": DRAFT_VOCAB,
        "num_attention_heads": DRAFT_QUERY_HEADS,
        "num_key_value_heads": DRAFT_KV_HEADS,
        "head_dim": DRAFT_HEAD_DIM,
        "sliding_window": DRAFT_WINDOW,
        "rms_norm_eps": 1e-6,
        "num_hidden_layers": DRAFT_LAYERS,
        "intermediate_size": DRAFT_INTERMEDIATE,
        "hidden_act": "silu",
        "is_causal": False,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "dflash_config": {
            "block_size": 16,
            "conv_group_size": DRAFT_CONV_GROUP,
            "conv_kernel_size": DRAFT_CONV_TAPS,
            "selector_rank": DRAFT_RANK,
            "selector_top_k": DRAFT_CANDIDATES,
            "mask_token_id": MASK_TOKEN,
            "target_layer_ids": TARGET_LAYERS,
        },
    }
    (root / "draft" / "config.json").write_text(json.dumps(draft_config, indent=2))
    (root / "model.json").write_text(json.dumps({
        "version": 1, "model": "test/ripple-tiny",
        "target_format": "gguf", "vision_format": "none",
    }))
    print(f"ripple-tiny model at {root}")


if __name__ == "__main__":
    main()
