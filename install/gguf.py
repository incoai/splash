"""Read GGUF metadata without mapping or decoding any weight tensors.

The supported tokenizer profile describes algorithms; vocabulary, merge ranks,
special-token IDs and the chat template always come from the selected GGUF.

This file is the metadata adapter, identified by the SHA-256 of its bytes:
assembly._metadata_key keys each derived metadata entry by that hash, the
tokenizers version and the size and digest of each source GGUF. After any
edit to this file, a comment included, preparing an installed GGUF model
reports that the adapter changed and derives its metadata again.
"""

from __future__ import annotations

import collections
import contextlib
import struct
from pathlib import Path

if __package__:
    from . import models
    from .models import ModelError
else:
    import models
    from models import ModelError

# The files derived_files derives from a target GGUF, by assembly path.
DERIVED_FILES = (
    "config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/chat_template.jinja",
)


class Metadata:
    """Bounded little-endian GGUF v3 metadata reader."""

    # The version, and the bound on each of the tensor and key counts, that
    # the native reader (runtime/model/GgufFile.cpp) accepts.
    VERSION = 3
    MAX_FIELDS = 16384
    MAX_BYTES = 128 * 1024 * 1024
    # The bound on one array's elements: a vocabulary or its merges.
    MAX_ITEMS = 1_000_000
    # GGML_MAX_DIMS, the highest tensor rank the native reader
    # (runtime/model/GgufFile.cpp) accepts.
    MAX_DIMENSIONS = 4
    # GGUF value types (gguf_type): a string, an array of one other type, and
    # the struct format of each scalar type.
    STRING = 8
    ARRAY = 9
    SCALAR_FORMATS = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }

    def __init__(self, source, *, tensors=False):
        """Read a local path, or a binary stream at the start of the file such
        as a Hub range reader, which its caller closes. tensors also reads
        each tensor's type."""
        self.values = {}
        self.tensors = {}
        self.consumed = 0
        with (
            Path(source).open("rb")
            if isinstance(source, (str, Path))
            else contextlib.nullcontext(source)
        ) as self.stream:
            if self.read(4) != b"GGUF" or self.scalar("I") != self.VERSION:
                raise ModelError("unsupported GGUF header (expected little-endian v3)")
            tensor_count = self.scalar("Q")
            key_count = self.scalar("Q")
            if key_count > self.MAX_FIELDS or tensor_count > self.MAX_FIELDS:
                raise ModelError("GGUF metadata has too many fields")
            for _ in range(key_count):
                key = self.string()
                if key in self.values:
                    raise ModelError("duplicate GGUF metadata key: " + key)
                self.values[key] = self.value(self.scalar("I"))
            for _ in range(tensor_count if tensors else 0):
                name = self.string()
                dimensions = self.scalar("I")
                if dimensions > self.MAX_DIMENSIONS:
                    raise ModelError("invalid GGUF tensor rank: " + name)
                self.read(dimensions * struct.calcsize("<Q"))  # Its extents.
                kind = self.scalar("I")
                self.scalar("Q")  # Data offset; the payload is never read here.
                if name in self.tensors:
                    raise ModelError("duplicate GGUF tensor: " + name)
                self.tensors[name] = kind

    def read(self, size):
        if size > self.MAX_BYTES - self.consumed:
            raise ModelError("GGUF metadata exceeds the size limit")
        data = self.stream.read(size)
        if len(data) != size:
            raise ModelError("truncated GGUF metadata")
        self.consumed += size
        return data

    def scalar(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        try:
            return self.read(self.scalar("Q")).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ModelError("invalid UTF-8 in GGUF metadata") from error

    def value(self, value_type):
        if value_type == self.STRING:
            return self.string()
        if value_type == self.ARRAY:
            element, count = self.scalar("I"), self.scalar("Q")
            if (
                element not in (*self.SCALAR_FORMATS, self.STRING)
                or count > self.MAX_ITEMS
            ):
                raise ModelError("unsupported or oversized GGUF metadata array")
            return [self.value(element) for _ in range(count)]
        if value_type not in self.SCALAR_FORMATS:
            raise ModelError("unknown GGUF metadata type")
        return self.scalar(self.SCALAR_FORMATS[value_type])

    def require(self, key, kind):
        value = self.values.get(key)
        if type(value) is not kind:
            raise ModelError("missing or invalid GGUF metadata: " + key)
        return value

    def positive(self, key):
        value = self.require(key, int)
        if value <= 0:
            raise ModelError("GGUF metadata must be positive: " + key)
        return value


# The qwen35 profile extends Qwen2's letter class to include combining marks.
# Its NFC normalizer and byte-level BPE match the original Qwen tokenizer.
QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# The GGUF token types (llama_token_type) a byte-level BPE vocabulary uses:
# control tokens are special added tokens, user-defined ones added tokens
# that are not special, and unused ones fill the vocabulary to its size.
NORMAL_TOKEN = 1
CONTROL_TOKEN = 3
USER_DEFINED_TOKEN = 4
UNUSED_TOKEN = 5


def derived_files(target, vision=None):
    """DERIVED_FILES' contents, derived from a target GGUF's metadata and its
    vision projector's (local paths)."""
    metadata = Metadata(target)
    config = model_config(metadata, None if vision is None else Metadata(vision))
    return {"config.json": models.json_bytes(config), **tokenizer_files(metadata)}


def tokenizer_files(metadata):
    from tokenizers import (
        AddedToken,
        Regex,
        Tokenizer,
        decoders,
        normalizers,
        pre_tokenizers,
        processors,
    )
    from tokenizers import models as token_models

    if (
        metadata.require("tokenizer.ggml.model", str) != "gpt2"
        or metadata.require("tokenizer.ggml.pre", str) != "qwen35"
    ):
        raise ModelError("unsupported GGUF tokenizer profile; expected gpt2/qwen35")
    tokens = metadata.require("tokenizer.ggml.tokens", list)
    types = metadata.require("tokenizer.ggml.token_type", list)
    merges = metadata.require("tokenizer.ggml.merges", list)
    if (
        not tokens
        or not all(isinstance(t, str) and t for t in tokens)
        or len(set(tokens)) != len(tokens)
        or len(types) != len(tokens)
        or any(
            type(t) is not int
            or t not in (NORMAL_TOKEN, CONTROL_TOKEN, USER_DEFINED_TOKEN, UNUSED_TOKEN)
            for t in types
        )
    ):
        raise ModelError("invalid or unsupported GGUF tokenizer vocabulary")
    if not set(pre_tokenizers.ByteLevel.alphabet()) <= set(tokens):
        raise ModelError("GGUF byte-level tokenizer is missing byte tokens")
    pairs = []
    for merge in merges:
        if not isinstance(merge, str) or len(parts := merge.split(" ")) != 2:
            raise ModelError("invalid GGUF BPE merge")
        pairs.append(tuple(parts))
    for key in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"):
        if metadata.values.get(key, False) is not False:
            raise ModelError("GGUF automatic BOS/EOS insertion is unsupported: " + key)
    try:
        backend = Tokenizer(
            token_models.BPE({t: i for i, t in enumerate(tokens)}, pairs)
        )
    except Exception as error:
        raise ModelError("invalid GGUF BPE vocabulary or merges") from error
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN35_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False, trim_offsets=False
            ),
        ]
    )
    backend.post_processor = processors.ByteLevel(
        add_prefix_space=False, use_regex=False, trim_offsets=False
    )
    backend.decoder = decoders.ByteLevel()
    # The index of each added token, and whether it is special.
    added = {
        index: kind == CONTROL_TOKEN
        for index, kind in enumerate(types)
        if kind in (CONTROL_TOKEN, USER_DEFINED_TOKEN)
    }
    backend.add_tokens(
        [
            AddedToken(tokens[index], normalized=False, special=is_special)
            for index, is_special in added.items()
        ]
    )

    def special_token(name, *, optional=False):
        key = "tokenizer.ggml." + name + "_token_id"
        if optional and key not in metadata.values:
            return None
        index = metadata.require(key, int)
        if not 0 <= index < len(tokens) or types[index] != CONTROL_TOKEN:
            raise ModelError("invalid GGUF special token: " + key)
        return tokens[index]

    template = metadata.require("tokenizer.chat_template", str)
    if not template.strip():
        raise ModelError("GGUF chat template is empty")
    config = {
        "tokenizer_class": "TokenizersBackend",
        "model_max_length": metadata.positive(
            text_architecture(metadata) + ".context_length"
        ),
        "clean_up_tokenization_spaces": False,
        "bos_token": special_token("bos", optional=True),
        "eos_token": special_token("eos"),
        "pad_token": special_token("padding", optional=True),
        "added_tokens_decoder": {
            str(index): {
                "content": tokens[index],
                "special": is_special,
                "normalized": False,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
            }
            for index, is_special in added.items()
        },
    }
    return {
        "tokenizer/tokenizer.json": backend.to_str().encode(),
        "tokenizer/tokenizer_config.json": models.json_bytes(config),
        "tokenizer/chat_template.jinja": template.encode(),
    }


# The GGUF text architectures Splash serves, and the model type of each one's
# text configuration.
TEXT_MODEL_TYPES = {"qwen35": "qwen3_5_text", "qwen35moe": "qwen3_5_moe_text"}


def text_architecture(metadata):
    arch = metadata.require("general.architecture", str)
    if arch not in TEXT_MODEL_TYPES:
        raise ModelError("unsupported GGUF model architecture: " + arch)
    return arch


def model_config(metadata, vision=None):
    arch = text_architecture(metadata)
    fields = {
        "hidden_size": "embedding_length",
        "max_position_embeddings": "context_length",
        "num_attention_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv",
        "head_dim": "attention.key_length",
    }
    text = {name: metadata.positive(arch + "." + key) for name, key in fields.items()}
    text.update(
        num_hidden_layers=loaded_layers(metadata, arch),
        model_type=TEXT_MODEL_TYPES[arch],
        vocab_size=len(metadata.require("tokenizer.ggml.tokens", list)),
    )
    if arch == "qwen35moe":
        text.update(
            num_experts=metadata.positive(arch + ".expert_count"),
            num_experts_per_tok=metadata.positive(arch + ".expert_used_count"),
        )
    config = {
        "model_type": TEXT_MODEL_TYPES[arch].removesuffix("_text"),
        "text_config": text,
    }
    if vision is not None:
        config["vision_config"] = vision_config(vision)
    return config


def loaded_layers(metadata, arch):
    """The target's layer count without its MTP layers, which are never
    loaded."""
    layers = metadata.positive(arch + ".block_count")
    mtp = metadata.values.get(arch + ".nextn_predict_layers", 0)
    if type(mtp) is not int or not 0 <= mtp < layers:
        raise ModelError("invalid GGUF MTP layer count")
    return layers - mtp


def vision_config(vision):
    if (
        vision.require("general.architecture", str) != "clip"
        or vision.require("clip.projector_type", str) != "qwen3vl_merger"
        or vision.require("clip.use_gelu", bool) is not True
    ):
        raise ModelError("unsupported GGUF vision architecture")
    fields = {
        "depth": "block_count",
        "hidden_size": "embedding_length",
        "num_heads": "attention.head_count",
        "intermediate_size": "feed_forward_length",
        "out_hidden_size": "projection_dim",
        "patch_size": "patch_size",
        "spatial_merge_size": "spatial_merge_size",
    }
    result = {
        name: vision.positive("clip.vision." + key) for name, key in fields.items()
    }
    image_size = vision.positive("clip.vision.image_size")
    if image_size % result["patch_size"]:
        raise ModelError("invalid GGUF vision position grid")
    if any(vision.require("clip.vision.is_deepstack_layers", list)):
        raise ModelError("GGUF vision deepstack layers are unsupported")
    # qwen3vl_merger uses RGB patches with two temporal slices; the native loader
    # independently checks both patch tensors and every other weight shape.
    result.update(
        temporal_patch_size=2,
        in_channels=3,
        num_position_embeddings=(image_size // result["patch_size"]) ** 2,
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=[],
    )
    return result


def processor_config(vision):
    config = vision_config(vision)
    return {
        "patch_size": config["patch_size"],
        "temporal_patch_size": config["temporal_patch_size"],
        "merge_size": config["spatial_merge_size"],
        "image_mean": vision.require("clip.vision.image_mean", list),
        "image_std": vision.require("clip.vision.image_std", list),
    }


# GGML tensor type names, and the quantized formats the native loader reads
# (runtime/metal/abi/QuantFormat.h, checked by the tests).
TENSOR_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    29: "IQ1_M",
    30: "BF16",
    39: "MXFP4",
}
QUANTIZED_TYPES = {
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "Q4_0",
    "Q4_1",
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ4_NL",
    "IQ4_XS",
    "MXFP4",
}
# The token rows the embedding kernels gather (gguf_embedding_format in
# runtime/metal/abi/Gguf.h).
EMBEDDING_TYPES = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "Q4_1"}
# The tensors the native loader reads from a target, and the types it accepts
# for each (runtime/model/GgufImage.cpp): quantized projections; F32 norms,
# small GDN vectors and MoE routers, which llama.cpp keeps unquantized and
# which run unrounded; GDN alpha and beta both Q8_0 or both F32.
F32 = {"F32"}
MODEL_TENSORS = {
    "token_embd.weight": EMBEDDING_TYPES,
    "output_norm.weight": F32,
    "output.weight": QUANTIZED_TYPES,
}
LAYER_TENSORS = {"attn_norm.weight": F32, "post_attention_norm.weight": F32}
ATTENTION_TENSORS = {
    "attn_q.weight": QUANTIZED_TYPES,
    "attn_k.weight": QUANTIZED_TYPES,
    "attn_v.weight": QUANTIZED_TYPES,
    "attn_q_norm.weight": F32,
    "attn_k_norm.weight": F32,
    "attn_output.weight": QUANTIZED_TYPES,
}
GDN_TENSORS = {
    "attn_qkv.weight": QUANTIZED_TYPES,
    "attn_gate.weight": QUANTIZED_TYPES,
    "ssm_alpha.weight": {"Q8_0", "F32"},
    "ssm_beta.weight": {"Q8_0", "F32"},
    "ssm_conv1d.weight": F32,
    "ssm_a": F32,
    "ssm_dt.bias": F32,
    "ssm_norm.weight": F32,
    "ssm_out.weight": QUANTIZED_TYPES,
}
DENSE_TENSORS = {
    "ffn_gate.weight": QUANTIZED_TYPES,
    "ffn_up.weight": QUANTIZED_TYPES,
    "ffn_down.weight": QUANTIZED_TYPES,
}
MOE_TENSORS = {
    "ffn_gate_inp.weight": F32,
    "ffn_gate_exps.weight": QUANTIZED_TYPES,
    "ffn_up_exps.weight": QUANTIZED_TYPES,
    "ffn_down_exps.weight": QUANTIZED_TYPES,
    "ffn_gate_inp_shexp.weight": F32,
    "ffn_gate_shexp.weight": QUANTIZED_TYPES,
    "ffn_up_shexp.weight": QUANTIZED_TYPES,
    "ffn_down_shexp.weight": QUANTIZED_TYPES,
}


def loaded_tensors(metadata):
    """Each tensor the native loader reads from the target that metadata
    describes, with the types it accepts. MTP layers are not loaded; every
    full_attention_interval-th layer is full attention, the others GDN."""
    arch = text_architecture(metadata)
    period = metadata.positive(arch + ".full_attention_interval")
    ffn = MOE_TENSORS if arch == "qwen35moe" else DENSE_TENSORS
    tensors = dict(MODEL_TENSORS)
    for layer in range(loaded_layers(metadata, arch)):
        mixer = ATTENTION_TENSORS if (layer + 1) % period == 0 else GDN_TENSORS
        for name, types in (LAYER_TENSORS | mixer | ffn).items():
            tensors[f"blk.{layer}.{name}"] = types
    return tensors


def require_loadable(metadata):
    """Reject a target the native loader cannot read, from its header alone,
    so an unusable file is never downloaded: every tensor it reads must be
    present, with a type it accepts for that tensor."""
    unsupported = collections.Counter()
    for name, types in loaded_tensors(metadata).items():
        kind = metadata.tensors.get(name)
        found = "missing" if kind is None else TENSOR_TYPES.get(kind, f"type {kind}")
        if found not in types:
            unsupported[
                name.split(".", 2)[-1] if name.startswith("blk.") else name, found
            ] += 1
        # The two GDN input gates run as one segment of their shared format.
        if name.endswith(".ssm_beta.weight") and kind != metadata.tensors.get(
            name.replace("ssm_beta", "ssm_alpha")
        ):
            unsupported[
                "ssm_alpha.weight and ssm_beta.weight", "of different types"
            ] += 1
    if unsupported:
        listed = ", ".join(
            f"{tensor} {found} ({count} {'tensor' if count == 1 else 'tensors'})"
            for (tensor, found), count in sorted(unsupported.items())
        )
        raise ModelError(
            f"this GGUF stores tensors Splash cannot load: {listed}; choose another variant"
        )
