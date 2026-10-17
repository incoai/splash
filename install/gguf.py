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
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import models
from .models import ModelError

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
        each tensor's type, its extents (innermost first, as GGUF lists them)
        and its offset in the file's data section."""
        self.values = {}
        self.tensors = {}
        self.extents = {}
        self.offsets = {}
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
                extents = struct.unpack(
                    f"<{dimensions}Q", self.read(dimensions * struct.calcsize("<Q"))
                )
                kind = self.scalar("I")
                offset = self.scalar("Q")  # The payload is never read here.
                if name in self.tensors:
                    raise ModelError("duplicate GGUF tensor: " + name)
                self.tensors[name] = kind
                self.extents[name] = extents
                self.offsets[name] = offset

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


def text_architecture(metadata):
    arch = metadata.require("general.architecture", str)
    if arch not in ARCHITECTURES:
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
        model_type=ARCHITECTURES[arch].model_type,
        vocab_size=len(metadata.require("tokenizer.ggml.tokens", list)),
    )
    config = {
        "model_type": ARCHITECTURES[arch].model_type.removesuffix("_text"),
        "text_config": text,
    }
    if vision is not None:
        config["vision_config"] = vision_config(vision)
    return config


def scalar_metadata(metadata):
    """The header's scalar values as the native reader keeps them, for the
    engine's model check, which holds them to the rules its GGUF planner
    holds the file to at every start (GgufMetadata in
    runtime/model/GgufFile.hpp): integers and booleans unsigned, a negative
    one as its 64-bit two's complement, floats and strings, each kind apart,
    as JSON alone would not keep them. JSON has no NaN or infinity, which no
    rule accepts."""
    typed = {"unsigned": {}, "float": {}, "string": {}}
    for key, value in metadata.values.items():
        if isinstance(value, int):
            typed["unsigned"][key] = value % (1 << 64)
        elif isinstance(value, float) and math.isfinite(value):
            typed["float"][key] = value
        elif isinstance(value, str):
            typed["string"][key] = value
    return typed


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


# GGML tensor type names, the quantized types the native loader reads, and the
# types of its formats: those, BF16 and F16 (runtime/metal/abi/QuantFormat.h,
# checked by the tests).
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
    40: "NVFP4",
    42: "Q2_0",
    142: "PQ2_0",
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
    "Q5_0",
    "Q5_1",
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
    "NVFP4",
    "Q2_0",
    "PQ2_0",
}
FORMAT_TYPES = QUANTIZED_TYPES | {"BF16", "F16"}
# Splash loads quantized GGUFs: BF16 and F16 are formats for the tensors a
# quantized file keeps unquantized, and require_loadable refuses a target whose
# projections and experts are all of these types.
UNQUANTIZED_TYPES = {"BF16", "F16", "F32"}
# The token rows the embedding kernels gather: native rows of every format
# (gguf_embedding_format in runtime/metal/abi/Gguf.h).
EMBEDDING_TYPES = FORMAT_TYPES
# The tensors the native loader reads from a target, and the types it accepts
# for each (runtime/model/GgufImage.cpp): projections of a format; the tensors
# llama.cpp keeps unquantized, which run unrounded: small GDN vectors of F32,
# norms (and Qwen3.8-Flash-Next's PLE convolution) of F32 or the BF16 and F16
# that widen to it (floatNorm), float projections (MoE routers, the QSA
# indexer) of F32 or BF16, which run as stored (floatTensor), and
# shared-expert scalar gates of F32, BF16 or Q8_0, which preparation turns
# into the F32 values they hold (floatRows); GDN alpha and beta, each of a
# format or F32, which preparation widens to the F32 values they hold unless
# both are of one quantized type or both BF16.
F32 = {"F32"}
NORM_TYPES = {"F32", "BF16", "F16"}
FLOAT_PROJECTION_TYPES = {"F32", "BF16"}
FLOAT_ROW_TYPES = {"F32", "BF16", "Q8_0"}
MODEL_TENSORS = {
    "token_embd.weight": EMBEDDING_TYPES,
    "output_norm.weight": NORM_TYPES,
    "output.weight": FORMAT_TYPES,
}
LAYER_TENSORS = {
    "attn_norm.weight": NORM_TYPES,
    "post_attention_norm.weight": NORM_TYPES,
}
ATTENTION_TENSORS = {
    "attn_q.weight": FORMAT_TYPES,
    "attn_k.weight": FORMAT_TYPES,
    "attn_v.weight": FORMAT_TYPES,
    "attn_q_norm.weight": NORM_TYPES,
    "attn_k_norm.weight": NORM_TYPES,
    "attn_output.weight": FORMAT_TYPES,
}
GDN_TENSORS = {
    "attn_qkv.weight": FORMAT_TYPES,
    "attn_gate.weight": FORMAT_TYPES,
    "ssm_alpha.weight": FORMAT_TYPES | F32,
    "ssm_beta.weight": FORMAT_TYPES | F32,
    "ssm_conv1d.weight": F32,
    "ssm_a": F32,
    "ssm_dt.bias": F32,
    "ssm_norm.weight": NORM_TYPES,
    "ssm_out.weight": FORMAT_TYPES,
}
DENSE_TENSORS = {
    "ffn_gate.weight": FORMAT_TYPES,
    "ffn_up.weight": FORMAT_TYPES,
    "ffn_down.weight": FORMAT_TYPES,
}
MOE_TENSORS = {
    "ffn_gate_inp.weight": FLOAT_PROJECTION_TYPES,
    "ffn_gate_exps.weight": FORMAT_TYPES,
    "ffn_up_exps.weight": FORMAT_TYPES,
    "ffn_down_exps.weight": FORMAT_TYPES,
    "ffn_gate_inp_shexp.weight": FLOAT_ROW_TYPES,
    "ffn_gate_shexp.weight": FORMAT_TYPES,
    "ffn_up_shexp.weight": FORMAT_TYPES,
    "ffn_down_shexp.weight": FORMAT_TYPES,
}
# A MoE layer whose gate and up experts are llama.cpp's fused tensor
# (create_tensor_gate_up_exps): each expert's gate rows, then its up rows.
FUSED_MOE_TENSORS = {
    name: types
    for name, types in MOE_TENSORS.items()
    if name not in ("ffn_gate_exps.weight", "ffn_up_exps.weight")
} | {"ffn_gate_up_exps.weight": FORMAT_TYPES}


# Qwen3.8-Flash-Next (qwen4exp): hyper-connections replace every norm, so a
# layer has none of its own and the model no output norm; its full-attention
# layers carry the QSA indexer, whose projections are float tensors; its PLE
# layer carries the n-gram embedding's projections, norms and convolution,
# and the model the embedding's table.
QWEN4_MODEL_TENSORS = {
    "token_embd.weight": EMBEDDING_TYPES,
    "per_layer_token_embd.weight": EMBEDDING_TYPES,
    "output.weight": FORMAT_TYPES,
    "output_hc_norm.weight": NORM_TYPES,
    "output_hc_down.weight": FORMAT_TYPES,
    "output_hc_up.weight": FORMAT_TYPES,
}
QWEN4_LAYER_TENSORS = {
    f"hc_{part}_{name}.weight": types
    for part in ("attn", "ffn")
    for name, types in (
        ("norm", NORM_TYPES),
        ("down", FORMAT_TYPES),
        ("up", FORMAT_TYPES),
        ("inject", FLOAT_ROW_TYPES),
    )
}
QWEN4_INDEXER_TENSORS = {
    "indexer.q_proj.weight": FLOAT_PROJECTION_TYPES,
    "indexer.k_proj.weight": FLOAT_PROJECTION_TYPES,
    "indexer.q_norm.weight": NORM_TYPES,
    "indexer.k_norm.weight": NORM_TYPES,
}
QWEN4_PLE_TENSORS = {
    "ple_key.weight": FORMAT_TYPES,
    "ple_value.weight": FORMAT_TYPES,
    "ple_norm_key.weight": NORM_TYPES,
    "ple_norm_query.weight": NORM_TYPES,
    "ple_norm_conv.weight": NORM_TYPES,
    "ple_conv1d.weight": NORM_TYPES,
}


@dataclass(frozen=True)
class Architecture:
    """A GGUF text architecture Splash serves, as the native loader reads it
    (runtime/model/GgufImage.cpp planImages)."""

    # The model type of its text configuration.
    model_type: str
    # The model's tensors, and each layer's beside its mixer's and FFN's.
    model_tensors: dict
    layer_tensors: dict
    # Whether its FFN is a sparse MoE, else dense.
    moe: bool
    # A full-attention layer's tensors beside ATTENTION_TENSORS.
    attention_tensors: dict = field(default_factory=dict)
    # The tokens of a QSA block: a full-attention layer's entry of
    # attention.compress_ratios, where a GDN layer's is 0. None without QSA.
    qsa_block: int | None = None
    # The one layer its PLE n-gram embedding feeds, which ple.layers names
    # alone; it reads QWEN4_PLE_TENSORS. None without one.
    ple_layer: int | None = None


# The GGUF text architectures Splash serves, by general.architecture; the
# QSA block and PLE layer of qwen4exp are Qwen4ExpLayout's
# indexerBlockTokens and pleLayer (runtime/model/Qwen4Exp.hpp).
ARCHITECTURES = {
    "qwen35": Architecture("qwen3_5_text", MODEL_TENSORS, LAYER_TENSORS, moe=False),
    "qwen35moe": Architecture(
        "qwen3_5_moe_text", MODEL_TENSORS, LAYER_TENSORS, moe=True
    ),
    "qwen4exp": Architecture(
        "qwen4_exp_text",
        QWEN4_MODEL_TENSORS,
        QWEN4_LAYER_TENSORS,
        moe=True,
        attention_tensors=QWEN4_INDEXER_TENSORS,
        qsa_block=4,
        ple_layer=1,
    ),
}


def loaded_tensors(metadata):
    """Each tensor the native loader reads from the target that metadata
    describes, with the types it accepts. MTP layers are not loaded; every
    full_attention_interval-th layer is full attention, the others GDN. A MoE
    layer whose header lists the fused gate/up experts tensor reads it in place
    of the two. A PLE layer reads the n-gram embedding's tensors too."""
    arch = text_architecture(metadata)
    architecture = ARCHITECTURES[arch]
    period = metadata.positive(arch + ".full_attention_interval")
    tensors = dict(architecture.model_tensors)
    for layer in range(loaded_layers(metadata, arch)):
        if (layer + 1) % period == 0:
            mixer = ATTENTION_TENSORS | architecture.attention_tensors
        else:
            mixer = GDN_TENSORS
        if not architecture.moe:
            ffn = DENSE_TENSORS
        elif f"blk.{layer}.ffn_gate_up_exps.weight" in metadata.tensors:
            ffn = FUSED_MOE_TENSORS
        else:
            ffn = MOE_TENSORS
        parts = architecture.layer_tensors | mixer | ffn
        if layer == architecture.ple_layer:
            parts |= QWEN4_PLE_TENSORS
        for name, types in parts.items():
            tensors[f"blk.{layer}.{name}"] = types
    return tensors


def require_arrays(metadata):
    """Reject a target whose metadata arrays, which scalar_metadata leaves out
    of the engine's model check, are not those the native planner requires
    at every start (Qwen4ExpGguf::arrays in runtime/model/GgufImage.cpp): with
    QSA, attention.compress_ratios, from its first entry the QSA block for
    each loaded full-attention layer and 0 for each other; with a PLE layer,
    ple.layers, that layer alone. One error names every mismatch."""
    arch = text_architecture(metadata)
    architecture = ARCHITECTURES[arch]
    mismatched = []
    if architecture.qsa_block is not None:
        period = metadata.positive(arch + ".full_attention_interval")
        ratios = [
            architecture.qsa_block if (layer + 1) % period == 0 else 0
            for layer in range(loaded_layers(metadata, arch))
        ]
        found = metadata.values.get(arch + ".attention.compress_ratios")
        if type(found) is not list or found[: len(ratios)] != ratios:
            mismatched.append("attention.compress_ratios")
    ple = architecture.ple_layer
    if ple is not None and metadata.values.get(arch + ".ple.layers") != [ple]:
        mismatched.append("ple.layers")
    if mismatched:
        raise ModelError(
            "GGUF metadata does not match the target: " + ", ".join(mismatched)
        )


def merged_split(headers):
    """One target's metadata from the headers of its split files (llama.cpp's
    gguf-split), in split order: the first file's keys, whose split.* keys
    must describe these files, and every file's tensor types."""
    first = headers[0]
    count = first.values.get("split.count", 1)
    if type(count) is not int or count != len(headers):
        raise ModelError("the GGUF's split files do not match its split.count")
    tensors = {}
    for number, header in enumerate(headers):
        if len(headers) > 1 and (
            header.values.get("split.no") != number
            or header.values.get("split.count") != count
        ):
            raise ModelError(f"GGUF split file {number + 1} is out of order")
        for name, kind in header.tensors.items():
            if name in tensors:
                raise ModelError("duplicate GGUF tensor across split files: " + name)
            tensors[name] = kind
    total = first.values.get("split.tensors.count", len(tensors))
    if total != len(tensors):
        raise ModelError("the GGUF's split files hold another number of tensors")
    first.tensors = tensors
    return first


# Prism ML's input rotation of a dense target: the parameters and keys
# GgufFile::readRotation (runtime/model/GgufFile.cpp) accepts;
# gguf::planImages (runtime/model/GgufImage.cpp, requireRotation) refuses MoE
# targets and ungrouped GDN value heads and checks the names.
ROTATION_PREFIX = "prism.hadamard."
ROTATION = {
    "version": 1,
    "block_size": 1024,
    "transform": "normalized-sylvester-walsh-hadamard",
    "axis": "input-last-dimension",
    "sign_mode": "explicit",
    "gdn_v_grouped": True,
}
ROTATION_ARRAYS = {"weight_names", "inverse_weight_names", "sign_widths", "sign_values"}


def require_rotation(metadata):
    """Reject a rotated target (Prism ML's GGUFs) whose rotation the native
    loader does not run: a MoE target, other rotation parameters or a
    rotation key it does not know."""
    keys = {
        key[len(ROTATION_PREFIX) :]
        for key in metadata.values
        if key.startswith(ROTATION_PREFIX)
    }
    if not keys:
        return
    if (
        text_architecture(metadata) != "qwen35"
        or not keys <= ROTATION.keys() | ROTATION_ARRAYS
        or any(
            type(metadata.values.get(ROTATION_PREFIX + key)) is not type(value)
            or metadata.values.get(ROTATION_PREFIX + key) != value
            for key, value in ROTATION.items()
        )
    ):
        raise ModelError(
            "this GGUF's input rotation is not one Splash runs; choose another variant"
        )


def require_loadable(metadata):
    """Reject a target Splash does not load, from its header alone, so an
    unusable file is never downloaded: metadata arrays of another model
    (require_arrays), a whole-BF16, F16 or F32 model, whose projections and
    experts are all unquantized, or a file the native loader cannot read,
    missing a tensor it reads or storing one in a type it does not accept for
    that tensor."""
    require_rotation(metadata)
    require_arrays(metadata)
    loaded = loaded_tensors(metadata)
    # The projections and experts are the tensors read only in a format, but
    # the token table; one missing or of a type the loader does not know is
    # listed below instead.
    weights = {
        TENSOR_TYPES.get(metadata.tensors.get(name))
        for name, types in loaded.items()
        if types == FORMAT_TYPES and name != "token_embd.weight"
    }
    if weights <= UNQUANTIZED_TYPES:
        raise ModelError(
            "this GGUF stores every projection and expert unquantized (BF16, F16 "
            "or F32): Splash loads quantized GGUFs, not whole-BF16 or F16 models; "
            "choose a quantized variant of the repository"
        )
    unsupported = collections.Counter()
    for name, types in loaded.items():
        kind = metadata.tensors.get(name)
        found = "missing" if kind is None else TENSOR_TYPES.get(kind, f"type {kind}")
        short = name.split(".", 2)[-1] if name.startswith("blk.") else name
        if found not in types:
            unsupported[short, found] += 1
        # llama.cpp multiplies a weight's products by a .scale tensor beside
        # it; the loader folds one into NVFP4's rows only.
        stem = name.removesuffix(".weight")
        if stem != name and stem + ".scale" in metadata.tensors and found != "NVFP4":
            scale = short.removesuffix(".weight") + ".scale"
            unsupported[scale, f"beside {found}"] += 1
    if unsupported:
        listed = ", ".join(
            f"{tensor} {found} ({count} {'tensor' if count == 1 else 'tensors'})"
            for (tensor, found), count in sorted(unsupported.items())
        )
        raise ModelError(
            f"this GGUF stores tensors Splash cannot load: {listed}; choose another variant"
        )
