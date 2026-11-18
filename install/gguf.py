"""Read GGUF metadata without mapping or decoding any weight tensors.

The supported tokenizer profiles describe algorithms; vocabulary, merge
ranks, special-token IDs and the chat template always come from the selected
GGUF.

This file is the metadata adapter, identified by the SHA-256 of its bytes:
assembly._metadata_key keys each derived metadata entry by that hash, the
tokenizers version and the size and digest of each source GGUF. After any
edit to this file, a comment included, preparing an installed GGUF model
reports that the adapter changed and derives its metadata again.
"""

from __future__ import annotations

import contextlib
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from server.chat_format import FLINT

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
        and its offset in the file's data section. raw keeps the bytes read:
        with tensors, the whole header, which the engine's model check plans
        the target from (upstream.gguf_target)."""
        self.values = {}
        self.tensors = {}
        self.extents = {}
        self.offsets = {}
        self.raw = bytearray()
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
        if size > self.MAX_BYTES - len(self.raw):
            raise ModelError("GGUF metadata exceeds the size limit")
        data = self.stream.read(size)
        if len(data) != size:
            raise ModelError("truncated GGUF metadata")
        self.raw += data
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
# The flint profile, as Flint's own tokenizer.json splits text before its
# byte-level BPE, without a normalizer: digits in runs of up to three, runs
# of Han, hiragana and katakana, then o200k's pattern.
FLINT_PATTERNS = (
    r"\p{N}{1,3}",
    "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)
# Tekken (Mistral's, and Nemotron-H's): no normalizer, and BPE that takes a
# word already in the vocabulary whole (ignore_merges), as the model's
# tokenizer.json states. Its GGUFs name it pixtral or tekken (llama.cpp
# #30894: Nemotron GGUFs converted before it say pixtral).
TEKKEN_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# Each GGUF tokenizer profile (tokenizer.ggml.pre) Splash derives a tokenizer
# for: the patterns that split text, each isolating its matches, in order,
# whether text is NFC-normalized before them, and whether its BPE takes
# vocabulary words whole (ignore_merges).
TOKENIZER_PROFILES = {
    "qwen35": ((QWEN35_PATTERN,), True, False),
    "flint": (FLINT_PATTERNS, False, False),
    "tekken": ((TEKKEN_PATTERN,), False, True),
    "pixtral": ((TEKKEN_PATTERN,), False, True),
}
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
    # A RADIO projector's position grid is its table's extents.
    vision = None if vision is None else Metadata(vision, tensors=True)
    config = model_config(metadata, vision)
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
        or (
            profile := TOKENIZER_PROFILES.get(
                metadata.require("tokenizer.ggml.pre", str)
            )
        )
        is None
    ):
        raise ModelError(
            "unsupported GGUF tokenizer profile; expected gpt2/"
            + " or gpt2/".join(TOKENIZER_PROFILES)
        )
    patterns, nfc, ignore_merges = profile
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
            token_models.BPE(
                {t: i for i, t in enumerate(tokens)}, pairs, ignore_merges=ignore_merges
            )
        )
    except Exception as error:
        raise ModelError("invalid GGUF BPE vocabulary or merges") from error
    if nfc:
        backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            *(
                pre_tokenizers.Split(Regex(pattern), behavior="isolated")
                for pattern in patterns
            ),
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
    """The configuration of the target metadata describes, in the keys and
    the shape of its family's own config.json: Qwen's text model under
    text_config, with its vision tower's beside it, or Flint's flat."""
    arch = text_architecture(metadata)
    architecture = ARCHITECTURES[arch]
    fields = {
        "hidden_size": "embedding_length",
        "max_position_embeddings": "context_length",
        "num_attention_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv",
        "head_dim": "attention.key_length",
    }
    if architecture.single_blocks:
        # Per-layer arrays: the engine reads them from the GGUF; the config
        # states each layer's block and the attention layers' KV heads.
        del fields["num_key_value_heads"]
    if architecture.moe:
        # The routed experts, which a pruned release keeps fewer of than its
        # family; the engine plans, sizes and loads them by this count.
        fields["num_experts"] = "expert_count"
    text = {name: metadata.positive(arch + "." + key) for name, key in fields.items()}
    text.update(
        num_hidden_layers=loaded_layers(metadata, arch),
        model_type=architecture.model_type,
        vocab_size=len(metadata.require("tokenizer.ggml.tokens", list)),
    )
    if architecture.text_config is not None:
        text.update(architecture.text_config(metadata, arch))
    if architecture.flat:
        return text
    config = {
        "model_type": architecture.model_type.removesuffix("_text"),
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


def dense_layers(metadata, arch):
    """The leading layers whose FFN is dense in a MoE target whose
    architecture has such layers (leading_dense_block_count), fewer than its
    layers; none in any other."""
    if not ARCHITECTURES[arch].leading_dense:
        return 0
    key = arch + ".leading_dense_block_count"
    dense = metadata.require(key, int)
    if not 0 <= dense < loaded_layers(metadata, arch):
        raise ModelError("invalid GGUF metadata: " + key)
    return dense


def layer_pattern(metadata, arch, name):
    """A boolean per loaded layer, the header's array `name`."""
    key = f"{arch}.{name}"
    values = metadata.values.get(key)
    if (
        type(values) is not list
        or len(values) != loaded_layers(metadata, arch)
        or any(type(value) is not bool for value in values)
    ):
        raise ModelError("missing or invalid GGUF metadata: " + key)
    return values


# The router functions of llama_expert_gating_func_type, by its value, as a
# configuration's router_score_function names them.
EXPERT_GATING_FUNCTIONS = {1: "softmax", 2: "sigmoid"}


def flint_stop_tokens(metadata):
    """The tokens that end a Flint turn: the header's
    tokenizer.ggml.eog_token_ids where it states them; otherwise the IDs of
    Flint's stop tokens (FLINT.stop_tokens), each a control token once in the
    vocabulary, one of them the header's EOS. unsloth/flint-GGUF states the
    list; other publishers' Flint GGUFs, of the same vocabulary, do not."""
    key = "tokenizer.ggml.eog_token_ids"
    if key in metadata.values:
        stops = metadata.values[key]
        if (
            type(stops) is not list
            or not stops
            or any(type(token) is not int for token in stops)
        ):
            raise ModelError("invalid GGUF metadata: " + key)
        return stops
    tokens = metadata.require("tokenizer.ggml.tokens", list)
    types = metadata.require("tokenizer.ggml.token_type", list)
    stops = []
    for text in FLINT.stop_tokens:
        found = [index for index, token in enumerate(tokens) if token == text]
        if (
            len(found) != 1
            or len(types) != len(tokens)
            or types[found[0]] != CONTROL_TOKEN
        ):
            raise ModelError(
                f"GGUF metadata states no {key}, and its vocabulary does not "
                f"hold {text} once as a control token"
            )
        stops += found
    if metadata.require("tokenizer.ggml.eos_token_id", int) not in stops:
        raise ModelError(
            f"GGUF metadata states no {key}, and its EOS is none of Flint's stop tokens"
        )
    return stops


def flint_text_config(metadata, arch):
    """What Flint's config.json states beyond the sizes every architecture's
    does, from its header: the dense, routed and shared experts' FFN widths,
    the experts each token takes and how the router weighs them, the leading
    dense layers, the sliding window, the layers that slide
    (attention.sliding_window_pattern) and those without RoPE
    (attention.rope_pattern), and the tokens that end a turn. The engine's
    model check holds each to the family's, and the header to them."""
    gating = metadata.require(arch + ".expert_gating_func", int)
    if gating not in EXPERT_GATING_FUNCTIONS:
        raise ModelError(f"unsupported GGUF expert gating function: {gating}")
    stops = flint_stop_tokens(metadata)
    sliding = layer_pattern(metadata, arch, "attention.sliding_window_pattern")
    rotated = layer_pattern(metadata, arch, "attention.rope_pattern")
    return {
        "intermediate_size": metadata.positive(arch + ".feed_forward_length"),
        "moe_intermediate_size": metadata.positive(
            arch + ".expert_feed_forward_length"
        ),
        "shared_expert_intermediate_size": metadata.positive(
            arch + ".expert_shared_feed_forward_length"
        ),
        "num_experts_per_tok": metadata.positive(arch + ".expert_used_count"),
        "norm_topk_prob": metadata.require(arch + ".expert_weights_norm", bool),
        "router_score_function": EXPERT_GATING_FUNCTIONS[gating],
        "mlp_only_layers": list(range(dense_layers(metadata, arch))),
        "sliding_window": metadata.positive(arch + ".attention.sliding_window"),
        "layer_types": [
            "sliding_attention" if slides else "full_attention" for slides in sliding
        ],
        "no_rope_layer_indices": [
            layer for layer, rotates in enumerate(rotated) if not rotates
        ],
        "eos_token_id": stops,
    }


def vision_config(vision):
    """The vision_config of a clip mmproj: Qwen3.5's tower (qwen3vl_merger),
    or Nemotron 3.5 Bolt's RADIO (nemotron_omni, radio_vision_config)."""
    projector = vision.require("clip.projector_type", str)
    if (
        vision.require("general.architecture", str) != "clip"
        or projector not in ("qwen3vl_merger", "nemotron_omni")
        or vision.require("clip.use_gelu", bool) is not True
    ):
        raise ModelError("unsupported GGUF vision architecture")
    if projector == "nemotron_omni":
        return radio_vision_config(vision)
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


def radio_vision_config(vision):
    """RADIO's vision_config, the keys of Qwen3.5's (the engine checks both,
    ModelDescriptor.mm): one-frame RGB patches, erf GELU, the 2x2 merge its
    projector states as scale_factor, and the position grid of its table,
    v.position_embd.weight [hidden, cells], which its image_size (the
    preprocessing's) does not give. The header must be read with tensors."""
    fields = {
        "depth": "block_count",
        "hidden_size": "embedding_length",
        "num_heads": "attention.head_count",
        "intermediate_size": "feed_forward_length",
        "out_hidden_size": "projection_dim",
        "patch_size": "patch_size",
        "spatial_merge_size": "projector.scale_factor",
    }
    result = {
        name: vision.positive("clip.vision." + key) for name, key in fields.items()
    }
    extents = vision.extents.get("v.position_embd.weight") or ()
    if extents[2:] == (1,):  # Super's mmproj states a third extent of 1
        extents = extents[:2]
    if len(extents) != 2 or extents[0] != result["hidden_size"]:
        raise ModelError("invalid GGUF vision position grid")
    # Super's tower ends in a LayerNorm (v.post_ln), Bolt's does not.
    if "v.post_ln.weight" in vision.extents:
        result["final_norm"] = True
    result.update(
        temporal_patch_size=1,
        in_channels=3,
        num_position_embeddings=extents[1],
        hidden_act="gelu",
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


# The names of GGML tensor types (ggml_type), by id, which refusals name a
# type by: names only, never which types the engine loads.
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


def require_quantized(headers):
    """Splash serves quantized GGUFs: a target whose files' headers hold more
    of its values in float tensors (BF16, F16 or F32) than in quantized ones
    is a whole-BF16, F16 or F32 model, refused before any download, even
    with its token table or a few small tensors quantized. A quantized
    release that keeps some projections in BF16, as UD-Q8_K_XL's do,
    quantizes most of its values."""
    floats = quantized = 0
    for header in headers:
        for name, kind in header.tensors.items():
            values = math.prod(header.extents[name])
            if TENSOR_TYPES.get(kind) in ("BF16", "F16", "F32"):
                floats += values
            else:
                quantized += values
    if floats > quantized:
        raise ModelError(
            "this GGUF stores its weights unquantized (BF16, F16 or F32): Splash "
            "loads quantized GGUFs, not whole-BF16 or F16 models; choose a "
            "quantized variant of the repository"
        )


@dataclass(frozen=True)
class Architecture:
    """A GGUF text architecture Splash serves, as model_config derives its
    configuration."""

    # The model type of its text configuration.
    model_type: str
    # Whether its FFN is a sparse MoE, else dense.
    moe: bool
    # Whether each layer is a single block, Mamba-2, attention or MoE
    # (nemotron_blocks), whose KV heads are a per-layer array.
    single_blocks: bool = False
    # Whether a MoE target's leading_dense_block_count leading layers have a
    # dense FFN (dense_layers).
    leading_dense: bool = False
    # What its configuration states beyond the sizes model_config derives,
    # given the metadata and the architecture's name; None for nothing more.
    text_config: Callable | None = None
    # Whether its configuration is flat, the text model's at its top, as
    # Flint's is; otherwise under text_config, as Qwen's is.
    flat: bool = False


def nemotron_blocks(metadata):
    """Each layer's block of a Nemotron-H target (nemotron_h_moe, Nemotron
    3.5 Bolt and Super), Mamba-2, NoPE attention or a latent MoE, as
    llama.cpp tells them apart (src/models/nemotron-h.cpp): a MoE where
    feed_forward_length is nonzero, attention where head_count_kv is
    instead, else Mamba-2. The arrays must cover every layer, and the expert
    count each MoE takes must be nonzero exactly there."""
    arch = text_architecture(metadata)
    layers = loaded_layers(metadata, arch)
    arrays = {}
    for key in ("feed_forward_length", "attention.head_count_kv", "expert_used_count"):
        value = metadata.values.get(f"{arch}.{key}")
        # Super's GGUFs state the experts each MoE takes as one scalar.
        if key == "expert_used_count" and type(value) is int:
            value = [value if ffn else 0 for ffn in arrays["feed_forward_length"]]
        if (
            type(value) is not list
            or len(value) < layers
            or not all(type(v) is int and v >= 0 for v in value)
        ):
            raise ModelError("missing or invalid GGUF metadata array: " + key)
        arrays[key] = value[:layers]
    kinds = []
    for ffn, kv, used in zip(
        arrays["feed_forward_length"],
        arrays["attention.head_count_kv"],
        arrays["expert_used_count"],
    ):
        if (ffn and kv) or bool(ffn) != bool(used):
            raise ModelError("GGUF layer arrays name no single block")
        kinds.append("moe" if ffn else "attention" if kv else "mamba")
    return kinds


def nemotron_text_config(metadata, arch):
    """What Nemotron-H's configuration states beyond the sizes every
    architecture's does: each layer's block, and the attention layers' KV
    heads, which its header states per layer."""
    kv = [n for n in metadata.values[arch + ".attention.head_count_kv"] if n]
    return {"layer_types": nemotron_blocks(metadata), "num_key_value_heads": max(kv)}


# The GGUF text architectures Splash serves, by general.architecture.
ARCHITECTURES = {
    "qwen35": Architecture("qwen3_5_text", moe=False),
    "qwen35moe": Architecture("qwen3_5_moe_text", moe=True),
    "qwen4exp": Architecture("qwen4_exp_text", moe=True),
    "nemotron_h_moe": Architecture(
        "nemotron_h_moe_text",
        moe=True,
        text_config=nemotron_text_config,
        single_blocks=True,
    ),
    "flint": Architecture(
        "flint",
        moe=True,
        leading_dense=True,
        text_config=flint_text_config,
        flat=True,
    ),
}
