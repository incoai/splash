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
# The inkling2 profile, as Inkling 2's own tokenizer.json splits text before
# its byte-level BPE, without a normalizer: o200k's pattern alone, digits in
# runs of up to three.
INKLING2_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# Each GGUF tokenizer profile (tokenizer.ggml.pre) Splash derives a tokenizer
# for: the patterns that split text, each isolating its matches, in order;
# whether text is NFC-normalized before them; and whether its BPE takes a
# piece its vocabulary holds whole as that token before any merge
# (ignore_merges), as o200k's does and Inkling 2's tokenizer.json states.
TOKENIZER_PROFILES = {
    "qwen35": ((QWEN35_PATTERN,), True, False),
    "flint": (FLINT_PATTERNS, False, False),
    "inkling2": ((INKLING2_PATTERN,), False, True),
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
    # A projector's tensor table states what its metadata does not, such as
    # Inkling's projector width (inkling_vision_config).
    config = model_config(
        metadata, None if vision is None else Metadata(vision, tensors=True)
    )
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
                {t: i for i, t in enumerate(tokens)},
                pairs,
                ignore_merges=ignore_merges,
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
    the shape of its family's own config.json: Qwen's and Inkling 2's text
    model under text_config, Qwen's with its vision tower's beside it, or
    Flint's flat."""
    arch = text_architecture(metadata)
    architecture = ARCHITECTURES[arch]
    fields = {
        "hidden_size": "embedding_length",
        "max_position_embeddings": "context_length",
        "num_attention_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv",
        "head_dim": "attention.key_length",
    }
    if architecture.moe:
        # The routed experts, which a pruned release keeps fewer of than its
        # family; the engine plans, sizes and loads them by this count.
        fields["num_experts"] = "expert_count"
    text = {name: layer_size(metadata, arch, key) for name, key in fields.items()}
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


def layer_size(metadata, arch, name, layers=None):
    """The positive size at the header's `name`, which llama.cpp states once,
    or as one value per loaded layer where an architecture's layers may
    differ (Inkling 2's head counts): the one value the layers of `layers`,
    every loaded layer by default, share."""
    key = f"{arch}.{name}"
    values = metadata.values.get(key)
    if type(values) is not list:
        return metadata.positive(key)
    count = loaded_layers(metadata, arch)
    shared = set()
    if len(values) == count:
        chosen = range(count) if layers is None else layers
        shared = {values[layer] for layer in chosen}
    if len(shared) != 1 or type(size := shared.pop()) is not int or size <= 0:
        raise ModelError("invalid GGUF metadata: " + key)
    return size


def real(metadata, key):
    """The header's float32 at key, as the shortest decimal that is the same
    float32: the value a release's config.json states, which its converter
    stored (1e-06, not 9.999999974752427e-07). A value no float32 holds, of
    a float64 key, as it is."""
    value = metadata.require(key, float)
    for digits in range(1, 10):
        shortest = float(f"{value:.{digits}g}")
        if struct.unpack("<f", struct.pack("<f", shortest))[0] == value:
            return shortest
    return value


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


# The loop's keys in each of Inkling 2's GGUF layouts: its first layer, the
# layer past its last and its passes. llama.cpp's converter writes
# inkling2_loop; Unsloth's own layout (inkling2, the root files of
# unsloth/Inkling-2-Mini-GGUF) holds the same tensors of its weight layers
# but states its layer counts and arrays per layer application, the loop's
# layers once a pass, and no loop.varying (its passes are every token's).
INKLING2_LOOP_KEYS = {
    "inkling2_loop": ("loop.start", "loop.end", "loop.count"),
    "inkling2": ("loop.core_start", "loop.core_end", "num_loops"),
}


def inkling2_text_config(metadata, arch):
    """What Inkling 2's text_config states beyond the sizes every
    architecture's does, by its release's keys, from its header of either
    layout (INKLING2_LOOP_KEYS): its weight layers (num_hidden_layers, the
    applications less the loop's later passes in Unsloth's layout), its
    window (model_max_length), its vocabulary padded to the token table's
    rows (vocab_size) and its tokens (unpadded_vocab_size, the tokenizer's),
    the relative bias's width and reach, the log scaling of its full layers'
    queries, its RMS epsilon, its query, key and bias clips, the layers that
    slide (local_layer_ids, from attention.sliding_window_pattern, of each
    weight layer's first application) and their window and heads, its short
    convolutions' taps, its block mixes' layers, the logits' and its
    experts' softcaps, the experts each token takes, their width and how
    their weights scale, its loop (loop_config), and the token that ends a
    turn, which its release states beside text_config. The engine's model
    check holds each to the family's."""

    def key(name):
        return f"{arch}.{name}"

    tokens = len(metadata.require("tokenizer.ggml.tokens", list))
    unpadded = metadata.positive(key("vocab_size_unpadded"))
    if unpadded != tokens:
        raise ModelError("invalid GGUF metadata: " + key("vocab_size_unpadded"))
    padded = metadata.positive(key("vocab_size"))
    if padded < unpadded:
        raise ModelError("invalid GGUF metadata: " + key("vocab_size"))
    # Layers [core_start, core_end) run num_passes times.
    start_key, end_key, passes_key = INKLING2_LOOP_KEYS[arch]
    start = metadata.require(key(start_key), int)
    end = metadata.positive(key(end_key))
    passes = metadata.positive(key(passes_key))
    applications = loaded_layers(metadata, arch)
    # The applications past the loop's first pass, which Unsloth's layout counts.
    repeated = (passes - 1) * (end - start) if arch == "inkling2" else 0
    layers = applications - repeated
    if not 0 <= start < end <= layers:
        raise ModelError(f"invalid GGUF metadata: {key(start_key)}, {key(end_key)}")

    def application(layer):
        """A weight layer's first application in the header's arrays."""
        return layer + repeated if layer >= end else layer

    pattern = layer_pattern(metadata, arch, "attention.sliding_window_pattern")
    sliding = [layer for layer in range(layers) if pattern[application(layer)]]
    sliding_applications = [application(layer) for layer in sliding]
    return {
        "num_hidden_layers": layers,
        "model_max_length": metadata.positive(key("context_length")),
        "vocab_size": padded,
        "unpadded_vocab_size": unpadded,
        "d_rel": metadata.positive(key("attention.relative_length")),
        "rel_extent": metadata.positive(key("attention.relative_extent")),
        "log_scaling_n_floor": metadata.positive(key("attention.log_scale_floor")),
        "log_scaling_alpha": real(metadata, key("attention.log_scale_alpha")),
        "rms_norm_eps": real(metadata, key("attention.layer_norm_rms_epsilon")),
        "qk_l2_clip_threshold": real(metadata, key("attention.qk_clip")),
        "rel_l2_clip_threshold": real(metadata, key("attention.relative_clip")),
        "local_layer_ids": sliding,
        "sliding_window_size": metadata.positive(key("attention.sliding_window")),
        "swa_head_dim": metadata.positive(key("attention.key_length_swa")),
        "swa_num_attention_heads": layer_size(
            metadata, arch, "attention.head_count", sliding_applications
        ),
        "swa_num_key_value_heads": layer_size(
            metadata, arch, "attention.head_count_kv", sliding_applications
        ),
        "sconv_kernel_size": metadata.positive(key("shortconv.l_cache")),
        "n_layers_per_attn_res_block": metadata.positive(key("attn_res.block_size")),
        "final_logit_softcapping": real(metadata, key("final_logit_softcapping")),
        "n_routed_experts": metadata.positive(key("expert_count")),
        "num_experts_per_tok": metadata.positive(key("expert_used_count")),
        # Every FFN is a MoE: the release's intermediate_size is an expert's.
        "intermediate_size": metadata.positive(key("expert_feed_forward_length")),
        "mlp_gate_softcap": real(metadata, key("activation.situ_beta")),
        "mlp_up_softcap": real(metadata, key("activation.situ_linear_beta")),
        "route_scale": real(metadata, key("expert_weights_scale")),
        "loop_config": {
            "core_start": start,
            "core_end": end,
            "num_passes": passes,
            "gate_rank": metadata.positive(key("loop.gate_rank")),
            "is_varloop": arch != "inkling2"
            and metadata.require(key("loop.varying"), bool),
        },
        "eos_token_id": metadata.require("tokenizer.ggml.eos_token_id", int),
    }


def projector_type(vision):
    """The clip projector type of a vision GGUF: Qwen's state it as
    clip.projector_type, Inkling's as clip.vision.projector_type."""
    if vision.require("general.architecture", str) != "clip":
        raise ModelError("unsupported GGUF vision architecture")
    kind = vision.values.get("clip.projector_type")
    return kind if kind is not None else vision.values.get("clip.vision.projector_type")


def inkling_vision_config(vision):
    """Inkling 2's vision_config (inkling2_vision) as its config.json states
    it, in transformers' names, from an inkling2v projector's header: its
    sizes, its MLP's width as a ratio of its own, its patches' side and merge
    (native_patch_size, adapter_merge_size), the side of its position grid
    (image_size over the patch side), its projector's hidden width (mm.1's
    rows) and whether that has biases, and its LayerNorms' epsilon as the
    shortest decimal of the header's float32."""
    fields = {
        "num_hidden_layers": "block_count",
        "hidden_size": "embedding_length",
        "num_attention_heads": "attention.head_count",
        "text_hidden_size": "projection_dim",
        "native_patch_size": "patch_size",
        "adapter_merge_size": "spatial_merge_size",
    }
    result = {
        name: vision.positive("clip.vision." + key) for name, key in fields.items()
    }
    image_size = vision.positive("clip.vision.image_size")
    if image_size % result["native_patch_size"]:
        raise ModelError("invalid GGUF vision position grid")
    projection = vision.extents.get("mm.1.weight")
    if projection is None or len(projection) != 2:
        raise ModelError("the inkling2v projector has no mm.1.weight")
    result.update(
        model_type="inkling2_vision",
        mlp_ratio=vision.positive("clip.vision.feed_forward_length")
        / result["hidden_size"],
        posemb_grid_size=image_size // result["native_patch_size"],
        projection_hidden_dim=projection[1],
        projector_bias="mm.1.bias" in vision.tensors,
        layer_norm_eps=real(vision, "clip.vision.attention.layer_norm_epsilon"),
    )
    return result


def vision_config(vision):
    kind = projector_type(vision)
    if kind == "inkling2v":
        return inkling_vision_config(vision)
    if kind != "qwen3vl_merger" or vision.require("clip.use_gelu", bool) is not True:
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
    """How the projector's tower takes images: its patches (in two temporal
    slices for Qwen's), their merge and the normalization."""
    config = vision_config(vision)
    if config.get("model_type") == "inkling2_vision":
        patch, temporal = config["native_patch_size"], {}
        merge = config["adapter_merge_size"]
    else:
        patch, merge = config["patch_size"], config["spatial_merge_size"]
        temporal = {"temporal_patch_size": config["temporal_patch_size"]}
    return {
        "patch_size": patch,
        **temporal,
        "merge_size": merge,
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
    # Whether a MoE target's leading_dense_block_count leading layers have a
    # dense FFN (dense_layers).
    leading_dense: bool = False
    # What its configuration states beyond the sizes model_config derives,
    # given the metadata and the architecture's name; None for nothing more.
    text_config: Callable | None = None
    # Whether its configuration is flat, the text model's at its top, as
    # Flint's is; otherwise under text_config, as Qwen's and Inkling 2's are.
    flat: bool = False


# The GGUF text architectures Splash serves, by general.architecture.
ARCHITECTURES = {
    "qwen35": Architecture("qwen3_5_text", moe=False),
    "qwen35moe": Architecture("qwen3_5_moe_text", moe=True),
    "qwen4exp": Architecture("qwen4_exp_text", moe=True),
    "flint": Architecture(
        "flint",
        moe=True,
        leading_dense=True,
        text_config=flint_text_config,
        flat=True,
    ),
    "inkling2_loop": Architecture(
        "inkling2_loop_text", moe=True, text_config=inkling2_text_config
    ),
    "inkling2": Architecture(
        "inkling2_loop_text", moe=True, text_config=inkling2_text_config
    ),
}
