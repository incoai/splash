#!/usr/bin/env python3
"""fp32 reference forward of DeepSeek-V4.1-Flash, read from its GGUFs.

This is the executable specification Splash's DeepSeek-V4.1 target is graded
against. It reads a split GGUF of the ``deepseek41`` architecture in
llama.cpp's layout, the Engram tables its last part (``per_layer_token_embd``,
a file of its own name beside the others where the variants share it), with
gguf-py, which also dequantizes their tensors, and runs the forward pass of DeepSeek's reference
implementation (deepseek-ai/DeepSeek-V4.1-Flash-1017, ``inference/model.py``
and ``inference/engram.py``, MIT) entirely in numpy fp32, one layer and one
routed expert at a time:

- the token table's rows, copied into ``hc_mult`` (4) hyper-connection
  streams. Each sublayer reads one weighted sum of the streams and writes its
  output back to every stream beside a Sinkhorn-balanced mix of them; the
  weights of a sublayer's input come from the sublayer before it (``hc_*``);
- Engram before the layers ``ple.layers`` names: the 2- to 4-gram hashes of
  each token's compressed id (``engram.token_map``) and the ids before it
  select 24 rows of the layer's heads in the tables (``ple.head_offsets``),
  whose projection gates a value into every stream;
- attention of 64 query heads over one latent KV head of 512, whose last 64
  values rotate as interleaved pairs, with a learned sink per head: over a
  sliding window of the 128 latest raw latents and, on the layers
  ``compress_ratios`` gives a ratio, over the compressed latents an indexer
  selects (its top 512 by ReLU-weighted scores, the later index layers among
  the 2048 blocks of 8 the candidate layer keeps). Compressed latents and the
  indexer's keys come from the ``attention.kv_source_layers`` layers, a
  selection from the ``attention.indexer.source_layers`` layers, each read by the layers up to the
  next. The output is rotated back and projected per group of 8 heads to 1024,
  then to the hidden size. Layers without a ratio rotate by
  ``rope.freq_base``; the others by ``compress_rope_freq_base`` with YaRN;
- the MoE: the top 6 of 384 experts by sqrtsoftplus score plus their routing
  bias, weighted by their scores normalized to sum to 1 and scaled by 1.5, as
  SwiGLU with the up branch clamped to [-10, 10] and the gate to at most 10,
  plus the ungated shared expert;
- the streams' last mix, the final norm and the untied output head.

Every row is computed as model.py prefills it: a row attends to the window of
128 tokens ending at it and to the compressed latents of the groups it has
passed, which is what model.py's decode steps compute too. ``--generate``
extends the tokens greedily, computing only the new rows from the layers'
cached inputs.

model.py rounds the window latents and the attention output to FP8 (E4M3,
power-of-two scales per 32 values), the compressed latents to FP4 (E2M1, an
E4M3 scale per 16) and the indexer's queries and keys to FP4 (power-of-two
scales per 32); ``--kv-rounding reference`` (the default) rounds them so,
``none`` keeps them fp32. Projections read the GGUF's dequantized weights with
fp32 activations: model.py's FP8 and FP4 GEMMs, which quantize activations
too, are not reproduced, nor its bf16 activations.

    python dev/tools/deepseek41_reference.py SHARD-00001-of-0000N.gguf TOKENS OUT_DIR

TOKENS is a .npy array or a text file of whitespace-separated token ids, as
the model's tokenizer and chat template give them. OUT_DIR receives:

- tokens.npy (uint32), the tokens with those --generate added;
- logits.npy, fp32 [tokens, vocabulary]: the logits after each prefix;
- residual_norms.npy, fp32 [layers, 2, tokens, streams]: the L2 norm of each
  stream of the residual after each layer's attention (0) and FFN (1);
- experts.npy (int32) and expert_weights.npy (fp32), [layers named, tokens,
  6]: each row's routed experts, by descending biased score, and their
  weights, at the layers --router-layers names;
- reference.json: the GGUF, its layout and what was dumped.

``--layers N`` computes the prompt's first N layers alone, for a comparison of
their streams (with ``--dump-streams``): no logits, no generated tokens.

Without the Engram tables' file beside the shards, ``--engram-url URL`` reads
the rows a run needs from that file's URL with HTTP range requests. The run needs
gguf-py (pip install gguf) and memory for one layer's weights in fp32, about
3 GB.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import struct
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import gguf
import numpy as np

ARCHITECTURE = "deepseek41"
# The Engram tables, every Engram layer's heads back to back.
ENGRAM_TABLE = "per_layer_token_embd.weight"
# DeepSeek's router, sqrt(softplus), as llama.cpp numbers its gating.
SQRT_SOFTPLUS_GATING = 4
# Engram rows read at once over HTTP.
REMOTE_READERS = 16
# Output head rows dequantized at once.
HEAD_ROWS = 16384
FP8_MAX = 448.0
FP4_MAX = 6.0


@dataclass(frozen=True)
class Layout:
    """A deepseek41 GGUF's geometry, from its header."""

    layers: int
    hidden: int
    heads: int
    head_dim: int
    rope_dim: int
    q_rank: int
    o_groups: int
    o_rank: int
    window: int
    compress_ratios: tuple[int, ...]
    kv_sources: tuple[int, ...]
    index_sources: tuple[int, ...]
    index_heads: int
    index_dim: int
    index_topk: int
    candidate_source: int
    candidate_blocks: int
    candidate_block_size: int
    streams: int
    sinkhorn_iterations: int
    hc_eps: float
    norm_eps: float
    experts: int
    experts_used: int
    expert_ffn: int
    route_scale: float
    normalize_routes: bool
    swiglu_limit: tuple[float, ...]
    swiglu_limit_shared: tuple[float, ...]
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    rope_original: int
    beta_fast: float
    beta_slow: float
    engram_layers: tuple[int, ...]
    engram_ngram: int
    engram_heads: int
    engram_dim: int
    engram_pad: int

    @classmethod
    def read(cls, meta: dict) -> Layout:
        a = ARCHITECTURE + "."

        def need(key):
            if key not in meta:
                raise SystemExit(f"the GGUF has no {key}")
            return meta[key]

        if need("general.architecture") != ARCHITECTURE:
            raise SystemExit(
                f"not a {ARCHITECTURE} GGUF: {meta['general.architecture']}"
            )
        if a + "engram.file" in meta:
            raise SystemExit(
                "the GGUF is in the layout before llama.cpp's (an Engram file the "
                "backbone names); convert it with ISTA-DASLab's convert_ds41_upstream.py"
            )
        if int(need(a + "expert_gating_func")) != SQRT_SOFTPLUS_GATING:
            raise SystemExit(f"unknown expert gating {meta[a + 'expert_gating_func']}")
        if need(a + "rope.scaling.type") != "yarn":
            raise SystemExit(f"unknown RoPE scaling {meta[a + 'rope.scaling.type']}")
        layers = int(need(a + "block_count"))
        return cls(
            layers=layers,
            hidden=int(need(a + "embedding_length")),
            heads=int(need(a + "attention.head_count")),
            head_dim=int(need(a + "attention.key_length")),
            rope_dim=int(need(a + "rope.dimension_count")),
            q_rank=int(need(a + "attention.q_lora_rank")),
            o_groups=int(need(a + "attention.output_group_count")),
            o_rank=int(need(a + "attention.output_lora_rank")),
            window=int(need(a + "attention.sliding_window")),
            compress_ratios=tuple(
                int(r) for r in need(a + "attention.compress_ratios")
            )[:layers],
            kv_sources=tuple(int(x) for x in need(a + "attention.kv_source_layers")),
            index_sources=tuple(
                int(x) for x in need(a + "attention.indexer.source_layers")
            ),
            index_heads=int(need(a + "attention.indexer.head_count")),
            index_dim=int(need(a + "attention.indexer.key_length")),
            index_topk=int(need(a + "attention.indexer.top_k")),
            candidate_source=int(need(a + "attention.indexer.candidate_source_layer")),
            candidate_blocks=int(need(a + "attention.indexer.candidate_top_k_blocks")),
            candidate_block_size=int(
                need(a + "attention.indexer.candidate_block_size")
            ),
            streams=int(need(a + "hyper_connection.count")),
            sinkhorn_iterations=int(need(a + "hyper_connection.sinkhorn_iterations")),
            hc_eps=float(need(a + "hyper_connection.epsilon")),
            norm_eps=float(need(a + "attention.layer_norm_rms_epsilon")),
            experts=int(need(a + "expert_count")),
            experts_used=int(need(a + "expert_used_count")),
            expert_ffn=int(need(a + "expert_feed_forward_length")),
            route_scale=float(need(a + "expert_weights_scale")),
            normalize_routes=bool(need(a + "expert_weights_norm")),
            swiglu_limit=tuple(float(x) for x in need(a + "swiglu_clamp_exp")),
            swiglu_limit_shared=tuple(float(x) for x in need(a + "swiglu_clamp_shexp")),
            rope_theta=float(need(a + "rope.freq_base")),
            compress_rope_theta=float(need(a + "attention.compress_rope_freq_base")),
            rope_factor=float(need(a + "rope.scaling.factor")),
            rope_original=int(need(a + "rope.scaling.original_context_length")),
            beta_fast=float(need(a + "rope.scaling.yarn_beta_fast")),
            beta_slow=float(need(a + "rope.scaling.yarn_beta_slow")),
            engram_layers=tuple(int(x) for x in need(a + "ple.layers")),
            engram_ngram=int(need(a + "ple.ngram_size")),
            engram_heads=int(need(a + "ple.heads_per_ngram")),
            engram_dim=int(need(a + "embedding_length_per_layer_input")),
            engram_pad=int(need(a + "engram.pad_token_id")),
        )

    def kv_source(self, layer: int) -> int:
        """The layer whose compressed latents and index keys `layer` reads."""
        return max(s for s in self.kv_sources if s <= layer)

    def index_source(self, layer: int) -> int:
        """The layer whose selection of compressed latents `layer` reads."""
        return max(s for s in self.index_sources if s <= layer)


def field_value(field):
    value = field.contents()
    return value.decode() if isinstance(value, bytes) else value


class RemoteRows:
    """Rows of one tensor of a remote GGUF, read with HTTP range requests and
    kept once read."""

    def __init__(self, url: str, offset: int, row_bytes: int, tensor_type):
        self.url, self.offset, self.row_bytes, self.tensor_type = (
            url,
            offset,
            row_bytes,
            tensor_type,
        )
        self.rows: dict[int, np.ndarray] = {}
        # A hub URL redirects to its storage; the rows are read from there.
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-{offset}"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            self.url = response.geturl()

    def read(self, rows: np.ndarray) -> np.ndarray:
        missing = sorted({int(r) for r in rows.ravel()} - self.rows.keys())
        with concurrent.futures.ThreadPoolExecutor(REMOTE_READERS) as pool:
            for row, data in zip(missing, pool.map(self.fetch, missing), strict=True):
                block = np.frombuffer(data, dtype=np.uint8).reshape(1, self.row_bytes)
                self.rows[row] = np.asarray(
                    gguf.dequantize(block, self.tensor_type), np.float32
                )[0]
        return np.stack([self.rows[int(r)] for r in rows.ravel()]).reshape(
            *rows.shape, -1
        )

    def fetch(self, row: int) -> bytes:
        start = self.offset + row * self.row_bytes
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{start + self.row_bytes - 1}"}
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                if len(data) == self.row_bytes:
                    return data
            except OSError:
                if attempt == 2:
                    raise
        raise OSError(f"short read of row {row} from {self.url}")


def remote_tensors(url: str) -> tuple[dict, dict]:
    """The metadata and tensors (type, numpy shape, absolute offset) of a
    remote GGUF, from its header alone."""
    size = 1 << 20
    while True:
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            header = response.read()
        try:
            return parse_header(header)
        except struct.error:
            if len(header) < size:
                raise
            size *= 4


def parse_header(b: bytes) -> tuple[dict, dict]:
    offset = 0

    def take(fmt):
        nonlocal offset
        values = struct.unpack_from("<" + fmt, b, offset)
        offset += struct.calcsize("<" + fmt)
        return values[0]

    def string():
        nonlocal offset
        n = take("Q")
        offset += n
        if offset > len(b):
            raise struct.error("past the header")
        return b[offset - n : offset].decode()

    scalars = {
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

    def value(kind):
        if kind in scalars:
            return take(scalars[kind])
        if kind == 8:
            return string()
        if kind == 9:
            item, count = take("I"), take("Q")
            return [value(item) for _ in range(count)]
        raise ValueError(f"GGUF value type {kind}")

    if b[:4] != b"GGUF":
        raise SystemExit("not a GGUF")
    offset = 8
    tensor_count, kv_count = take("Q"), take("Q")
    meta = {}
    for _ in range(kv_count):
        key = string()
        meta[key] = value(take("I"))
    infos = []
    for _ in range(tensor_count):
        name = string()
        dims = [take("Q") for _ in range(take("I"))]
        infos.append((name, dims, gguf.GGMLQuantizationType(take("I")), take("Q")))
    alignment = int(meta.get("general.alignment", 32))
    data = (offset + alignment - 1) // alignment * alignment
    tensors = {
        name: (kind, tuple(reversed(dims)), data + at) for name, dims, kind, at in infos
    }
    return meta, tensors


class Model:
    """A deepseek41 GGUF's tensors, each dequantized to fp32 as it is read."""

    def __init__(self, first: Path, engram_url: str | None):
        # Absolute, not resolved: a Hugging Face cache names the parts by
        # symlinks to blobs whose names say nothing of the split.
        first = first.absolute()
        reader = gguf.GGUFReader(first)
        self.meta = {
            name: field_value(field)
            for name, field in reader.fields.items()
            if not name.startswith("tokenizer.ggml.") or name.endswith("_id")
        }
        self.layout = Layout.read(self.meta)
        self.paths = [first]
        count = int(self.meta.get("split.count", 1))
        if count > 1:
            suffix = f"-00001-of-{count:05d}.gguf"
            if not first.name.endswith(suffix):
                raise SystemExit(
                    f"{first.name} is not the first of {count} parts (…{suffix})"
                )
            stem = first.name[: -len(suffix)]
            self.paths = [
                first.with_name(f"{stem}-{k:05d}-of-{count:05d}.gguf")
                for k in range(1, count + 1)
            ]
            # The last part the variants share under a name of its own.
            if not self.paths[-1].exists():
                self.paths[-1] = self.shared_part(first, count)
        self.tensors = {}
        self.engram_source = None
        for path in self.paths:
            if path is None:
                continue
            part = reader if path == first else gguf.GGUFReader(path)
            for tensor in part.tensors:
                self.tensors[tensor.name] = tensor
            if ENGRAM_TABLE in self.tensors and self.engram_source is None:
                self.engram_source = str(path)
        if self.layout.engram_layers and ENGRAM_TABLE not in self.tensors:
            if not engram_url:
                raise SystemExit(
                    f"the Engram tables, part {count} of {count}, are missing; "
                    "download them beside the shards or pass --engram-url"
                )
            _, tensors = remote_tensors(engram_url)
            if ENGRAM_TABLE not in tensors:
                raise SystemExit(f"--engram-url holds no {ENGRAM_TABLE}")
            kind, shape, offset = tensors[ENGRAM_TABLE]
            block, block_bytes = gguf.GGML_QUANT_SIZES[kind]
            self.tensors[ENGRAM_TABLE] = RemoteRows(
                engram_url, offset, shape[-1] // block * block_bytes, kind
            )
            self.engram_source = engram_url
        expected = int(self.meta.get("split.tensors.count", len(self.tensors)))
        if len(self.tensors) != expected:
            raise SystemExit(
                f"the parts hold {len(self.tensors)} tensors, the header states {expected}"
            )

    def shared_part(self, first: Path, count: int) -> Path | None:
        """The GGUF beside the first part, named as no split's part, whose
        header makes it the last part, as DeepSeek-V4.1's Engram tables."""
        expected = self.meta.get("split.tensors.count")
        for path in sorted(first.parent.glob("*.gguf")):
            if "-of-" in path.name:
                continue
            fields = gguf.GGUFReader(path).fields
            values = {
                name: field_value(fields[name])
                for name in fields
                if name.startswith("split.")
            }
            if (
                values.get("split.no") == count - 1
                and values.get("split.count") == count
                and values.get("split.tensors.count") == expected
            ):
                return path
        return None

    def has(self, name: str) -> bool:
        return name in self.tensors

    def tensor(self, name: str) -> np.ndarray:
        tensor = self.tensors[name]
        if tensor.tensor_type in (
            gguf.GGMLQuantizationType.I32,
            gguf.GGMLQuantizationType.I64,
        ):
            return np.asarray(tensor.data)
        return np.asarray(
            gguf.dequantize(tensor.data, tensor.tensor_type), dtype=np.float32
        )

    def rows(self, name: str, rows) -> np.ndarray:
        """Rows of a 2-D tensor, such as the token table's."""
        tensor = self.tensors[name]
        return np.asarray(
            gguf.dequantize(tensor.data[rows], tensor.tensor_type), dtype=np.float32
        )

    def row_count(self, name: str) -> int:
        return int(self.tensors[name].data.shape[0])

    def expert(self, name: str, expert: int) -> np.ndarray:
        tensor = self.tensors[name]
        return np.asarray(
            gguf.dequantize(tensor.data[expert], tensor.tensor_type), dtype=np.float32
        )

    def engram_rows(self, rows: np.ndarray) -> np.ndarray:
        """Rows of the Engram tables, which every Engram layer's heads share."""
        table = self.tensors[ENGRAM_TABLE]
        if isinstance(table, RemoteRows):
            return table.read(rows)
        flat = rows.ravel()
        values = gguf.dequantize(table.data[flat], table.tensor_type)
        return np.asarray(values, np.float32).reshape(*rows.shape, -1)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def ceil_pow2(x: np.ndarray) -> np.ndarray:
    """2 ** ceil(log2(x)), as model.py's power-of-two scales round."""
    mantissa, exponent = np.frexp(x.astype(np.float32))
    return np.ldexp(
        np.float32(1.0), np.where(mantissa == 0.5, exponent - 1, exponent)
    ).astype(np.float32)


def round_e4m3(x: np.ndarray) -> np.ndarray:
    """x on the E4M3 grid (3 mantissa bits, normals from 2**-6, subnormals in
    steps of 2**-9), rounded to nearest even."""
    magnitude = np.abs(x)
    exponent = np.floor(np.log2(np.maximum(magnitude, 2.0**-6)))
    step = np.exp2(exponent - 3)
    return (np.sign(x) * np.round(magnitude / step) * step).astype(np.float32)


def round_e2m1(x: np.ndarray) -> np.ndarray:
    """x on the E2M1 grid (0, 0.5, 1, 1.5, 2, 3, 4, 6), rounded to nearest even."""
    magnitude = np.abs(x)
    step = np.where(magnitude < 2.0, 0.5, np.where(magnitude < 4.0, 1.0, 2.0))
    return (np.sign(x) * np.round(magnitude / step) * step).astype(np.float32)


def fp8_round(x: np.ndarray, block: int = 32) -> np.ndarray:
    """model.py's act_quant(..., inplace=True): E4M3 with a power-of-two scale
    per `block` values."""
    groups = x.reshape(*x.shape[:-1], -1, block)
    amax = np.maximum(np.max(np.abs(groups), axis=-1, keepdims=True), 1e-4)
    scale = ceil_pow2(amax / FP8_MAX)
    return (round_e4m3(np.clip(groups / scale, -FP8_MAX, FP8_MAX)) * scale).reshape(
        x.shape
    )


def fp4_round(x: np.ndarray, block: int, e4m3_scale: bool) -> np.ndarray:
    """model.py's fp4_act_quant(..., inplace=True): E2M1 with a power-of-two
    scale per `block` values, or an E4M3 one."""
    groups = x.reshape(*x.shape[:-1], -1, block)
    amax = np.max(np.abs(groups), axis=-1, keepdims=True)
    if e4m3_scale:
        scale = round_e4m3(np.maximum(amax, FP4_MAX * 2.0**-9) / FP4_MAX)
    else:
        scale = ceil_pow2(np.maximum(amax, FP4_MAX * 2.0**-126) / FP4_MAX)
    return (round_e2m1(np.clip(groups / scale, -FP4_MAX, FP4_MAX)) * scale).reshape(
        x.shape
    )


def rope_angles(layout: Layout, compressed: bool, positions: np.ndarray) -> np.ndarray:
    """The rotation angles of `positions`, [len, rope_dim / 2]: model.py's
    precompute_freqs_cis, with YaRN on the layers that compress."""
    dim = layout.rope_dim
    base = layout.compress_rope_theta if compressed else layout.rope_theta
    freqs = (
        1.0 / (np.float32(base) ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    ).astype(np.float32)
    if compressed:
        original = layout.rope_original

        def corrected(rotations):
            return dim * np.log(original / (rotations * 2 * np.pi)) / (2 * np.log(base))

        low = max(int(np.floor(corrected(layout.beta_fast))), 0)
        high = min(int(np.ceil(corrected(layout.beta_slow))), dim - 1)
        ramp = np.clip(
            (np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1
        )
        smooth = 1 - ramp
        freqs = (freqs / layout.rope_factor * (1 - smooth) + freqs * smooth).astype(
            np.float32
        )
    return np.outer(positions.astype(np.float32), freqs).astype(np.float32)


def rotate(x: np.ndarray, angles: np.ndarray, inverse: bool = False) -> np.ndarray:
    """x with its last rope_dim values rotated as interleaved pairs; angles
    broadcast over x's leading dimensions but the row (axis 0)."""
    x = x.copy()
    dim = angles.shape[-1] * 2
    tail = x[..., -dim:]
    even, odd = tail[..., 0::2], tail[..., 1::2]
    shape = (angles.shape[0],) + (1,) * (x.ndim - 2) + (angles.shape[1],)
    cos, sin = np.cos(angles).reshape(shape), np.sin(angles).reshape(shape)
    if inverse:
        sin = -sin
    rotated = np.empty_like(tail)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    x[..., -dim:] = rotated
    return x


def hc_mixes(
    layout: Layout, h: np.ndarray, fn: np.ndarray, scale: np.ndarray, base: np.ndarray
):
    """The pre, post and Sinkhorn-balanced comb weights of a sublayer from its
    streams h [rows, streams, hidden]: model.py's hc_mixes and hc_split_sinkhorn."""
    n, eps = layout.streams, layout.hc_eps
    flat = h.reshape(h.shape[0], -1)
    rsqrt = 1.0 / np.sqrt(
        np.mean(flat * flat, axis=-1, keepdims=True) + layout.norm_eps
    )
    mixes = (flat @ fn.T) * rsqrt
    pre = sigmoid(mixes[:, :n] * scale[0] + base[:n]) + eps
    post = 2 * sigmoid(mixes[:, n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = (mixes[:, 2 * n :] * scale[2] + base[2 * n :]).reshape(-1, n, n)
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = comb / comb.sum(axis=-1, keepdims=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(layout.sinkhorn_iterations - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def hc_pre(h: np.ndarray, pre: np.ndarray) -> np.ndarray:
    return np.einsum("rs,rsd->rd", pre, h)


def hc_post(
    x: np.ndarray, residual: np.ndarray, post: np.ndarray, comb: np.ndarray
) -> np.ndarray:
    """Stream j: post[j] * x plus the sum over i of comb[i, j] * residual[i]."""
    return post[:, :, None] * x[:, None, :] + np.einsum("rij,rid->rjd", comb, residual)


def engram_hashes(model: Model, tokens: np.ndarray) -> np.ndarray:
    """The table rows of each token, [tokens, engram layers, (ngram - 1) * heads]:
    engram.py's NgramHashState over compressed ids, without image spans, each
    head's rows from its offset in the one table (ple.head_offsets)."""
    layout = model.layout
    a, layers = ARCHITECTURE + ".", len(layout.engram_layers)
    token_map = np.asarray(model.meta[a + "engram.token_map"], np.int64)
    multipliers = np.asarray(model.meta[a + "ple.layer_multipliers"], np.int64).reshape(
        layers, -1
    )  # [layers, ngram]
    primes = np.asarray(model.meta[a + "ple.head_vocab_sizes"], np.int64).reshape(
        layers, -1
    )  # [layers, (ngram - 1) * heads]
    offsets = np.asarray(model.meta[a + "ple.head_offsets"], np.int64).reshape(
        layers, -1
    )
    pad = token_map[layout.engram_pad]
    compressed = token_map[tokens]
    count, heads = len(tokens), layout.engram_heads
    lookback = np.full((count, layout.engram_ngram), pad, dtype=np.int64)
    for shift in range(layout.engram_ngram):
        lookback[shift:, shift] = compressed[: count - shift]
    products = lookback[:, None, :] * multipliers[None, :, :]  # [tokens, layers, ngram]
    rolling, hashes = products[..., 0], []
    for n in range(1, layout.engram_ngram):
        rolling = np.bitwise_xor(rolling, products[..., n])
        hashes.append(rolling[..., None] % primes[None, :, (n - 1) * heads : n * heads])
    return np.concatenate(hashes, axis=-1) + offsets[None]


class LayerWeights:
    """One layer's weights but its routed experts, in fp32."""

    def __init__(self, model: Model, layer: int):
        prefix = f"blk.{layer}."
        self.w = {}
        for name in model.tensors:
            if name.startswith(prefix) and "_exps." not in name:
                self.w[name[len(prefix) :]] = model.tensor(name)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.w[name]

    def get(self, name: str):
        return self.w.get(name)


@dataclass
class Shared:
    """What a forward's layers hand down: model.py's SharedAttentionRuntime,
    over every row of the sequence."""

    compressed: np.ndarray | None = None  # [groups, head_dim], rotated and rounded
    index_keys: np.ndarray | None = None  # [groups, index_dim]
    selection: list | None = None  # per query row, the selected groups
    candidates: np.ndarray | None = None  # [query rows, groups] bool


class Forward:
    """A sequence's forward: every layer's attention input of the rows before
    the new ones, from which their latents are computed again."""

    def __init__(
        self,
        model: Model,
        kv_rounding: bool,
        router_layers: list[int],
        dump_streams: bool = False,
        layers: int | None = None,
    ):
        self.model, self.layout, self.kv_rounding = model, model.layout, kv_rounding
        # The leading layers a run computes; with fewer than all it stops
        # before the head and gives no logits.
        self.layers = self.layout.layers if layers is None else layers
        self.tokens = np.zeros(0, dtype=np.int64)
        self.inputs = [
            np.zeros((0, self.layout.hidden), np.float32)
            for _ in range(self.layout.layers)
        ]
        self.router_layers = router_layers
        # With dump_streams, the first run's streams after each sublayer,
        # [layers, 2, rows, streams, hidden], and each Engram layer's
        # gathered rows, {layer: [rows, width]}, which an engine's layers are
        # compared against.
        self.streams = [] if dump_streams else None
        self.engram_values = {}
        self.logits, self.residual_norms, self.experts, self.expert_weights = (
            [],
            [],
            [],
            [],
        )

    def run(self, new_tokens: np.ndarray, log=print) -> np.ndarray:
        """The logits of the new rows, [new, vocabulary]."""
        layout, model = self.layout, self.model
        start = len(self.tokens)
        self.tokens = np.concatenate([self.tokens, np.asarray(new_tokens, np.int64)])
        rows = np.arange(start, len(self.tokens))
        embed = model.rows("token_embd.weight", self.tokens[rows])
        h = np.repeat(embed[:, None, :], layout.streams, axis=1)
        pre = np.zeros((len(rows), layout.streams), np.float32)
        pre[:, 0] = 1.0
        hashes = (
            engram_hashes(model, self.tokens)[rows] if layout.engram_layers else None
        )
        shared = Shared()
        norms = np.zeros((layout.layers, 2, len(rows), layout.streams), np.float32)
        routes = {}
        for layer in range(self.layers):
            began = time.time()
            w = LayerWeights(model, layer)
            if layer in layout.engram_layers:
                h = self.engram(
                    w, layer, h, hashes[:, layout.engram_layers.index(layer)]
                )
            residual = h
            attn_pre, attn_post, attn_comb = hc_mixes(
                layout,
                h,
                w["hc_attn_fn.weight"],
                w["hc_attn_scale.weight"],
                w["hc_attn_base.weight"],
            )
            x = rms_norm(hc_pre(h, pre), w["attn_norm.weight"], layout.norm_eps)
            self.inputs[layer] = np.concatenate([self.inputs[layer][:start], x])
            h = hc_post(
                self.attention(w, layer, rows, shared), residual, attn_post, attn_comb
            )
            norms[layer, 0] = np.linalg.norm(h, axis=-1)
            if self.streams is not None and start == 0:
                self.streams.append(h.copy())
            residual = h
            ffn_pre, ffn_post, ffn_comb = hc_mixes(
                layout,
                h,
                w["hc_ffn_fn.weight"],
                w["hc_ffn_scale.weight"],
                w["hc_ffn_base.weight"],
            )
            x = rms_norm(hc_pre(h, attn_pre), w["ffn_norm.weight"], layout.norm_eps)
            out, experts, weights = self.moe(w, layer, x)
            if layer in self.router_layers:
                routes[layer] = (experts, weights)
            h = hc_post(out, residual, ffn_post, ffn_comb)
            norms[layer, 1] = np.linalg.norm(h, axis=-1)
            if self.streams is not None and start == 0:
                self.streams.append(h.copy())
            pre = ffn_pre
            log(
                f"layer {layer:2d}: {time.time() - began:5.1f} s, stream norm {norms[layer, 1].mean():.3g}"
            )
        self.experts.append(
            np.stack([routes[k][0] for k in self.router_layers])
            if self.router_layers
            else None
        )
        self.expert_weights.append(
            np.stack([routes[k][1] for k in self.router_layers])
            if self.router_layers
            else None
        )
        if self.layers < layout.layers:
            self.residual_norms.append(norms[: self.layers])
            return None
        x = rms_norm(
            hc_pre(h, pre), model.tensor("output_norm.weight"), layout.norm_eps
        )
        vocabulary = model.row_count("output.weight")
        logits = np.concatenate(
            [
                x @ model.rows("output.weight", slice(at, at + HEAD_ROWS)).T
                for at in range(0, vocabulary, HEAD_ROWS)
            ],
            axis=1,
        )
        self.logits.append(logits)
        self.residual_norms.append(norms)
        return logits

    def engram(
        self, w: LayerWeights, layer: int, h: np.ndarray, rows: np.ndarray
    ) -> np.ndarray:
        """engram.py's Engram: the rows' projection gates a value into each stream."""
        layout = self.layout
        values = self.model.engram_rows(rows).reshape(len(rows), -1)
        if self.streams is not None:
            self.engram_values.setdefault(layer, values)
        kv = values @ w["engram_kv.weight"].T
        key = kv[:, : layout.streams * layout.hidden].reshape(
            len(rows), layout.streams, layout.hidden
        )
        value = kv[:, layout.streams * layout.hidden :]
        # engram_gate: the product of DeepSeek's gate query and key.
        weight = w["engram_gate.weight"]
        eps = layout.norm_eps
        rstd = (
            1.0
            / np.sqrt(np.mean(h * h, axis=-1) + eps)
            / np.sqrt(np.mean(key * key, axis=-1) + eps)
        )
        dot = np.sum(h * weight[None] * key, axis=-1) * rstd * layout.hidden**-0.5
        gate = sigmoid(np.sign(dot) * np.sqrt(np.maximum(np.abs(dot), 1e-6)))
        return h + gate[..., None] * value[:, None, :]

    def latents(self, w: LayerWeights, layer: int) -> np.ndarray:
        """A KV source's compressed latents of every complete group, normalized
        and not rotated: model.py's Compressor."""
        layout, ratio = self.layout, self.layout.compress_ratios[layer]
        x = self.inputs[layer]
        groups = len(x) // ratio
        kv = x[: groups * ratio] @ w["attn_compressor_kv.weight"].T
        if ratio > 1:
            score = x[: groups * ratio] @ w["attn_compressor_gate.weight"].T
            kv, score = kv.reshape(groups, ratio, -1), score.reshape(groups, ratio, -1)
            score = np.exp(score - score.max(axis=1, keepdims=True))
            kv = np.sum(kv * score / score.sum(axis=1, keepdims=True), axis=1)
        return rms_norm(kv, w["attn_compressor_norm.weight"], layout.norm_eps)

    def attention(
        self, w: LayerWeights, layer: int, rows: np.ndarray, shared: Shared
    ) -> np.ndarray:
        layout = self.layout
        ratio = layout.compress_ratios[layer]
        x = self.inputs[layer]
        count = len(x)
        angles = rope_angles(layout, ratio > 0, np.arange(count))
        qr = rms_norm(
            x[rows] @ w["attn_q_a.weight"].T, w["attn_q_a_norm.weight"], layout.norm_eps
        )
        q = rotate(
            (qr @ w["attn_q_b.weight"].T).reshape(
                len(rows), layout.heads, layout.head_dim
            ),
            angles[rows],
        )
        first = max(0, int(rows[0]) - layout.window + 1)
        window = rms_norm(
            x[first:] @ w["attn_kv.weight"].T,
            w["attn_kv_a_norm.weight"],
            layout.norm_eps,
        )
        window = rotate(window, angles[first:])
        if self.kv_rounding:
            window = fp8_round(window)
        if ratio:
            if layer in layout.kv_sources:
                latent = self.latents(w, layer)
                group_positions = np.arange(len(latent)) * ratio
                group_angles = rope_angles(layout, True, group_positions)
                keys = rotate(
                    rms_norm(
                        latent @ w["indexer.attn_k.weight"].T,
                        w["indexer.k_norm.weight"],
                        layout.norm_eps,
                    ),
                    group_angles,
                )
                compressed = rotate(latent, group_angles)
                if self.kv_rounding:
                    keys = fp4_round(keys, 32, e4m3_scale=False)
                    compressed = fp4_round(compressed, 16, e4m3_scale=True)
                shared.compressed, shared.index_keys = compressed, keys
            if layer in layout.index_sources:
                shared.selection = self.select(
                    w, layer, rows, qr, x[rows], angles[rows], shared
                )
        scale = layout.head_dim**-0.5
        sinks = w["attn_sinks.weight"]
        out = np.zeros((len(rows), layout.heads, layout.head_dim), np.float32)
        for r, row in enumerate(rows):
            keys = window[
                max(0, int(row) - layout.window + 1) - first : int(row) + 1 - first
            ]
            if ratio and len(shared.selection[r]):
                keys = np.concatenate([keys, shared.compressed[shared.selection[r]]])
            scores = (q[r] @ keys.T) * scale  # [heads, keys]
            peak = np.maximum(scores.max(axis=-1, keepdims=True), sinks[:, None])
            p = np.exp(scores - peak)
            p /= p.sum(axis=-1, keepdims=True) + np.exp(sinks[:, None] - peak)
            out[r] = p @ keys
        out = rotate(out, angles[rows], inverse=True)
        out = out.reshape(len(rows), layout.o_groups, -1)
        if self.kv_rounding:
            out = fp8_round(out)
        # attn_output_a: [groups * output rank, group width], each group's rows
        # in turn.
        groups = w["attn_output_a.weight"].reshape(layout.o_groups, layout.o_rank, -1)
        projected = np.einsum("rgd,god->rgo", out, groups).reshape(len(rows), -1)
        return projected @ w["attn_output_b.weight"].T

    def select(self, w, layer, rows, qr, x, angles, shared: Shared) -> list:
        """Each query row's compressed groups: model.py's Indexer and
        select_candidate_blocks."""
        layout, ratio = self.layout, self.layout.compress_ratios[layer]
        groups = len(shared.index_keys)
        q = rotate(
            (qr @ w["indexer.attn_q_b.weight"].T).reshape(
                len(rows), layout.index_heads, layout.index_dim
            ),
            angles,
        )
        if self.kv_rounding:
            q = fp4_round(q, 32, e4m3_scale=False)
        weights = (x @ w["indexer.proj.weight"].T) * (
            layout.index_dim**-0.5 * layout.index_heads**-0.5
        )
        scores = np.einsum("rhd,gd->rhg", q, shared.index_keys)
        scores = np.sum(
            np.maximum(scores, 0) * weights[:, :, None], axis=1
        )  # [rows, groups]
        reachable = (rows + 1) // ratio
        scores[np.arange(groups)[None, :] >= reachable[:, None]] = -np.inf
        if layer == layout.candidate_source:
            shared.candidates = self.candidates(scores, reachable)
        elif 0 <= layout.candidate_source < layer and shared.candidates is not None:
            scores = np.where(shared.candidates, scores, -np.inf)
        selection = []
        for r in range(len(rows)):
            finite = np.flatnonzero(np.isfinite(scores[r]))
            if len(finite) > layout.index_topk:
                finite = finite[
                    np.argpartition(-scores[r, finite], layout.index_topk - 1)[
                        : layout.index_topk
                    ]
                ]
            selection.append(np.sort(finite))
        return selection

    def candidates(self, scores: np.ndarray, reachable: np.ndarray) -> np.ndarray:
        layout = self.layout
        size = layout.candidate_block_size
        groups = scores.shape[1]
        blocks = -(-groups // size)
        padded = np.full((len(scores), blocks * size), -np.inf, np.float32)
        padded[:, :groups] = scores
        block_scores = padded.reshape(len(scores), blocks, size).max(axis=-1)
        last = (reachable - 1) // size
        for r in range(len(scores)):
            if last[r] >= 0:
                block_scores[r, last[r]] = np.inf
        keep = np.zeros_like(block_scores, dtype=bool)
        for r in range(len(scores)):
            top = np.argsort(-block_scores[r], kind="stable")[
                : min(layout.candidate_blocks, blocks)
            ]
            keep[r, top[block_scores[r, top] > -np.inf]] = True
        return np.repeat(keep, size, axis=1)[:, :groups]

    def moe(self, w: LayerWeights, layer: int, x: np.ndarray):
        layout = self.layout
        scores = np.sqrt(softplus(x @ w["ffn_gate_inp.weight"].T))
        biased = scores + w["exp_probs_b.bias"]
        experts = np.argsort(-biased, axis=-1, kind="stable")[:, : layout.experts_used]
        weights = np.take_along_axis(scores, experts, axis=-1)
        if layout.normalize_routes and layout.experts_used > 1:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * layout.route_scale
        limit = layout.swiglu_limit[layer]
        out = np.zeros_like(x)
        for expert in np.unique(experts):
            rows, slot = np.nonzero(experts == expert)
            gate = self.model.expert(f"blk.{layer}.ffn_gate_exps.weight", expert)
            up = self.model.expert(f"blk.{layer}.ffn_up_exps.weight", expert)
            down = self.model.expert(f"blk.{layer}.ffn_down_exps.weight", expert)
            out[rows] += swiglu(x[rows], gate, up, down, limit, weights[rows, slot])
        out += swiglu(
            x,
            w["ffn_gate_shexp.weight"],
            w["ffn_up_shexp.weight"],
            w["ffn_down_shexp.weight"],
            layout.swiglu_limit_shared[layer],
        )
        return out, experts.astype(np.int32), weights.astype(np.float32)


def swiglu(x, gate, up, down, limit: float, weights=None) -> np.ndarray:
    """model.py's Expert: the up branch clamped to [-limit, limit], the gate
    to at most limit."""
    g, u = x @ gate.T, x @ up.T
    if limit > 0:
        u, g = np.clip(u, -limit, limit), np.minimum(g, limit)
    y = silu(g) * u
    if weights is not None:
        y = y * weights[:, None]
    return y @ down.T


def read_tokens(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.int64).ravel()
    return np.array([int(t) for t in path.read_text().split()], dtype=np.int64)


def layer_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("gguf", type=Path, help="the GGUF's first part")
    parser.add_argument(
        "tokens", type=Path, help=".npy or whitespace-separated token ids"
    )
    parser.add_argument(
        "out", type=Path, help="the directory the reference is written to"
    )
    parser.add_argument(
        "--generate", type=int, default=0, help="tokens to add greedily"
    )
    parser.add_argument(
        "--kv-rounding", choices=("reference", "none"), default="reference"
    )
    parser.add_argument(
        "--router-layers",
        type=layer_list,
        default=None,
        help="layers whose routes are dumped (default: first, middle, last)",
    )
    parser.add_argument(
        "--engram-url",
        help="the Engram file's URL, read with range requests when it is not beside the shards",
    )
    parser.add_argument(
        "--dump-streams",
        action="store_true",
        help="also write the prompt's streams after every sublayer and its Engram rows",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="a tokenizer.json to print the generated text with",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="compute the prompt's first N layers alone: their streams, no logits",
    )
    args = parser.parse_args()
    if args.layers is not None and args.generate:
        parser.error("--layers computes no logits to generate from")

    began = time.time()
    model = Model(args.gguf, args.engram_url)
    layout = model.layout
    router_layers = (
        args.router_layers
        if args.router_layers is not None
        else [0, layout.layers // 2, layout.layers - 1]
    )
    tokens = read_tokens(args.tokens)
    if args.layers is not None:
        router_layers = [k for k in router_layers if k < args.layers]
    forward = Forward(
        model,
        args.kv_rounding == "reference",
        router_layers,
        args.dump_streams,
        args.layers,
    )
    logits = forward.run(tokens)
    for step in range(args.generate):
        token = int(np.argmax(logits[-1]))
        print(f"generated {step + 1}/{args.generate}: {token}", flush=True)
        logits = forward.run(np.array([token]))
    generated = forward.tokens[len(tokens) :]
    if args.tokenizer is not None and len(generated):
        from tokenizers import Tokenizer

        print(
            "text:",
            repr(
                Tokenizer.from_file(str(args.tokenizer)).decode(
                    generated.tolist(), skip_special_tokens=False
                )
            ),
        )

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "tokens.npy", forward.tokens.astype(np.uint32))
    if forward.logits:
        np.save(
            args.out / "logits.npy", np.concatenate(forward.logits).astype(np.float32)
        )
    np.save(
        args.out / "residual_norms.npy", np.concatenate(forward.residual_norms, axis=2)
    )
    if forward.streams:
        streams = np.stack(forward.streams).astype(np.float32)
        np.save(
            args.out / "streams.npy",
            streams.reshape(forward.layers, 2, *streams.shape[1:]),
        )
        for layer, values in forward.engram_values.items():
            np.save(args.out / f"engram_values_{layer}.npy", values.astype(np.float32))
    if router_layers:
        np.save(args.out / "experts.npy", np.concatenate(forward.experts, axis=1))
        np.save(
            args.out / "expert_weights.npy",
            np.concatenate(forward.expert_weights, axis=1),
        )
    reference = {
        "gguf": [str(p) for p in model.paths],
        "engram": model.engram_source,
        "gguf_py": metadata.version("gguf"),
        "layout": asdict(layout),
        "prompt_tokens": len(tokens),
        "generated": generated.tolist(),
        "kv_rounding": args.kv_rounding,
        "layers": forward.layers,
        "router_layers": router_layers,
        "seconds": round(time.time() - began, 1),
    }
    (args.out / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")
    print(f"wrote {args.out} in {reference['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
