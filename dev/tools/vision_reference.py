#!/usr/bin/env python3
"""fp32 reference for the vision tower of an installed model: Qwen3.5's or
Inkling 2's.

This is the executable specification the native Metal encoder is graded
against. It reads the vision weights of an installed model from the source the
engine loads (``config.json`` gives the language width): the
``vision_tower.*`` tensors of an MLX checkpoint's ``vision/`` shards,
a checkpoint's ``model.visual.*`` in the same place, or the ``v.*`` and
``mm.*`` tensors of a GGUF projector, ``vision/mmproj.gguf``, bound as
``VisionLoader`` binds them. It encodes resized uint8 RGB pixels into
language-space embeddings entirely in numpy fp32: patchify and normalize,
patch embedding plus the bilinear (align-corners) resample of the learned
48x48 position table, 27 pre-norm blocks with 2D rotary attention, and the
2x2 spatial merger. Token order is spatial-merge-block-major throughout.

Inkling 2's tower (a clip inkling2v projector, whose matrices may be Q8_0)
encodes padded uint8 RGB pixels of 20-pixel patches: patch embedding plus the
bilinear (half-pixel) resample of its learned 32x32 position table, a class
token in front, a LayerNorm, 23 pre-norm blocks with LayerScale and 2D rotary
attention over interleaved pairs, and the 2x2 merge into its two-layer
projector (encode_inkling).

    python dev/tools/vision_reference.py MODEL_ROOT OUT_DIR

writes a small deterministic parity fixture (pixels, grid, fp32 embeddings)
consumed by ``dev/tests/engine/vision_encoder_test.mm``. The fixtures in
``dev/tests/fixtures/vision-parity/`` are the default image, 96x128 for a
Qwen tower and 120x160 for Inkling's, but the 35B's, which is ``--height 192
--width 192``.
"""

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from install import gguf  # noqa: E402

PATCH = 16
MERGE = 2
NORM_EPS = 1e-6
ROPE_THETA = 10000.0
# Inkling 2's tower: 20-pixel patches, LayerNorms of epsilon 1e-5, and the
# pixel value its preprocessing pads with, -1 on the 0-255 scale.
INKLING_PATCH = 20
INKLING_NORM_EPS = 1e-5
INKLING_PAD = -1.0


@dataclass(frozen=True)
class VisionLayout:
    """The tower's geometry, mirroring ``ops::VisionLayout``."""

    depth: int = 27
    hidden: int = 1152
    patch_dim: int = 1536
    intermediate: int = 4304
    merged_hidden: int = 4608
    out_hidden: int = 5120
    heads: int = 16
    position_grid_side: int = 48


def tensor_names(layout: VisionLayout, namespace: str = "vision_tower."):
    """(reference name, checkpoint tensor name) of every weight the tower
    reads: MLX's ``vision_tower.*`` or the original ``model.visual.*``."""

    def pair(reference, mlx):
        yield reference + ".weight", namespace + mlx + ".weight"
        yield reference + ".bias", namespace + mlx + ".bias"

    yield from pair("patch_embed", "patch_embed.proj")
    yield "pos_embed", namespace + "pos_embed.weight"
    for block in range(layout.depth):
        prefix = f"blocks.{block}."
        yield from pair(prefix + "norm1", prefix + "norm1")
        yield from pair(prefix + "qkv", prefix + "attn.qkv")
        yield from pair(prefix + "proj", prefix + "attn.proj")
        yield from pair(prefix + "norm2", prefix + "norm2")
        yield from pair(prefix + "fc1", prefix + "mlp.linear_fc1")
        yield from pair(prefix + "fc2", prefix + "mlp.linear_fc2")
    yield from pair("merger.norm", "merger.norm")
    yield from pair("merger.fc1", "merger.linear_fc1")
    yield from pair("merger.fc2", "merger.linear_fc2")


def gguf_names(layout: VisionLayout):
    """(reference name, mmproj tensor name) of every weight the tower reads,
    as VisionLoader binds a clip qwen3vl_merger projector. The patch embedding
    is one [hidden, channel, patch-row, patch-col] weight per temporal frame,
    ``v.patch_embd.weight`` and ``v.patch_embd.weight.1``."""

    def pair(reference, gguf):
        yield reference + ".weight", gguf + ".weight"
        yield reference + ".bias", gguf + ".bias"

    yield "patch_embed.weight", "v.patch_embd.weight"
    yield "patch_embed.weight.1", "v.patch_embd.weight.1"
    yield "patch_embed.bias", "v.patch_embd.bias"
    yield "pos_embed", "v.position_embd.weight"
    for block in range(layout.depth):
        prefix, gguf = f"blocks.{block}.", f"v.blk.{block}."
        yield from pair(prefix + "norm1", gguf + "ln1")
        yield from pair(prefix + "qkv", gguf + "attn_qkv")
        yield from pair(prefix + "proj", gguf + "attn_out")
        yield from pair(prefix + "norm2", gguf + "ln2")
        yield from pair(prefix + "fc1", gguf + "ffn_up")
        yield from pair(prefix + "fc2", gguf + "ffn_down")
    yield from pair("merger.norm", "v.post_ln")
    yield from pair("merger.fc1", "mm.0")
    yield from pair("merger.fc2", "mm.2")


def bf16_to_f32(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


def safetensors_names(directory: Path) -> set:
    """The tensor names of directory's safetensors shards."""
    names = set()
    for shard in sorted(directory.glob("*.safetensors")):
        with shard.open("rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            names.update(json.loads(file.read(length)))
    names.discard("__metadata__")
    return names


# ggml type -> (name, element dtype); the ones a projector's tensors may be.
# Q8_0 (Inkling's) is blocks of an F16 scale and 32 int8 values.
GGUF_TYPES = {
    0: ("F32", np.float32),
    1: ("F16", np.float16),
    8: ("Q8_0", np.uint8),
    30: ("BF16", np.uint16),
}


def dequantize_q8_0(raw: np.ndarray) -> np.ndarray:
    blocks = raw.reshape(-1, 34)
    scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    values = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (scales * values).reshape(-1)


def read_gguf_tensors(path: Path, names, projector: str = "qwen3vl_merger") -> dict:
    """The fp32 values of the named tensors of a GGUF file, F32, F16, BF16 or
    Q8_0. GGUF lists a tensor's dimensions innermost first; each value is
    returned row-major, in the reversed shape."""
    header = gguf.Metadata(path, tensors=True)
    # Qwen's projectors state their type as clip.projector_type, Inkling's
    # as clip.vision.projector_type.
    if header.values.get("general.architecture") != "clip" or projector not in (
        header.values.get("clip.projector_type"),
        header.values.get("clip.vision.projector_type"),
    ):
        raise ValueError(f"{path} is not a clip {projector} projector")
    # The data section starts at the first multiple of the alignment past
    # the header.
    alignment = header.values.get("general.alignment", 32)
    data = (len(header.raw) + alignment - 1) // alignment * alignment
    found = {}
    for name in names:
        if name not in header.tensors:
            continue
        dims, kind = header.extents[name], header.tensors[name]
        if kind not in GGUF_TYPES:
            raise ValueError(f"unsupported vision tensor type {kind}: {name}")
        label, dtype = GGUF_TYPES[kind]
        count = int(np.prod(dims))
        offset = data + header.offsets[name]
        if label == "Q8_0":
            raw = np.fromfile(path, dtype=dtype, count=count // 32 * 34, offset=offset)
            values = dequantize_q8_0(raw)
        else:
            raw = np.fromfile(path, dtype=dtype, count=count, offset=offset)
            values = bf16_to_f32(raw) if label == "BF16" else raw.astype(np.float32)
        found[name] = values.reshape(dims[::-1])
    missing = sorted(set(names) - set(found))
    if missing:
        raise ValueError(f"{path} lacks vision tensors: {', '.join(missing[:3])}")
    return found


def read_tensors(directory: Path, names) -> dict:
    """The fp32 values of the named tensors among directory's safetensors
    shards, BF16, F16 or F32."""
    found = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with shard.open("rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(length))
        data = np.memmap(shard, dtype=np.uint8, mode="r", offset=8 + length)
        for name in names:
            if name in found or name not in header:
                continue
            record = header[name]
            begin, end = record["data_offsets"]
            raw = np.asarray(data[begin:end])
            if record["dtype"] == "BF16":
                values = bf16_to_f32(raw.view(np.uint16))
            elif record["dtype"] == "F16":
                values = raw.view(np.float16).astype(np.float32)
            elif record["dtype"] == "F32":
                values = raw.view(np.float32).copy()
            else:
                raise ValueError(
                    f"unsupported vision tensor dtype {record['dtype']}: {name}"
                )
            found[name] = values.reshape(record["shape"])
    missing = sorted(set(names) - set(found))
    if missing:
        raise ValueError(f"{directory} lacks vision tensors: {', '.join(missing[:3])}")
    return found


def load_model(root: Path) -> tuple[VisionLayout, dict]:
    """The installed model's vision layout and its fp32 weights by reference
    name, from ``vision/mmproj.gguf`` when the installation links one, else
    the ``vision/`` safetensors shards; Inkling's tower (inkling_weights)
    when its config names one. Patchify orders a patch [channel,
    frame, patch-row, patch-col]: MLX stores the patch embedding [frame,
    patch-row, patch-col, channel], the original layout [channel, frame,
    patch-row, patch-col], and a GGUF projector one [channel, patch-row,
    patch-col] weight per frame."""
    config = json.loads((root / "config.json").read_text())
    if config.get("vision_config", {}).get("model_type") == "inkling2_vision":
        return INKLING_LAYOUT, inkling_weights(root / "vision" / "mmproj.gguf")
    layout = VisionLayout(out_hidden=config["text_config"]["hidden_size"])
    mmproj = root / "vision" / "mmproj.gguf"
    if mmproj.exists():
        names = dict(gguf_names(layout))
        values = read_gguf_tensors(mmproj, list(names.values()))
        weights = {reference: values[name] for reference, name in names.items()}
        patch = np.stack(
            [weights.pop("patch_embed.weight"), weights.pop("patch_embed.weight.1")],
            axis=2,
        )
    else:
        shards = root / "vision"
        visual_names = "model.visual.patch_embed.proj.weight" in safetensors_names(
            shards
        )
        names = dict(
            tensor_names(layout, "model.visual." if visual_names else "vision_tower.")
        )
        values = read_tensors(shards, list(names.values()))
        weights = {reference: values[name] for reference, name in names.items()}
        patch = weights["patch_embed.weight"]
        if not visual_names:
            patch = patch.transpose(0, 4, 1, 2, 3)
    if patch.shape != (layout.hidden, 3, 2, PATCH, PATCH):
        raise ValueError(f"unexpected patch embedding shape {patch.shape}")
    weights["patch_embed.weight"] = np.ascontiguousarray(patch).reshape(
        layout.hidden, layout.patch_dim
    )
    return layout, weights


def block_major_positions(grid_h: int, grid_w: int):
    """(row, col) per patch token in spatial-merge-block-major order."""
    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    shape = (grid_h // MERGE, MERGE, grid_w // MERGE, MERGE)
    rows = rows.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    cols = cols.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    return rows, cols


def patchify(pixels: np.ndarray) -> tuple[np.ndarray, int, int]:
    """pixels: (H, W, 3) uint8 with sides divisible by 32 -> (tokens, 1536)."""
    height, width, channels = pixels.shape
    if channels != 3 or height % (PATCH * MERGE) or width % (PATCH * MERGE):
        raise ValueError("image must be RGB with sides divisible by 32")
    grid_h, grid_w = height // PATCH, width // PATCH
    normalized = pixels.astype(np.float32) / 127.5 - 1.0
    normalized = normalized.transpose(2, 0, 1)  # (C, H, W)
    patches = normalized.reshape(
        3, grid_h // MERGE, MERGE, PATCH, grid_w // MERGE, MERGE, PATCH
    )
    # (block_row, block_col, in_row, in_col, channel, patch_row, patch_col)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
    patches = patches.reshape(grid_h * grid_w, 3, 1, PATCH, PATCH)
    patches = np.broadcast_to(patches, (grid_h * grid_w, 3, 2, PATCH, PATCH))
    return np.ascontiguousarray(patches).reshape(grid_h * grid_w, -1), grid_h, grid_w


def interpolated_positions(table: np.ndarray, grid_h: int, grid_w: int, side: int):
    rows, cols = block_major_positions(grid_h, grid_w)

    def taps(index, size):
        src = index * (side - 1) / max(size - 1, 1)
        lower = np.floor(src)
        offsets = np.arange(2)
        tap = np.clip(lower[:, None] + offsets, 0, side - 1).astype(np.int64)
        weight = np.clip(1.0 - np.abs(src[:, None] - lower[:, None] - offsets), 0, None)
        return tap, weight

    h_taps, h_weights = taps(rows, grid_h)
    w_taps, w_weights = taps(cols, grid_w)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return (table[indices] * weights[:, :, None].astype(np.float32)).sum(axis=1)


def rope_tables(grid_h: int, grid_w: int, head_dim: int):
    rows, cols = block_major_positions(grid_h, grid_w)
    frequencies = 1.0 / (
        ROPE_THETA
        ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2))
    )
    half = np.concatenate(
        [rows[:, None] * frequencies[None, :], cols[:, None] * frequencies[None, :]],
        axis=1,
    ).astype(np.float32)
    angles = np.concatenate([half, half], axis=1)
    return np.cos(angles), np.sin(angles)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def layer_norm(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + NORM_EPS) * weight + bias


def linear(x, weight, bias):
    return x @ weight.T + bias


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def gelu_erf(x):
    from math import erf

    return 0.5 * x * (1.0 + np.vectorize(erf)(x * 0.7071067811865476))


def encode(layout: VisionLayout, weights: dict, pixels: np.ndarray) -> np.ndarray:
    """Returns (tokens / 4, out_hidden) fp32 language-space embeddings."""
    patches, grid_h, grid_w = patchify(pixels)
    tokens = grid_h * grid_w
    head_dim = layout.hidden // layout.heads
    x = linear(patches, weights["patch_embed.weight"], weights["patch_embed.bias"])
    x = x + interpolated_positions(
        weights["pos_embed"], grid_h, grid_w, layout.position_grid_side
    )
    cos, sin = rope_tables(grid_h, grid_w, head_dim)
    scale = head_dim**-0.5
    for block in range(layout.depth):
        p = f"blocks.{block}."
        normalized = layer_norm(
            x, weights[p + "norm1.weight"], weights[p + "norm1.bias"]
        )
        qkv = linear(normalized, weights[p + "qkv.weight"], weights[p + "qkv.bias"])
        qkv = qkv.reshape(tokens, 3, layout.heads, head_dim)
        queries = qkv[:, 0] * cos[:, None, :] + rotate_half(qkv[:, 0]) * sin[:, None, :]
        keys = qkv[:, 1] * cos[:, None, :] + rotate_half(qkv[:, 1]) * sin[:, None, :]
        values = qkv[:, 2]
        scores = np.einsum("qhd,khd->hqk", queries, keys) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        context = np.einsum("hqk,khd->qhd", probabilities, values)
        context = context.reshape(tokens, layout.hidden)
        x = x + linear(context, weights[p + "proj.weight"], weights[p + "proj.bias"])
        normalized = layer_norm(
            x, weights[p + "norm2.weight"], weights[p + "norm2.bias"]
        )
        hidden = gelu_tanh(
            linear(normalized, weights[p + "fc1.weight"], weights[p + "fc1.bias"])
        )
        x = x + linear(hidden, weights[p + "fc2.weight"], weights[p + "fc2.bias"])
    merged = layer_norm(x, weights["merger.norm.weight"], weights["merger.norm.bias"])
    merged = merged.reshape(tokens // (MERGE * MERGE), layout.merged_hidden)
    merged = linear(merged, weights["merger.fc1.weight"], weights["merger.fc1.bias"])
    merged = gelu_erf(merged)
    return linear(merged, weights["merger.fc2.weight"], weights["merger.fc2.bias"])


# Inkling 2's tower, as ops::kInklingVisionLayout states it.
INKLING_LAYOUT = VisionLayout(
    depth=23,
    hidden=1024,
    patch_dim=3 * INKLING_PATCH * INKLING_PATCH,
    intermediate=4096,
    merged_hidden=4096,
    out_hidden=2304,
    heads=16,
    position_grid_side=32,
)


def inkling_weights(mmproj: Path) -> dict:
    """The fp32 weights of a clip inkling2v projector by its tensor names.
    The patch embedding is [hidden, channel, patch-row, patch-col], which
    patchify_inkling orders a patch's values by."""
    layout = INKLING_LAYOUT
    names = [
        "v.patch_embd.weight",
        "v.position_embd.weight",
        "v.class_embd",
        "v.class_position_embd.weight",
        "v.pre_ln.weight",
        "v.pre_ln.bias",
        "mm.1.weight",
        "mm.2.weight",
    ]
    for block in range(layout.depth):
        prefix = f"v.blk.{block}."
        for name in (
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_out",
            "ffn_up",
            "ffn_down",
            "ln1",
            "ln2",
        ):
            names += [prefix + name + ".weight", prefix + name + ".bias"]
        names += [prefix + "ls1.weight", prefix + "ls2.weight"]
    weights = read_gguf_tensors(mmproj, names, "inkling2v")
    weights["v.patch_embd.weight"] = weights["v.patch_embd.weight"].reshape(
        layout.hidden, layout.patch_dim
    )
    return weights


def patchify_inkling(pixels: np.ndarray, content: tuple[int, int] | None = None):
    """pixels: (H, W, 3) uint8 with sides divisible by 40 -> (patches, 1200)
    in spatial-merge-block order, each patch's values [channel, patch-row,
    patch-col], normalized x / 127.5 - 1. Pixels past content, the image's
    (height, width) before its padding, take the pad value, as Inkling's
    preprocessing pads them."""
    height, width, channels = pixels.shape
    align = INKLING_PATCH * MERGE
    if channels != 3 or height % align or width % align:
        raise ValueError("image must be RGB with sides divisible by 40")
    values = pixels.astype(np.float32)
    if content is not None:
        values[content[0] :, :, :] = INKLING_PAD
        values[:, content[1] :, :] = INKLING_PAD
    normalized = values / 127.5 - 1.0
    grid_h, grid_w = height // INKLING_PATCH, width // INKLING_PATCH
    patches = normalized.transpose(2, 0, 1).reshape(
        3, grid_h // MERGE, MERGE, INKLING_PATCH, grid_w // MERGE, MERGE, INKLING_PATCH
    )
    # (block_row, block_col, in_row, in_col, channel, patch_row, patch_col)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
    return np.ascontiguousarray(patches).reshape(grid_h * grid_w, -1), grid_h, grid_w


def inkling_positions(table: np.ndarray, grid_h: int, grid_w: int, side: int):
    """The learned side x side table resampled to the grid as F.interpolate's
    bilinear mode without aligned corners (half-pixel centres, edges
    clamped), per patch in spatial-merge-block order."""
    rows, cols = block_major_positions(grid_h, grid_w)

    def taps(index, size):
        src = (index + 0.5) * side / size - 0.5
        lower = np.floor(src)
        offsets = np.arange(2)
        tap = np.clip(lower[:, None] + offsets, 0, side - 1).astype(np.int64)
        weight = np.clip(1.0 - np.abs(src[:, None] - lower[:, None] - offsets), 0, None)
        return tap, weight

    h_taps, h_weights = taps(rows, grid_h)
    w_taps, w_weights = taps(cols, grid_w)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return (table[indices] * weights[:, :, None].astype(np.float32)).sum(axis=1)


def inkling_rope(grid_h: int, grid_w: int, head_dim: int):
    """cos and sin per token, the class token's first at (0, 0), then each
    patch's at its 1-based (row, column): the first half of a head rotates
    by the column, the second by the row, each over interleaved pairs at the
    frequencies theta^(-2i / (head_dim / 2))."""
    rows, cols = block_major_positions(grid_h, grid_w)
    rows = np.concatenate([[0], rows + 1]).astype(np.float32)
    cols = np.concatenate([[0], cols + 1]).astype(np.float32)
    quarter = head_dim // 4
    frequencies = 1.0 / (
        ROPE_THETA ** (np.arange(quarter, dtype=np.float32) * 2 / (head_dim // 2))
    )
    angles = np.concatenate(
        [cols[:, None] * frequencies[None, :], rows[:, None] * frequencies[None, :]],
        axis=1,
    ).astype(np.float32)
    angles = np.repeat(angles, 2, axis=1)
    return np.cos(angles), np.sin(angles)


def rotate_pairs(x):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return np.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)


def inkling_layer_norm(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + INKLING_NORM_EPS) * weight + bias


def encode_inkling(weights: dict, pixels: np.ndarray, content=None) -> np.ndarray:
    """Returns (patches / 4, 2304) fp32 language-space embeddings, row-major
    over the merged grid."""
    layout = INKLING_LAYOUT
    patches, grid_h, grid_w = patchify_inkling(pixels, content)
    head_dim = layout.hidden // layout.heads
    x = patches @ weights["v.patch_embd.weight"].T
    x = x + inkling_positions(
        weights["v.position_embd.weight"], grid_h, grid_w, layout.position_grid_side
    )
    cls = weights["v.class_embd"] + weights["v.class_position_embd.weight"]
    x = np.concatenate([cls[None, :], x], axis=0)
    tokens = x.shape[0]
    x = inkling_layer_norm(x, weights["v.pre_ln.weight"], weights["v.pre_ln.bias"])
    cos, sin = inkling_rope(grid_h, grid_w, head_dim)
    scale = head_dim**-0.5
    for block in range(layout.depth):
        p = f"v.blk.{block}."

        def project(values, name):
            return (
                values @ weights[p + name + ".weight"].T + weights[p + name + ".bias"]
            )

        normalized = inkling_layer_norm(
            x, weights[p + "ln1.weight"], weights[p + "ln1.bias"]
        )
        heads = (tokens, layout.heads, head_dim)
        queries = project(normalized, "attn_q").reshape(heads)
        keys = project(normalized, "attn_k").reshape(heads)
        values = project(normalized, "attn_v").reshape(heads)
        queries = queries * cos[:, None, :] + rotate_pairs(queries) * sin[:, None, :]
        keys = keys * cos[:, None, :] + rotate_pairs(keys) * sin[:, None, :]
        scores = np.einsum("qhd,khd->hqk", queries, keys) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        context = np.einsum("hqk,khd->qhd", probabilities, values).reshape(
            tokens, layout.hidden
        )
        x = x + weights[p + "ls1.weight"] * project(context, "attn_out")
        normalized = inkling_layer_norm(
            x, weights[p + "ln2.weight"], weights[p + "ln2.bias"]
        )
        hidden = gelu_erf(project(normalized, "ffn_up"))
        x = x + weights[p + "ls2.weight"] * project(hidden, "ffn_down")
    merged = x[1:].reshape(tokens // (MERGE * MERGE), layout.merged_hidden)
    merged = gelu_erf(merged @ weights["mm.1.weight"].T)
    return merged @ weights["mm.2.weight"].T


def fixture_image(height: int = 96, width: int = 128) -> np.ndarray:
    """A deterministic image with structure along both axes and all channels."""
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    red = 127.5 + 120.0 * np.sin(rows / 9.0) * np.cos(cols / 13.0)
    green = 255.0 * ((rows // 16 + cols // 16) % 2)
    blue = np.broadcast_to(255.0 * cols / (width - 1), (height, width))
    return np.clip(np.stack([red, green, blue], axis=-1), 0, 255).astype(np.uint8)


def write_fixture(root: Path, out: Path, height=None, width=None) -> None:
    layout, weights = load_model(root)
    inkling = layout == INKLING_LAYOUT
    patch = INKLING_PATCH if inkling else PATCH
    pixels = fixture_image(
        height or (120 if inkling else 96), width or (160 if inkling else 128)
    )
    if inkling:
        expected = encode_inkling(weights, pixels).astype(np.float32)
    else:
        expected = encode(layout, weights, pixels).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.txt").write_text(
        f"{pixels.shape[0] // patch} {pixels.shape[1] // patch}\n"
    )
    (out / "pixels.bin").write_bytes(pixels.tobytes())
    (out / "expected_fp32.bin").write_bytes(expected.tobytes())
    print(
        f"wrote parity fixture for grid {pixels.shape[0] // patch}x"
        f"{pixels.shape[1] // patch}: {expected.shape} embeddings"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    args = parser.parse_args()
    write_fixture(args.model, args.output, args.height, args.width)
