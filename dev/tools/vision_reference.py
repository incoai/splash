#!/usr/bin/env python3
"""fp32 reference for the packed Qwen3.5 vision tower.

This is the executable specification the native Metal encoder is graded
against. It reads a runtime package's ``vision/model.bin`` (the padded engine
layout that ``runtime/model/QwenVision.cpp`` maps directly) and encodes
resized uint8 RGB pixels into language-space embeddings entirely in numpy
fp32: patchify and normalize, patch embedding plus the bilinear
(align-corners) resample of the learned 48x48 position table, 27 pre-norm
blocks with 2D rotary attention, and the 2x2 spatial merger. Token order is
spatial-merge-block-major throughout.

    python dev/tools/vision_reference.py PACKAGE_ROOT OUT_DIR

writes a small deterministic parity fixture (pixels, grid, fp32 embeddings)
consumed by ``dev/tests/engine/vision_encoder_test.mm``.
"""

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from install import legacy  # noqa: E402

MAGIC = b"MDFV0001"
PATCH = 16
MERGE = 2
NORM_EPS = 1e-6
ROPE_THETA = 10000.0


@dataclass(frozen=True)
class VisionLayout:
    """The packed tower's geometry, mirroring ``ops::VisionLayout``."""

    depth: int = 27
    hidden: int = 1152
    patch_dim: int = 1536
    intermediate: int = 4304
    padded_intermediate: int = 4352
    merged_hidden: int = 4608
    out_hidden: int = 5120
    heads: int = 16
    position_grid_side: int = 48


def sections(layout: VisionLayout):
    """(section name, shape) in the order the engine reads the pack."""

    def affine(name, out_size, in_size):
        yield name + ".weight", (out_size, in_size)
        yield name + ".bias", (out_size,)

    def norm(name):
        yield name + ".weight", (layout.hidden,)
        yield name + ".bias", (layout.hidden,)

    yield from affine("patch_embed", layout.hidden, layout.patch_dim)
    yield "pos_embed", (layout.position_grid_side**2, layout.hidden)
    for block in range(layout.depth):
        prefix = f"blocks.{block}."
        yield from norm(prefix + "norm1")
        yield from affine(prefix + "qkv", 3 * layout.hidden, layout.hidden)
        yield from affine(prefix + "proj", layout.hidden, layout.hidden)
        yield from norm(prefix + "norm2")
        yield from affine(prefix + "fc1", layout.padded_intermediate, layout.hidden)
        yield from affine(prefix + "fc2", layout.hidden, layout.padded_intermediate)
    yield from norm("merger.norm")
    yield from affine("merger.fc1", layout.merged_hidden, layout.merged_hidden)
    yield from affine("merger.fc2", layout.out_hidden, layout.merged_hidden)


def bf16_to_f32(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


def load_pack(root: Path) -> tuple[VisionLayout, dict]:
    """The package's vision layout and its fp32 weights by section name."""
    config = json.loads((root / "tokenizer" / "config.json").read_text())
    layout = VisionLayout(out_hidden=config["text_config"]["hidden_size"])
    path = root / "vision" / "model.bin"
    data = np.memmap(path, dtype=np.uint8, mode="r")
    magic, depth, kind = struct.unpack("<8sII", data[:16].tobytes())
    if magic != MAGIC or depth != layout.depth or kind != 0:
        raise ValueError(f"invalid vision pack header: {path}")
    weights = {}
    offset = 16
    for name, shape in sections(layout):
        count = math.prod(shape)
        offset = -(-offset // legacy.ALIGNMENT) * legacy.ALIGNMENT
        words = np.asarray(data[offset : offset + count * 2]).view(np.uint16)
        weights[name] = bf16_to_f32(words).reshape(shape)
        offset += count * 2
    if -(-offset // legacy.ALIGNMENT) * legacy.ALIGNMENT != data.size:
        raise ValueError(f"unexpected vision pack size: {path}")
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
    intermediate = layout.intermediate
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
        # The pack pads the MLP width to 4352 with zero rows/columns; the
        # padded activations are exactly zero and contribute nothing.
        hidden = linear(
            normalized,
            weights[p + "fc1.weight"][:intermediate],
            weights[p + "fc1.bias"][:intermediate],
        )
        hidden = gelu_tanh(hidden)
        x = x + linear(
            hidden,
            weights[p + "fc2.weight"][:, :intermediate],
            weights[p + "fc2.bias"],
        )
    merged = layer_norm(x, weights["merger.norm.weight"], weights["merger.norm.bias"])
    merged = merged.reshape(tokens // (MERGE * MERGE), layout.merged_hidden)
    merged = linear(merged, weights["merger.fc1.weight"], weights["merger.fc1.bias"])
    merged = gelu_erf(merged)
    return linear(merged, weights["merger.fc2.weight"], weights["merger.fc2.bias"])


def fixture_image(height: int = 96, width: int = 128) -> np.ndarray:
    """A deterministic image with structure along both axes and all channels."""
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    red = 127.5 + 120.0 * np.sin(rows / 9.0) * np.cos(cols / 13.0)
    green = 255.0 * ((rows // 16 + cols // 16) % 2)
    blue = np.broadcast_to(255.0 * cols / (width - 1), (height, width))
    return np.clip(np.stack([red, green, blue], axis=-1), 0, 255).astype(np.uint8)


def write_fixture(root: Path, out: Path, height: int = 96, width: int = 128) -> None:
    layout, weights = load_pack(root)
    pixels = fixture_image(height, width)
    expected = encode(layout, weights, pixels).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.txt").write_text(
        f"{pixels.shape[0] // PATCH} {pixels.shape[1] // PATCH}\n"
    )
    (out / "pixels.bin").write_bytes(pixels.tobytes())
    (out / "expected_fp32.bin").write_bytes(expected.tobytes())
    print(
        f"wrote parity fixture for grid {pixels.shape[0] // PATCH}x"
        f"{pixels.shape[1] // PATCH}: {expected.shape} embeddings"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()
    write_fixture(args.package, args.output, args.height, args.width)
