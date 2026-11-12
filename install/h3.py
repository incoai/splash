"""MiniMax-H3's recipe (conversion.py): the units that convert one variant
of it into the engine's images (runtime/video/model), in an entry's
layout:

  dit-<variant>/         the DiT (VDN-H3): per block, FL2VA's base weights,
                         the OpenVDN stage's linear branch and its adapters'
                         LoRAs merged the OpenVDN way (merge_lora_state into
                         the bf16 model: W = bf16(W + bf16(scale (B @ A))),
                         delta by delta, in adapter order; the adaLN and
                         final-layer weights too), the int8 linears rotated by a
                         Hadamard transform in groups of 128 (splitmix64
                         signs, seed HAD_SEED) and quantized per output row
                         (absmax / 127; QUANT_VIDEO_W8 of
                         runtime/metal/abi/QuantFormat.h), the small tensors
                         in the kernels' layouts (VDN_LAYOUT); the fp16 q / k
                         / v of the QKFIX blocks; the final layer's globals;
                         the patch projections (io); the request-independent
                         prep inputs, whose token refiner merges its LoRAs
                         the OpenVDN way; the adaLN tables of the variant's
                         schedule, which the engine tabulates (splash
                         video-adaln) from FL2VA's adaLN bases, read first
                         into scratch; and manifest.json
  vae/                   the video VAE's decoder, its linears W8 like the
                         DiT's (post_quant_conv folded into x_embedder in
                         fp64, the RoPE table of a 7 x 16 x 16 tile-clip),
                         and its encoder in fp16
  audio/, preview/       the audio VAE's decoder (weight norm folded, fp16
                         hi / lo pairs) and TAEH3's, fp16

Every file, manifests included (with source paths relative to their
repositories), is byte for byte the reference conversion's.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re

import numpy as np

from . import conversion, models
from .families import H3
from .video import RECIPE

# Each variant's OpenVDN stage, and whether its turbo adapter applies.
STAGES = {
    "turbo": ("stage-dmd-step-250", True),
    "default": ("stage-b-step-2000", False),
}
TRANSFORMER = "FL2VA/transformer/"
VIDEO_VAE = "FL2VA/video_vae/source/"
AUDIO_VAE = "FL2VA/audio_vae/"
# TAEH3 is published in its GitHub repository, not on the Hub: its file at
# this commit, checked by its SHA-256 (22.7 MB, read whole).
TAEH3_REPO = "github.com/madebyollin/taehv"
TAEH3_COMMIT = "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"
TAEH3 = "safetensors/taeh3.safetensors"
TAEH3_SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
# The sources a conversion reads besides the release's and OpenVDN's commits.
PINNED = {TAEH3_REPO: TAEH3_COMMIT}
BLOCKS = 50
# The recipe's QKFIX blocks, whose q / k / v run in fp16: FL2VA's blocks of
# QK gain above QKFIX_GAIN, which its manifest checks (check_qkfix).
QKFIX, QKFIX_GAIN = (45, 48, 49), 20.0
H, HD, D, F = 56, 128, 5376, 14336
HAD_GROUP = 128
# The rows of a weight converted at once, which bounds a conversion's
# temporaries; every step is row by row, so any split writes the same bytes.
ROWS = 256
HAD_SEED = 0x48335138  # "H3Q8"; video/model/Hadamard.hpp's
VDN_LAYOUT = {
    # canonical suffix (blocks.i.) : (kind, description)
    "attn.qkv": ("q8", "[21504, 5376] rows [q | k | v], head-major; to_q/k/v + LoRA; rotated (signs 0..5375)"),
    "attn.gates": ("q8", "[256, 5376] rows [beta 56 | softmax gate 56 | zero 16 | og_down 128]; rotated"),
    "attn.out": ("q8", "[5376, 7168] to_out.0 + LoRA; rotated (signs 0..7167)"),
    "attn.out_linear": ("q8", "[5376, 7168] to_out_linear; rotated with signs 7168..14335"),
    "mlp.fc1": ("q8", "[28672, 5376] rows [gate | value] (ff.net.0.proj swapped) (+ turbo LoRA); rotated"),
    "mlp.fc2": ("q8", "[5376, 14336] ff.net.2 (+ turbo LoRA); rotated"),
    "norm1": ("BF16", "[5376]"), "norm2": ("BF16", "[5376]"),
    "attn.q_norm": ("BF16", "[128] attn.norm_q"), "attn.k_norm": ("BF16", "[128] attn.norm_k"),
    "adaln_proj.weight": ("BF16", "[96768, 2688] base weight + turbo LoRA, merged in bf16 as OpenVDN does"),
    "adaln_proj.bias": ("BF16", "[96768]; ada table = (W silu(temb) + b).view(3, 6, 5376)"),
    "vdn.sg_bias": ("F32", "[56] softmax_gate.up.bias"),
    "vdn.og_up": ("F16", "[7168, 128] output_gate.up.weight"), "vdn.og_bias": ("F32", "[7168]"),
    "vdn.alpha_down": ("F32", "[128, 5376] alpha.down.weight, rotated like x"),
    "vdn.alpha_up": ("F32", "[7168, 128]"), "vdn.dt_bias": ("F32", "[7168]"), "vdn.A_log": ("F32", "[56]"),
    "vdn.k_sp": ("F16", "[25, 7168] short_conv.k_sp tap-major"), "vdn.k_tm": ("F16", "[5, 7168]"),
    "vdn.v_sp": ("F16", "[25, 7168]"), "vdn.v_tm": ("F16", "[5, 7168]"),
    "vdn.norm": ("F32", "[128] linear_attention.norm.weight"),
}  # fmt: skip
# FL2VA's names of the base weights a block reads, by their diffusers names
# (the q / k / v rows of its fused, per-head interleaved qkv_proj aside).
NATIVE = {
    "attn.norm_q.weight": "attn.q_norm.weight",
    "attn.norm_k.weight": "attn.k_norm.weight",
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "attn.to_out.0.weight": "attn.out_proj.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
    "ff.net.2.weight": "mlp.fc2.weight",
}
IO_KEYS = (
    "video_patch_proj.weight", "video_patch_proj.bias", "audio_patch_proj.weight",
    "audio_patch_proj.bias", "final_layer.norm.weight", "final_layer.video_out.weight",
    "final_layer.video_out.bias", "final_layer.audio_out.weight", "final_layer.audio_out.bias",
)  # fmt: skip
# The final layer's weights (its adaLN weight with the turbo LoRA merged).
GLOBALS = {
    "norm_out.linear.weight": "final_layer.adaln_proj.linear.weight",
    "norm_out.linear.bias": "final_layer.adaln_proj.linear.bias",
    "norm_out.norm.weight": "final_layer.norm.weight",
}
PREP = (
    "time_embedder.",
    "condition_proj.",
    "token_refiner.",
    "final_layer.adaln_proj.",
)
# The video VAE's decoder: layers, width, heads, head width, MLP width, and
# the rows of its RoPE table.
VAE_LAYERS, VAE_D, VAE_H, VAE_HD = 36, 2048, 32, 64
VAE_S = 7 * 16 * 16 + 5
VAE_I8 = ("qkv", "o", "1", "2")
ENC_CH = (128, 256, 256, 512, 512, 1024)
ENC_SPACE_DOWN = (2, 2, 2, 2, 1, 1)
ENC_TIME_DOWN = (1, 2, 2, 1, 1, 1)
# cos | sin [2][23][8] of the decoder's RoPE angles, 2 pi x (t of 7, then h
# and w of 16 grid points) x 8 frequencies, as the MLX port's ViTRotaryPosEmbed
# computes them on the CPU (MLX's float32 cos and sin, which numpy's do not
# match bit for bit).
VAE_ROPE = """
C50fP6Fdfr9TDge+KU0TP03DWz9oWXQ/4Et8P67Tfj/npWa/seLdvmr01z7kkU0/F61vP8LMej8L
Wn4/eHp/P4DcY74RQwg/WNhXP9MQcz/P4ns/V7J+P22Wfz+c3n8/AACAPwAAgD8AAIA/AACAPwAA
gD8AAIA/AACAPwAAgD+A3GO+EUMIP1jYVz/TEHM/z+J7P1eyfj9tln8/nN5/P+elZr+x4t2+avTX
PuSRTT8XrW8/wsx6Pwtafj94en8/BJ0fP6Bdfr9LDge+KU0TP07DWz9oWXQ/4Et8P67Tfj9fg2w/
mkV8v8pbk7523v8+MttUP1AVcj9Okns/yZh+Pxrvwz7Zq3a/d2wyvcOOHT/mW18/7IV1P9urfD8h
8n4/C+/Dvha7Qb8XPlA+yhw4PxJ8aD/xe3g/gp19P70+fz9eg2y/kDjPvuYn4D7oFE8/Ty1wP+X1
ej8dZ34/m35/P2CDbL82ncw8lTclP5wEYj93Y3Y/jfJ8P4wIfz+3sX8/F+/Dvu9X5j6ZN1A/fY1w
P78Uez/qcH4/toF/Pw7Yfz8W78M+nNhJP3Jwbj8TZ3o/vDl+Pzxwfz+H0n8/nvF/P16DbD+CyHk/
Uwd+P0Rgfz95zX8/BfB/P/L6fz9n/n8/XoNsP4LIeT9TB34/RGB/P3nNfz8F8H8/8vp/P2f+fz8W
78M+nNhJP3Jwbj8TZ3o/vDl+Pzxwfz+H0n8/nvF/Pxfvw77vV+Y+mTdQP32NcD+/FHs/6nB+P7aB
fz8O2H8/YINsvzadzDyVNyU/nARiP3djdj+N8nw/jAh/P7exfz9eg2y/kDjPvuYn4D7oFE8/Ty1w
P+X1ej8dZ34/m35/Pwvvw74Wu0G/Fz5QPsocOD8SfGg/8Xt4P4KdfT+9Pn8/Gu/DPtmrdr93bDK9
w44dP+ZbXz/shXU/26t8PyHyfj9fg2w/mkV8v8pbk7523v8+MttUP1AVcj9Okns/yZh+PxkmSD/l
Cue9f8N9v0BgUb9eTQO/nrOYvoWNLb7C1sO9/CXePhi2Zr8sHWi/bpEYv2bns76tWE2+4gbovZSr
gr3hlHm/+rhYv9GlCb/Lr6C+ndc2vrBlzr25Zmi9orwCvQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAA4ZR5P/q4WD/RpQk/y6+gPp3XNj6wZc49uWZoPaK8Aj38Jd6+GLZmPywdaD9ukRg/
ZuezPq1YTT7iBug9lKuCPR4mSL8FC+c9f8N9Pz9gUT9dTQM/nbOYPoSNLT7B1sM9Fe/DPiAfLj4E
K3W/hb1dv9k5Dr98g6a+maQ9vm4i1r1dg2w/5vSIvsvBf79oxUm/jyz6vnL5kL4umCS+8qq5vWCD
bD9YWCe/iaZ6v1nfMb+SWta+DU52vsVxC74yKp29Fu/DPooYar94KWa/foIWv0A2sb45Lkq+rWrk
vZihgL0Q78O+jut/v3qMQ7+KbvC+PvqKvnapHb7BzbG9JSVIvV6DbL8qoWS/Vu8Uv2Ypr77BxUe+
8qvhvX8pfr0a/Q69XoNsvyh2Hb/DXLq+9PVUvnK38L1MlIe9XI8YvdSbq7wW78O+Ikxgvhys/b3J
5o695s4gvdXjtLyFdEu8ldPkuxbvwz4iTGA+HKz9Pcnmjj3mziA91eO0PIV0SzyV0+Q7XoNsPyh2
HT/DXLo+9PVUPnK38D1MlIc9XI8YPdSbqzxeg2w/KqFkP1bvFD9mKa8+wcVHPvKr4T1/KX49Gv0O
PRDvwz6O638/eoxDP4pu8D4++oo+dqkdPsHNsT0lJUg9Fu/DvooYaj94KWY/foIWP0A2sT45Lko+
rWrkPZihgD1gg2y/WFgnP4mmej9Z3zE/klrWPg1Odj7FcQs+MiqdPV2DbL/m9Ig+y8F/P2jFST+P
LPo+cvmQPi6YJD7yqrk9Fe/DviAfLr4EK3U/hb1dP9k5Dj98g6Y+maQ9Pm4i1j0=
"""
RATES = (5, 5, 2, 2, 2, 2, 2)
KERNELS = (9, 9, 4, 4, 4, 4, 4)
RES_KERNELS = (3, 7, 11)
DILATIONS = (1, 3, 5)


# ---------------------------------------------------------------- tensors
def to_f32(dtype, array):
    if dtype == "BF16":
        bits = np.asarray(array, np.uint16)
        return np.left_shift(bits, 16, dtype=np.uint32).view(np.float32)
    return np.asarray(array, dtype=np.float32)


def bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class Tensors(dict):
    """A unit's tensors, name -> (dtype, array), as fp32 by get."""

    def get(self, name):
        return to_f32(*self[name])


def as_bytes(array):
    """array's bytes, a uint8 view of it (a copy only if it is not
    contiguous)."""
    return np.ascontiguousarray(array).reshape(-1).view(np.uint8)


def safetensors(tensors, meta=None):
    """A safetensors file of tensors, name -> (dtype, array): keys sorted,
    header padded to 8 bytes."""
    return b"".join(safetensors_parts(tensors, meta))


def safetensors_parts(tensors, meta=None):
    """safetensors(tensors, meta) as the buffers it joins, which a unit
    writes in order (conversion.py) without joining them."""
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        dtype, array = tensors[name]
        data = as_bytes(array)
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blobs.append(data)
    if meta:
        header["__metadata__"] = meta
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    return [len(encoded).to_bytes(8, "little"), encoded, *blobs]


def manifest(value):
    return (json.dumps(value, indent=1) + "\n").encode()


class Blob:
    """A flat binary of tensors, each 256-byte aligned, from offset 0: the
    buffers it is written from (parts), size bytes."""

    def __init__(self):
        self.parts, self.size = [], 0

    def put(self, array):
        offset = self.size + -self.size % 256
        data = as_bytes(array)
        self.parts += [bytes(offset - self.size), data]
        self.size = offset + len(data)
        return offset


class Adapter:
    """A PEFT adapter of an OpenVDN stage: its spec and its tensors' file."""

    def __init__(self, repo, directory):
        self.name = directory.rpartition("/")[2]
        self.spec = models.read_json(repo.file(f"{directory}/adapter_spec.json"))
        self.source = conversion.Source(repo, f"{directory}/adapter_model.safetensors")

    def names(self, prefix):
        return [n for n in self.source.tensors if n.startswith(prefix)]

    def targets(self):
        """The base weight each LoRA module targets: 'X.attn.orig.to_q'
        targets 'X.attn.to_q.weight'."""
        return {
            m.group(1).replace(".attn.orig.", ".attn.") + ".weight"
            for n in self.source.tensors
            if (m := re.match(r"^(.*)\.lora_A\.([^.]+)\.weight$", n))
        }

    def deltas(self, tensors):
        """Base weight name -> (A [r, in], B [out, r], scale) of this
        adapter's modules among tensors."""
        cfg = self.spec["config"]
        out = {}
        for name in tensors:
            m = re.match(r"^(.*)\.lora_A\.([^.]+)\.weight$", name)
            if not m or m.group(2) != self.name:
                continue
            module = m.group(1)
            a = tensors.get(name)
            b = tensors.get(f"{module}.lora_B.{m.group(2)}.weight")
            alpha = cfg["alpha"]
            for k, v in (cfg.get("alpha_pattern", {}) or {}).items():
                # PEFT: the key matches the module name or its suffix
                pattern = re.escape(k).replace("\\*", "[^.]*")
                if re.fullmatch(pattern, module) or re.search(
                    r"(^|\.)" + pattern + "$", module
                ):
                    alpha = v
            out[module.replace(".attn.orig.", ".attn.") + ".weight"] = (
                a,
                b,
                alpha / a.shape[0],
            )
        return out


def all_deltas(adapters, tensors):
    """Base weight name -> (A, B, scale) of each adapter, in order (summed,
    not overwritten)."""
    out = {}
    for adapter in adapters:
        for name, delta in adapter.deltas(tensors).items():
            out.setdefault(name, []).append(delta)
    return out


# ---------------------------------------------------------------- Hadamard rotation + int8
def splitmix64(x):
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def had_signs(k, seed=HAD_SEED):
    x = (np.uint64(seed) << np.uint64(32)) + np.arange(k, dtype=np.uint64)
    return np.where((splitmix64(x) >> np.uint64(63)) != 0, -1, 1).astype(np.int8)


def hadamard128():
    i = np.arange(HAD_GROUP)
    a = i[:, None] & i[None, :]
    pc = np.zeros_like(a)
    for b in range(7):
        pc += (a >> b) & 1
    return np.where(pc % 2 == 0, 1, -1).astype(np.int64)


def had_rotate(w, s=None):
    """W'[n, g] = (1/sqrt(128)) H (s_g (.) W[n, g]) per 128-group of input
    columns (float64 in and out); s, the signs, default to had_signs."""
    w = np.asarray(w, dtype=np.float64)
    n, k = w.shape
    s = had_signs(k) if s is None else s
    s = np.asarray(s, dtype=np.float64).reshape(k // HAD_GROUP, HAD_GROUP)
    hn = hadamard128().astype(np.float64) / np.sqrt(128.0)
    out = np.empty_like(w)
    for r0 in range(0, n, 4096):  # bounded temporaries for the 28672-row fc1
        g = w[r0 : r0 + 4096].reshape(-1, k // HAD_GROUP, HAD_GROUP)
        out[r0 : r0 + 4096] = ((g * s) @ hn.T).reshape(-1, k)
    return out


def q8_rows(n, rows, signs=None, keep=None):
    """The int8 rows and fp32 row scales of an n-row linear, rotated
    (had_rotate, signs) and quantized (quant_rows) ROWS rows at a time:
    rows(r0, r1) is its fp32 rows r0:r1, and keep(r0, w) sees each."""
    q = scale = None
    for r0 in range(0, n, ROWS):
        w = rows(r0, min(r0 + ROWS, n))
        if keep:
            keep(r0, w)
        qr, sr = quant_rows(had_rotate(w, signs).astype(np.float32))
        if q is None:
            q, scale = np.empty((n, qr.shape[1]), np.int8), np.empty(n, np.float32)
        q[r0 : r0 + len(qr)], scale[r0 : r0 + len(qr)] = qr, sr
    return q, scale


def quant_rows(w):
    w = np.ascontiguousarray(w, dtype=np.float32)
    am = np.abs(w).max(axis=1)
    inv = np.where(am > 0, np.float32(127.0) / np.where(am > 0, am, 1), np.float32(0))
    q = np.clip(w * inv.astype(np.float32)[:, None], np.float32(-127), np.float32(127))
    return np.rint(q).astype(np.int8), (am / np.float32(127.0)).astype(np.float32)


def swap_halves(w):
    """Diffusers ff.net.0.proj rows [value | gate] <-> FL2VA's [gate | value]."""
    n = w.shape[0] // 2
    return np.concatenate([w[n:], w[:n]], axis=0)


# ---------------------------------------------------------------- DiT
class BaseSource:
    """A block's base weights under diffusers names, from FL2VA's native
    tensors."""

    def __init__(self, tensors):
        self.native = tensors

    def get(self, name):
        m = re.match(r"^transformer_blocks\.(\d+)\.(.*)$", name)
        b, rest = int(m.group(1)), m.group(2)
        if rest in NATIVE:
            return self.native.get(f"blocks.{b}.{NATIVE[rest]}")
        if rest == "ff.net.0.proj.weight":
            return swap_halves(self.native.get(f"blocks.{b}.mlp.fc1.weight"))
        p = "qkv".index(re.fullmatch(r"attn\.to_([qkv])\.weight", rest).group(1))
        arr = self.native[f"blocks.{b}.attn.qkv_proj.weight"][
            1
        ]  # rows [head][q|k|v][128]
        rows = (
            np.arange(H)[:, None] * 3 * HD + p * HD + np.arange(HD)[None, :]
        ).reshape(-1)
        return to_f32("BF16", arr[rows])

    def rows(self, name):
        """rows(r0, r1), name's fp32 rows r0:r1 (a NATIVE weight, or q / k /
        v)."""
        m = re.match(r"^transformer_blocks\.(\d+)\.(.*)$", name)
        b, rest = int(m.group(1)), m.group(2)
        if rest in NATIVE:
            dtype, arr = self.native[f"blocks.{b}.{NATIVE[rest]}"]
            return lambda r0, r1: to_f32(dtype, arr[r0:r1])
        p = "qkv".index(re.fullmatch(r"attn\.to_([qkv])\.weight", rest).group(1))
        dtype, arr = self.native[f"blocks.{b}.attn.qkv_proj.weight"]

        def get(r0, r1):
            r = np.arange(r0, r1)  # of [head][128] -> rows [head][q|k|v][128]
            return to_f32(dtype, arr[r // HD * 3 * HD + p * HD + r % HD])

        return get


def bf16_merge(w, d):
    """OpenVDN's merge_lora_state of one delta into a bf16 weight:
    w.add_(delta.to(bf16)) on a bf16 parameter, i.e. bf16(w + bf16(d)), as
    fp32 values."""
    return to_f32("BF16", bf16_bits(w + to_f32("BF16", bf16_bits(d))))


def merged_rows(rows, deltas):
    """rows(r0, r1) of a weight, its fp32 rows rows(r0, r1) gives, with its
    LoRA deltas (A, B, scale) summed in fp32, delta by delta."""

    def get(r0, r1):
        w = rows(r0, r1)
        for a, b, sc in deltas:
            w = w + np.float32(sc) * (b[r0:r1] @ a)
        return w

    return get


def vdn_merged_rows(rows, deltas):
    """rows(r0, r1) of a bf16 weight, its fp32 rows rows(r0, r1) gives, with
    its OpenVDN LoRA deltas (A, B, scale) merged as OpenVDN merges them, delta
    by delta in adapter order: each delta, computed in fp32, is rounded to
    bf16 and added to the bf16 weight, the sum rounded to bf16. A delta below
    half a bf16 ulp of an element leaves it as it is, as in the official
    model."""

    def get(r0, r1):
        w = rows(r0, r1)
        for a, b, sc in deltas:
            w = bf16_merge(w, np.float32(sc) * (b[r0:r1] @ a))
        return w

    return get


def stacked(parts, n):
    """rows(r0, r1) of parts' rows stacked, n rows each."""

    def get(r0, r1):
        return np.concatenate([
            part(max(r0 - k * n, 0), min(r1 - k * n, n))
            for k, part in enumerate(parts) if r0 < (k + 1) * n and r1 > k * n
        ])  # fmt: skip

    return get


def vdn_block(src, br, i, deltas, qkv16=None):
    """Block i's tensors, canonical name -> (dtype, array): rotated int8 and
    scales, small tensors in kernel layouts; qkv16, of a QKFIX block, gets
    its merged q / k / v in fp16. The linears are merged, rotated and
    quantized ROWS rows at a time."""
    t = "transformer_blocks.%d." % i
    la = t + "attn.linear_attention."
    out = {}
    sig5, sig14 = had_signs(D), had_signs(F)

    def q8(canon, n, rows, signs, keep=None):
        q, s = q8_rows(n, rows, signs, keep)
        out["blocks.%d.%s.qweight" % (i, canon)] = ("I8", q)
        out["blocks.%d.%s.wscale" % (i, canon)] = ("F32", s)

    def lora(name):
        return vdn_merged_rows(src.rows(t + name), deltas.get(t + name, []))

    def whole(w):
        return lambda r0, r1: w[r0:r1]

    def put(canon, dt, a):
        a = np.asarray(a, dtype=np.float32)
        if dt == "BF16":
            out["blocks.%d.%s" % (i, canon)] = ("BF16", bf16_bits(a))
        elif dt == "F16":
            out["blocks.%d.%s" % (i, canon)] = ("F16", a.astype(np.float16))
        else:
            out["blocks.%d.%s" % (i, canon)] = ("F32", a)

    def keep(r0, w):
        qkv16[r0 : r0 + len(w)] = w.astype(np.float16)

    qkv = stacked([lora("attn.to_%s.weight" % p) for p in "qkv"], H * HD)
    q8("attn.qkv", 3 * H * HD, qkv, sig5, keep if qkv16 is not None else None)
    g = np.zeros((256, D), dtype=np.float32)
    g[:56] = br.get(la + "beta_proj.weight")
    g[56:112] = br.get(t + "attn.softmax_gate.up.weight")
    g[128:] = br.get(la + "output_gate.down.weight")
    q8("attn.gates", 256, whole(g), sig5)
    q8("attn.out", D, lora("attn.to_out.0.weight"), sig14[: H * HD])
    q8("attn.out_linear", D, whole(br.get(t + "attn.to_out_linear.weight")),
       sig14[H * HD :])  # fmt: skip
    # fc1 [gate | value]: FL2VA's rows, the LoRAs' [value | gate] rows swapped
    dtype, fc1 = src.native["blocks.%d.mlp.fc1.weight" % i]
    swapped = [
        (a, swap_halves(b), sc)
        for a, b, sc in deltas.get(t + "ff.net.0.proj.weight", [])
    ]
    q8(
        "mlp.fc1",
        2 * F,
        vdn_merged_rows(lambda r0, r1: to_f32(dtype, fc1[r0:r1]), swapped),
        sig5,
    )
    q8("mlp.fc2", D, lora("ff.net.2.weight"), sig14)
    put("norm1", "BF16", src.get(t + "norm1.weight"))
    put("norm2", "BF16", src.get(t + "norm2.weight"))
    put("attn.q_norm", "BF16", src.get(t + "attn.norm_q.weight"))
    put("attn.k_norm", "BF16", src.get(t + "attn.norm_k.weight"))
    put("adaln_proj.bias", "BF16", src.get(t + "adaln_proj.linear.bias"))
    put("vdn.sg_bias", "F32", br.get(t + "attn.softmax_gate.up.bias"))
    put("vdn.og_up", "F16", br.get(la + "output_gate.up.weight"))
    put("vdn.og_bias", "F32", br.get(la + "output_gate.up.bias"))
    put("vdn.alpha_down", "F32", had_rotate(br.get(la + "alpha.down.weight"), sig5))
    put("vdn.alpha_up", "F32", br.get(la + "alpha.up.weight"))
    put("vdn.dt_bias", "F32", br.get(la + "alpha.dt_bias"))
    put("vdn.A_log", "F32", br.get(la + "alpha.A_log"))
    c = H * HD
    for kv in "kv":
        put(
            "vdn.%s_sp" % kv,
            "F16",
            br.get(la + "short_conv.%s_sp.weight" % kv).reshape(c, 25).T,
        )
        put(
            "vdn.%s_tm" % kv,
            "F16",
            br.get(la + "short_conv.%s_tm.weight" % kv).reshape(c, 5).T,
        )
    put("vdn.norm", "F32", br.get(la + "norm.weight"))
    return out


def qkfix_split_dims(gq, gk, n=2):
    """The n dims of the largest |q_norm * k_norm| gain product, with their
    RoPE partners, int32[4], -1 unused."""
    out = []
    for d in np.argsort(
        -np.abs(np.asarray(gq, np.float64) * np.asarray(gk, np.float64)), kind="stable"
    )[:n]:
        d = int(d)
        out.append(d)
        if d < 48:
            out.append(d + 48)
        elif d < 96:
            out.append(d - 48)
    out = sorted(set(out))[:4]
    return out + [-1] * (4 - len(out))


def qk_gain(gq, gk):
    """A block's QK gain, max |q_norm * k_norm|."""
    return float(np.abs(np.asarray(gq, np.float64) * np.asarray(gk, np.float64)).max())


def check_qkfix(i, gain, qkfix, what):
    """Refuses block i of what, of QK gain gain, unless the gain is above
    QKFIX_GAIN exactly when i is one of qkfix, a recipe's QKFIX blocks."""
    if (gain > QKFIX_GAIN) != (i in qkfix):
        raise models.ModelError(
            f"block {i} of {what} has QK gain {gain:.3g}: its QKFIX blocks are not {list(qkfix)}"
        )


def qkfix_block(src, i, qkv16):
    """QKFIX block i's file: its merged fp16 q / k / v [21504, 5376] (qkv16,
    vdn_block's), not rotated, and its split dims."""
    t = "transformer_blocks.%d." % i
    split = qkfix_split_dims(
        src.get(t + "attn.norm_q.weight"), src.get(t + "attn.norm_k.weight")
    )
    tensors = {"blocks.%d.qkfix.wqkv" % i: ("F16", qkv16),
               "blocks.%d.qkfix.split" % i: ("I32", np.array(split, dtype=np.int32))}  # fmt: skip
    return safetensors_parts(tensors, {"format": "h3-qkfix-v1", "block": str(i)})


def dit_manifest(variant, adapters, norms):
    """The DiT's manifest.json; norms holds the QKFIX blocks' q / k norm
    weights, by FL2VA name."""
    stage, turbo = STAGES[variant]
    targets = set().union(*(adapter.targets() for adapter in adapters))
    merged_suffixes = ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
                       "attn.to_out.0.weight", "ff.net.0.proj.weight", "ff.net.2.weight",
                       "adaln_proj.linear.weight")  # fmt: skip
    blocks = {k for k in targets if k.startswith("transformer_blocks.")}
    used = {k for k in blocks if k.endswith(merged_suffixes)}
    used |= targets & {"norm_out.linear.weight"}
    tgt = blocks | {k for k in targets if k.startswith("norm_out")}
    qkfix = {
        "blocks": {},
        "qk": "fp16 x fp16 -> fp32 q/k/v GEMM (merged weights, not rotated), fp32 QK-norm + RoPE epilogue",
        "attn": "fp16 Q.K + exact fp32 dot of the split dims (zeroed in the fp16 Q/K)",
        "base": "all",
        "stage": stage,
        "adapters": [adapter.name for adapter in adapters],
        "lora_merged": sorted(
            k for b in QKFIX for p in "qkv"
            if (k := f"transformer_blocks.{b}.attn.to_{p}.weight") in targets
        ),
    }  # fmt: skip
    for b in QKFIX:
        gq, gk = (norms.get(f"blocks.{b}.attn.{n}_norm.weight") for n in "qk")
        qkfix["blocks"][str(b)] = {
            "file": "qkfix-block%02d.safetensors" % b,
            "split": qkfix_split_dims(gq, gk),
            "gq_max": float(np.abs(gq).max()),
            "gk_max": float(np.abs(gk).max()),
        }
    return manifest({
        "format": RECIPE, "base": "all", "stage": stage,
        "adapters": [f"{stage}/adapters/{adapter.name}" for adapter in adapters],
        "hadamard": {"group": HAD_GROUP, "seed": HAD_SEED, "signs": "splitmix64",
                     "out_linear_signs": "7168..14335 of the 14336 sequence"},
        "quant": "symmetric int8, per-output-row fp32 scale = absmax/127 (after rotation)",
        "layout": {k: list(v) for k, v in VDN_LAYOUT.items()},
        "files": {str(i): "vdn-w8-block%02d.safetensors" % i for i in range(BLOCKS)}
        | {"globals": "vdn-globals.safetensors"},
        "adaln_base_in_blocks": False,
        "adaln_base_source": "base shards: transformer_blocks.N.adaln_proj.linear.weight (bf16, = FL2VA "
                             "blocks.N.adaln_proj.linear.weight), turbo LoRA merged in bf16",
        "lora_merged": sorted(used),
        "lora_targets_not_merged": sorted(tgt - used),
        "qkfix": qkfix,
    })  # fmt: skip


def prep(tensors, adapters, variant):
    """prep.safetensors: FL2VA's time embedder, condition projection, final
    adaLN and token refiner, whose LoRAs merge the OpenVDN way
    (merge_lora_state: W = bf16(W + bf16(scale (B @ A))), delta by delta, in
    adapter order); q / k / v deltas land on the per-head interleaved
    qkv_proj rows, ff.net.0.proj [value; gate] on fc1 [gate; value]. The
    bases are merged in place, ROWS rows at a time."""
    out = {
        k: (dt, a)
        for k, (dt, a) in tensors.items()
        if k.startswith(PREP) and ".lora_" not in k
    }
    for k, (dt, _) in out.items():
        if dt != ("F32" if k.startswith("time_embedder.") else "BF16"):
            raise models.ModelError(f"{k} of MiniMax-H3 is {dt}")
    deltas = {}
    for adapter in adapters:
        for name, (a, b, sc) in adapter.deltas(tensors).items():
            if name == "norm_out.linear.weight":
                deltas.setdefault("final_layer.adaln_proj.linear.weight", []).append(
                    (a, b, sc, None)
                )
                continue
            if not name.startswith("token_refiner.refiner_blocks."):
                continue
            blk, rest = int(name.split(".")[2]), ".".join(name.split(".")[3:])
            base = f"token_refiner.blocks.{blk}."
            # (A, B, scale, the part of q / k / v rows it lands on, else None)
            if rest in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"):
                key, delta = "attn.qkv_proj.weight", (a, b, sc, "qkv".index(rest[8]))
            elif rest == "attn.to_out.0.weight":
                key, delta = "attn.out_proj.weight", (a, b, sc, None)
            elif rest == "ff.net.0.proj.weight":
                key, delta = "mlp.fc1.weight", (a, swap_halves(b), sc, None)
            elif rest == "ff.net.2.weight":
                key, delta = "mlp.fc2.weight", (a, b, sc, None)
            else:
                raise models.ModelError(f"unexpected token refiner LoRA {name}")
            deltas.setdefault(base + key, []).append(delta)
    count = 0
    for k, ds in deltas.items():
        w = out[k][1]
        if not w.flags.writeable:
            w = w.copy()
        for r0 in range(0, len(w), ROWS):
            r1 = min(r0 + ROWS, len(w))
            r = np.arange(r0, r1)
            wb = to_f32("BF16", w[r0:r1])
            for a, b, sc, part in ds:
                if part is None:
                    d = np.float32(sc) * (b[r0:r1] @ a)
                else:  # the per-head interleaved rows of part; 0 elsewhere
                    d = np.zeros((len(r), a.shape[1]), np.float32)
                    mine = r % (3 * HD) // HD == part
                    rows = r[mine] // (3 * HD) * HD + r[mine] % HD
                    d[mine] = np.float32(sc) * (b[rows] @ a)
                wb = bf16_merge(wb, d)
            w[r0:r1] = bf16_bits(wb)
        count += len(ds)
        out[k] = ("BF16", w)
    return safetensors_parts(
        out, {"variant": variant, "refiner_lora_deltas": str(count)}
    )


# ---------------------------------------------------------------- video VAE
def f16(w, name):
    if not float(np.abs(w).max()) < 60000:
        raise models.ModelError(f"MiniMax-H3's {name} overflows fp16")
    return w.astype(np.float16)


def vae_rope():
    """cos, sin [VAE_S][48] fp32: per row (t, h, w) the 8 angles of each
    axis, twice; the 5 suffix rows at position 0."""
    table = np.frombuffer(base64.b64decode(VAE_ROPE), "<f4").reshape(2, 23, 8)
    d, h, w = 7, 16, 16
    idx = np.arange(d * h * w)
    axes = (idx // (h * w), d + (idx // w) % h, d + idx % w)
    out = []
    for at_zero, t in zip((1.0, 0.0), table):
        rows = np.concatenate([t[a] for a in axes], axis=1)
        suffix = np.full((VAE_S - d * h * w, 48), at_zero, np.float32)
        out.append(
            np.concatenate([np.concatenate([rows, rows], axis=1), suffix]).astype(
                np.float32
            )
        )
    return out


def vae_globals(tensors):
    """The decoder's globals, as a part of its blob."""
    get = tensors.get
    blob, g = Blob(), {}
    we = get("decoder.x_embedder.weight").astype(np.float64)
    be = get("decoder.x_embedder.bias").astype(np.float64)
    wpq = get("post_quant_conv.weight").astype(np.float64).reshape(24, 24)
    bpq = get("post_quant_conv.bias").astype(np.float64)
    g["we"] = blob.put((we @ wpq).astype(np.float32))
    g["be"] = blob.put((we @ bpq + be).astype(np.float32))
    g["reg"] = blob.put(get("decoder.register_tokens").reshape(4, VAE_D))
    g["lnw"] = blob.put(get("decoder.norm_out.weight"))
    g["lnb"] = blob.put(get("decoder.norm_out.bias"))
    g["wp"] = blob.put(f16(get("decoder.proj_out.weight"), "proj_out"))
    g["bp"] = blob.put(get("decoder.proj_out.bias"))
    g["cos"], g["sin"] = (blob.put(t) for t in vae_rope())
    g["sg2048"] = blob.put(had_signs(2048))
    g["sg8192"] = blob.put(had_signs(8192))
    return blob, g


def vae_layer(tensors, i):
    """Decoder layer i, as a part of its blob: the linears W8 (rotated,
    int8 rows, fp32 row scales q*), q / k / v rows reordered from per-head
    interleaved (h, part, d) to (part, h, d)."""
    get, p = tensors.get, f"decoder.transformer_blocks.{i}."
    perm = np.array(
        [
            h * 192 + part * 64 + d
            for part in range(3)
            for h in range(VAE_H)
            for d in range(VAE_HD)
        ]
    )
    blob, rec = Blob(), {}
    for nm, w, b in (
        ("qkv", get(p + "attn.to_qkv.weight")[perm], get(p + "attn.to_qkv.bias")[perm]),
        ("o", get(p + "attn.to_out.weight"), get(p + "attn.to_out.bias")),
        ("1", get(p + "ff.w1.weight"), get(p + "ff.w1.bias")),
        ("2", get(p + "ff.w2.weight"), get(p + "ff.w2.bias")),
    ):
        q, s = q8_rows(len(w), lambda r0, r1, w=w: w[r0:r1])
        rec["w" + nm], rec["q" + nm] = blob.put(q), blob.put(s)
        rec["b" + nm] = blob.put(b)
    for key, name in (
        ("s1", "scale1"),
        ("s2", "scale2"),
        ("n1", "norm1.weight"),
        ("n2", "norm2.weight"),
    ):
        rec[key] = blob.put(get(p + name))
    return blob, rec


def vae_encoder(tensors, config, src):
    """vae_enc_f16.bin and manifest_enc.json: every 3D convolution fp16
    [kd][kh][kw][cout_pad][cin_pad] with an fp32 bias (conv_in's 3 inputs
    padded to 32, conv_out's 48 outputs to 64), group norms and quant_conv
    fp32; src names the source in the manifest."""
    if (tuple(config["ch"] * m for m in config["ch_mult"]) != ENC_CH
            or config["num_res_blocks"] != 2
            or tuple(config["space_down"]) != ENC_SPACE_DOWN
            or tuple(config["time_down"]) != ENC_TIME_DOWN):  # fmt: skip
        raise models.ModelError("MiniMax-H3's video VAE encoder has another shape")
    get, blob = tensors.get, Blob()
    man = {"convs": {}, "norms": {}, "src": src}

    def conv(name, cin_pad=None, cout_pad=None):
        w, b = get(f"encoder.{name}.weight"), get(f"encoder.{name}.bias")
        cout, cin, kd, kh, kw = w.shape
        cp, op = cin_pad or cin, cout_pad or cout
        wt = np.zeros((kd, kh, kw, op, cp), np.float32)
        wt[:, :, :, :cout, :cin] = w.transpose(2, 3, 4, 0, 1)
        bt = np.zeros(op, np.float32)
        bt[:cout] = b
        man["convs"][name] = {"w": blob.put(f16(wt, name)), "b": blob.put(bt), "cin": cin, "cin_pad": cp,
                              "cout": cout, "cout_pad": op, "kd": kd, "k": kh}  # fmt: skip

    def norm(name):
        w = get(f"encoder.{name}.weight")
        man["norms"][name] = {
            "w": blob.put(w),
            "b": blob.put(get(f"encoder.{name}.bias")),
            "c": int(w.shape[0]),
        }

    conv("conv_in", cin_pad=32)
    for i in range(len(ENC_CH)):
        for j in range(2):
            p = f"down.{i}.block.{j}"
            norm(p + ".norm1")
            conv(p + ".conv1")
            norm(p + ".norm2")
            conv(p + ".conv2")
            if f"encoder.{p}.nin_shortcut.weight" in tensors:
                conv(p + ".nin_shortcut")
        if ENC_SPACE_DOWN[i] * ENC_TIME_DOWN[i] > 1:
            conv(f"down.{i}.downsample.conv")
    norm("norm_out")
    conv("conv_out", cout_pad=64)
    qw = get("quant_conv.weight").reshape(48, 48)
    man["quant_conv"] = {"w": blob.put(qw), "b": blob.put(get("quant_conv.bias"))}
    man["bytes"] = blob.size
    return blob.parts, manifest(man)


# ---------------------------------------------------------------- audio VAE, TAEH3
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """minimax_h3_mlx.audio_vae.kaiser_sinc_filter1d (alias-free-torch),
    float32 [kernel_size]."""
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = np.kaiser(kernel_size, beta)
    if kernel_size % 2 == 0:
        time = np.arange(-half_size, half_size, dtype=np.float64) + 0.5
    else:
        time = np.arange(kernel_size, dtype=np.float64) - half_size
    filt = 2 * cutoff * window * np.sinc(2 * cutoff * time)
    return (filt / filt.sum()).astype(np.float32)


def audio(tensors, wrapper):
    """audio_dec_f16.bin and its manifest: dec_in_proj and the BigVGAN
    decoder, weight norm folded (fp64), each conv W[tap][Cout][Cin] as fp16
    hi / lo, SnakeBeta [2][C] exp(alpha), 1 / (exp(beta) + 1e-9)."""
    w = {k: to_f32(dt, a) for k, (dt, a) in tensors.items()}
    if bad := [k for k, (dt, _) in tensors.items() if dt != "F32"]:
        raise models.ModelError(f"MiniMax-H3's audio VAE {bad[0]} is not F32")
    blob, man = Blob(), {}

    def put(name, arr):
        offset = blob.put(arr)
        man[name] = {"off": offset, "shape": list(arr.shape), "dtype": str(arr.dtype)}

    def folded(key):
        if key + ".weight" in w:
            return w[key + ".weight"].astype(np.float64)
        g = w[key + ".weight_g"].astype(np.float64)
        v = w[key + ".weight_v"].astype(np.float64)
        return (
            g
            * v
            / np.sqrt((v.reshape(v.shape[0], -1) ** 2).sum(axis=1)).reshape(-1, 1, 1)
        )

    def conv(name, key, transposed=False, bias=True):
        k = folded(key)
        k = k.transpose(2, 1, 0) if transposed else k.transpose(2, 0, 1)
        hi = k.astype(np.float16)
        put(name + ".w", hi)
        put(name + ".wl", (k - hi.astype(np.float64)).astype(np.float16))
        if bias:
            put(name + ".b", w[key + ".bias"].astype(np.float32))

    def snake(name, key):
        a = np.exp(w[key + ".act.alpha"].astype(np.float32))
        b = np.float32(1.0) / (
            np.exp(w[key + ".act.beta"].astype(np.float32)) + np.float32(1e-9)
        )
        put(name, np.stack([a, b]).astype(np.float32))

    put("filter", kaiser_sinc_filter1d(0.5 / 2, 0.6 / 2, 12))
    conv("dec_in", "dec_in_proj")
    conv("conv_pre", "decoder.conv_pre")
    for i in range(len(RATES)):
        conv(f"ups{i}", f"decoder.ups.{i}.0", transposed=True)
        for j in range(len(RES_KERNELS)):
            r = i * len(RES_KERNELS) + j
            for d in range(len(DILATIONS)):
                conv(f"rb{r}.c1.{d}", f"decoder.resblocks.{r}.convs1.{d}")
                conv(f"rb{r}.c2.{d}", f"decoder.resblocks.{r}.convs2.{d}")
            for a in range(2 * len(DILATIONS)):
                snake(f"rb{r}.act{a}", f"decoder.resblocks.{r}.activations.{a}")
    snake("post.act", "decoder.activation_post")
    conv("conv_post", "decoder.conv_post", bias=False)
    put("latents_mean", np.array(wrapper["latents_mean"], np.float32))
    put("latents_std", np.array(wrapper["latents_std"], np.float32))
    man["config"] = {"decoder_dim": 1024, "rates": list(RATES), "kernels": list(KERNELS),
                     "res_kernels": list(RES_KERNELS), "dilations": list(DILATIONS), "latent_channels": 32,
                     "latent_dim": 2048, "sample_rate": int(wrapper.get("sample_rate", 32000)), "hop": 800}  # fmt: skip
    return blob.parts, manifest(man)


def audio_encoder(tensors, wrapper):
    """audio_enc_f16.bin and manifest_enc.json: the audio VAE's DAC encoder
    for the engine's AudioEncoder, weight norm folded (fp64). conv_in, the
    residual units' convs and the k3 output conv as W[tap][Cout][Cin] fp16
    hi / lo; each strided conv (k 2s, stride s) as row pairs W[q][Cout][s Cin]
    (W[q][co][r Cin + ci] = weight[co][ci][q s + r]) fp16 hi / lo; fp32
    biases; Snake1d [2][C] alpha, 1 / (alpha + 1e-9). pre_block, mean_proj
    (the posterior mean; logs_proj is never used) and the latent
    normalisation stay fp32 for the host, linear weights transposed [in][out]
    (.wt)."""
    w = {k: to_f32(dt, a) for k, (dt, a) in tensors.items()}
    if bad := [k for k, (dt, _) in tensors.items() if dt != "F32"]:
        raise models.ModelError(f"MiniMax-H3's audio VAE {bad[0]} is not F32")
    blob, man = Blob(), {}

    def put(name, arr):
        offset = blob.put(arr)
        man[name] = {"off": offset, "shape": list(arr.shape), "dtype": str(arr.dtype)}

    def folded(key):
        g = w[key + ".weight_g"].astype(np.float64)
        v = w[key + ".weight_v"].astype(np.float64)
        return (
            g
            * v
            / np.sqrt((v.reshape(v.shape[0], -1) ** 2).sum(axis=1)).reshape(-1, 1, 1)
        )

    def hilo(name, k):
        hi = k.astype(np.float16)
        put(name + ".w", hi)
        put(name + ".wl", (k - hi.astype(np.float64)).astype(np.float16))

    def conv(name, key):
        hilo(name, folded(key).transpose(2, 0, 1))
        put(name + ".b", w[key + ".bias"].astype(np.float32))

    def down(name, key, s):
        k = folded(key)  # [Cout][Cin][2 s]
        cout, cin, _ = k.shape
        hilo(
            name,
            k.reshape(cout, cin, 2, s).transpose(2, 0, 3, 1).reshape(2, cout, s * cin),
        )
        put(name + ".b", w[key + ".bias"].astype(np.float32))

    def snake(name, key):
        a = w[key + ".alpha"].astype(np.float32).reshape(-1)
        put(
            name,
            np.stack([a, np.float32(1.0) / (a + np.float32(1e-9))]).astype(np.float32),
        )

    def f32(name, key):
        put(name, np.ascontiguousarray(w[key].astype(np.float32)))

    def linear(name, key):
        put(
            name + ".wt",
            np.ascontiguousarray(
                w[key + ".weight"].reshape(w[key + ".weight"].shape[0], -1).T
            ),
        )
        put(name + ".b", w[key + ".bias"].astype(np.float32))

    rates = [int(r) for r in wrapper.get("encoder_rates", (2, 4, 4, 5, 5))]
    conv("enc_in", "encoder.block.0")
    for i, s in enumerate(rates):
        e = f"encoder.block.{i + 1}.block"
        for u in range(3):
            snake(f"blk{i}.u{u}.act1", f"{e}.{u}.block.0")
            conv(f"blk{i}.u{u}.c7", f"{e}.{u}.block.1")
            snake(f"blk{i}.u{u}.act2", f"{e}.{u}.block.2")
            conv(f"blk{i}.u{u}.c1", f"{e}.{u}.block.3")
        snake(f"blk{i}.act", f"{e}.3")
        down(f"blk{i}.down", f"{e}.4", s)
    snake("final.act", f"encoder.block.{len(rates) + 1}")
    conv("enc_out", f"encoder.block.{len(rates) + 2}")
    for n in ("norm1", "norm2", "norm3", "mlp.norm"):
        f32(f"pre.{n}.w", f"pre_block.{n}.weight")
        f32(f"pre.{n}.b", f"pre_block.{n}.bias")
    put("pre.attn.qkv.wt", np.ascontiguousarray(w["pre_block.attn.qkv.weight"].T))
    f32("pre.attn.q_bias", "pre_block.attn.q_bias")
    f32("pre.attn.v_bias", "pre_block.attn.v_bias")
    for n in ("attn.proj", "proj", "mlp.w0", "mlp.w1", "mlp.w2"):
        linear(f"pre.{n}", f"pre_block.{n}")
    linear("mean_proj", "mean_proj")
    put("latents_mean", np.array(wrapper["latents_mean"], np.float32))
    put("latents_std", np.array(wrapper["latents_std"], np.float32))
    man["config"] = {"encoder_dim": int(wrapper.get("encoder_dim", 64)), "rates": rates,
                     "latent_dim": int(wrapper.get("latent_dim", 2048)), "latent_channels": 32,
                     "heads": int(wrapper.get("num_attention_heads", 8)), "sample_rate": 32000, "hop": 800}  # fmt: skip
    return blob.parts, manifest(man)


def preview(tensors):
    """TAEH3's decoder (FastVideo's port): conv3x3 W[src][tap 9][Cout][Cin]
    fp16 (MemBlock conv.0's two sources x and past), tgrow 1x1 W[Cout][Cin],
    up (nearest 2x then 3x3) as 4 output phases x 2x2 taps on the low-res
    input, summed in fp32; decoder.1's inputs padded to 32, decoder.22's
    outputs to 16."""
    w = {k: to_f32(dt, a) for k, (dt, a) in tensors.items()}
    blob, man = Blob(), {}

    def put(name, arr):
        offset = blob.put(arr)
        man[name] = {"off": offset, "shape": list(arr.shape), "dtype": str(arr.dtype)}

    def conv3(name, key, nsrc=1, cin_pad=None, cout_pad=None):
        k = w[key + ".weight"]
        co, ci = k.shape[0], k.shape[1] // nsrc
        k = (
            k.reshape(co, nsrc, ci, 3, 3)
            .transpose(1, 3, 4, 0, 2)
            .reshape(nsrc, 9, co, ci)
        )
        if cin_pad and cin_pad > ci:
            k = np.concatenate(
                [k, np.zeros(k.shape[:3] + (cin_pad - ci,), np.float32)], axis=3
            )
        if cout_pad and cout_pad > co:
            k = np.concatenate(
                [k, np.zeros((nsrc, 9, cout_pad - co, k.shape[3]), np.float32)], axis=2
            )
        put(name + ".w", k.astype(np.float16))
        if (b := w.get(key + ".bias")) is not None:
            if cout_pad and cout_pad > co:
                b = np.concatenate([b, np.zeros(cout_pad - co, np.float32)])
            put(name + ".b", b.astype(np.float32))

    def up(name, key):
        k = w[key + ".weight"]
        comb = {0: [[0], [1, 2]], 1: [[0, 1], [2]]}
        ph = np.zeros((4, 4) + k.shape[:2], np.float32)
        for py in range(2):
            for px in range(2):
                for a in range(2):
                    for bb in range(2):
                        ph[py * 2 + px, a * 2 + bb] = sum(
                            k[:, :, ky, kx] for ky in comb[py][a] for kx in comb[px][bb]
                        )
        put(name + ".w", ph.astype(np.float16))

    conv3("d1", "decoder.1", cin_pad=32)
    for idx in (3, 4, 5, 9, 10, 11, 15, 16, 17):
        conv3(f"m{idx}.0", f"decoder.{idx}.conv.0", nsrc=2)
        conv3(f"m{idx}.2", f"decoder.{idx}.conv.2")
        conv3(f"m{idx}.4", f"decoder.{idx}.conv.4")
    for idx in (7, 13, 19):
        put(f"g{idx}.w", w[f"decoder.{idx}.conv.weight"][:, :, 0, 0].astype(np.float16))
    for idx in (8, 14, 20):
        up(f"u{idx}", f"decoder.{idx}")
    conv3("d22", "decoder.22", cout_pad=16)
    return blob.parts, manifest(man)


# ---------------------------------------------------------------- units
def inputs(variant, release, vdn):
    """The files the variant's conversion reads, REPO/NAME -> the content ID
    the Hub reports (an LFS file's SHA-256, else its Git blob ID), and
    TAEH3's SHA-256; release and vdn as the Hub listed them."""
    stage, turbo = STAGES[variant]
    names = {
        release: [n for n in release.files if n.startswith(TRANSFORMER)
                  and n.endswith((".safetensors", ".index.json"))]
        + [f"{d}{n}" for d in (VIDEO_VAE, AUDIO_VAE) for n in ("model.safetensors", "config.json")],
        vdn: [f"{stage}/linear_branch/model.safetensors"]
        + [f"{stage}/adapters/{a}/{n}" for a in ("default", "turbo")[: 1 + turbo]
           for n in ("adapter_model.safetensors", "adapter_spec.json")],
    }  # fmt: skip
    files = {f"{TAEH3_REPO}/{TAEH3}": TAEH3_SHA256}
    for repo, wanted in names.items():
        if missing := sorted(n for n in wanted if n not in repo.sizes):
            raise models.ModelError(
                f"{repo.name} is missing files MiniMax-H3 needs: {', '.join(missing)}"
            )
        files |= {f"{repo.name}/{n}": repo.sizes[n][1] for n in sorted(wanted)}
    return files


def taeh3():
    """TAEH3's tensors, from its file on GitHub."""
    import httpx

    url = f"https://{TAEH3_REPO}/raw/{TAEH3_COMMIT}/{TAEH3}"
    try:
        data = (
            httpx.get(url, follow_redirects=True, timeout=60).raise_for_status().content
        )
    except httpx.HTTPError as error:
        raise models.ModelError(f"cannot read {url}: {error}") from error
    if hashlib.sha256(data).hexdigest() != TAEH3_SHA256:
        raise models.ModelError(f"{url} is not the TAEH3 this release converts")
    length = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + length])
    header.pop("__metadata__", None)
    return Tensors({
        name: (e["dtype"], np.frombuffer(
            data, conversion.DTYPES[e["dtype"]], count=math.prod(e["shape"]),
            offset=8 + length + e["data_offsets"][0]).reshape(e["shape"]))
        for name, e in header.items()
    })  # fmt: skip


def concatenated(parts):
    """The bytes of parts, (padding, file) pairs, in order, read a chunk at a
    time."""
    for padding, path in parts:
        yield bytes(padding)
        with path.open("rb") as stream:
            while chunk := stream.read(conversion.CHUNK):
                yield chunk


def writes(outputs, convert):
    """run of a unit whose outputs convert(tensors) returns in order."""
    return lambda t: dict(zip(outputs, convert(Tensors(t)), strict=True))


def single(source, match=lambda name: True):
    """reads of a unit: the tensors of source that match."""
    return lambda: [(source, [n for n in source.tensors if match(n)])]


def pinned(variant):
    """The sources the variant's conversion reads at a pinned commit."""
    return PINNED


def decoders(stage, release):
    """The units converting the release's decoders, every variant's: the
    video VAE's decoder (its parts in scratch, then their blob) and encoder,
    the audio VAE's decoder, and TAEH3's (PINNED)."""
    U, out = conversion.Unit, []
    vae = conversion.Source(release, VIDEO_VAE + "model.safetensors")
    audio_vae = conversion.Source(release, AUDIO_VAE + "model.safetensors")
    decoder = ("vae/vae_dec_w8.bin", "vae/manifest_w8.json")
    global_names = ("decoder.x_embedder.", "post_quant_conv.", "decoder.register_tokens",
                "decoder.norm_out.", "decoder.proj_out.")  # fmt: skip
    for number in range(1 + VAE_LAYERS):
        layer = number - 1
        paths = (f".vae/{number:02d}.bin", f".vae/{number:02d}.json")
        prefix = (
            global_names if number == 0 else (f"decoder.transformer_blocks.{layer}.",)
        )

        def part(t, layer=layer):
            blob, offsets = vae_globals(t) if layer < 0 else vae_layer(t, layer)
            return [blob.parts, manifest(offsets)]

        name = "globals" if number == 0 else f"layer {layer}"
        out.append(U(f"video VAE decoder {name}", paths, writes(paths, part),
                     single(vae, lambda n, prefix=prefix: n.startswith(prefix)), consumed_by=decoder[0]))  # fmt: skip

    def blob(_):
        man = {"tag": "w8", "i8": sorted(VAE_I8), "layers": [], "globals": {}, "S": VAE_S,
               "had_seed": HAD_SEED, "src": VIDEO_VAE + "model.safetensors"}  # fmt: skip
        parts, size = [], 0
        for number in range(1 + VAE_LAYERS):
            start = size + -size % 256
            part = stage / f".vae/{number:02d}.bin"
            parts.append((start - size, part))
            size = start + part.stat().st_size
            offsets = json.loads((stage / f".vae/{number:02d}.json").read_text())
            offsets = {k: v + start for k, v in offsets.items()}
            if number:
                man["layers"].append(offsets)
            else:
                man["globals"] = offsets
        man["bytes"] = size
        man["layer_keys"] = list(man["layers"][0])
        return [concatenated(parts), manifest(man)]

    def encoder(t):
        config = models.read_json(release.file(VIDEO_VAE + "config.json"))
        # The source as the published manifest names it: its LFS SHA-256, a
        # Hub cache's blob name.
        name = VIDEO_VAE + "model.safetensors"
        sha = (
            release.sizes[name][1]
            if name in release.sizes
            else release.file(name).resolve().name
        )
        return vae_encoder(t, config, sha)

    encoded = ("vae/vae_enc_f16.bin", "vae/manifest_enc.json")
    audio_files = ("audio/audio_dec_f16.bin", "audio/manifest.json")
    preview_files = ("preview/taeh3_dec_f16.bin", "preview/manifest.json")
    return out + [
        U("video VAE decoder", decoder, writes(decoder, blob), discard=(".vae",)),
        U("video VAE encoder", encoded, writes(encoded, encoder),
          single(vae, lambda n: n.startswith(("encoder.", "quant_conv.")))),
        U("audio VAE decoder", audio_files,
          writes(audio_files, lambda t: audio(t, models.read_json(release.file(AUDIO_VAE + "config.json")))),
          single(audio_vae)),
        U("TAEH3 preview decoder", preview_files, lambda _: dict(zip(preview_files, preview(taeh3())))),
    ]  # fmt: skip


def units(stage, variant, release, vdn, run_engine):
    """The units converting variant's entry into the staging folder stage
    from MiniMax's release, OpenVDN's stages (hub.Repository each, listed)
    and TAEH3 (PINNED); run_engine(arguments) runs the engine."""
    vdn_stage, turbo = STAGES[variant]
    U, dit = conversion.Unit, f"dit-{variant}"
    index = models.read_json(
        release.file(TRANSFORMER + "model.safetensors.index.json")
    )["weight_map"]
    shards = {
        f: conversion.Source(release, TRANSFORMER + f)
        for f in sorted(set(index.values()))
    }
    branch = conversion.Source(vdn, f"{vdn_stage}/linear_branch/model.safetensors")
    adapters = [
        Adapter(vdn, f"{vdn_stage}/adapters/{name}")
        for name in ("default", "turbo")[: 1 + turbo]
    ]

    def base(names):
        """Reads of FL2VA's transformer tensors names, by shard."""
        by_shard = {}
        for name in names:
            by_shard.setdefault(index[name], []).append(name)
        return [(shards[f], n) for f, n in by_shard.items()]

    def lora(match):
        return [(a.source, [n for n in a.source.tensors if match(n)]) for a in adapters]

    def final(t):
        g = {k: ("BF16", bf16_bits(t.get(n))) for k, n in GLOBALS.items()}
        w = g["norm_out.linear.weight"][1]
        for a, b, sc in all_deltas(adapters, t).get("norm_out.linear.weight", []):
            w = bf16_bits(bf16_merge(to_f32("BF16", w), np.float32(sc) * (b @ a)))
        g["norm_out.linear.weight"] = ("BF16", w)
        return [safetensors_parts(g, {"format": RECIPE})]

    out = [
        U("prep inputs", (f"{dit}/prep.safetensors",),
          writes((f"{dit}/prep.safetensors",), lambda t: [prep(t, adapters, variant)]),
          lambda: base([k for k in index if k.startswith(PREP)])
          + lora(lambda n: n.startswith(("token_refiner.", "norm_out.")))),
        U("final layer", (f"{dit}/vdn-globals.safetensors",), writes((f"{dit}/vdn-globals.safetensors",), final),
          lambda: base(GLOBALS.values()) + lora(lambda n: n.startswith("norm_out."))),
    ]  # fmt: skip
    # What the engine's adaLN tabulation reads, in scratch: FL2VA's adaLN
    # bases (26 GB), each with its turbo LoRA merged the OpenVDN way.
    tables, scratch = f"{dit}/adaln.safetensors", stage / ".adaln"

    def adaln_base(t, i):
        native = f"blocks.{i}.adaln_proj.linear.weight"
        deltas = all_deltas(adapters, t).get(
            f"transformer_blocks.{i}.adaln_proj.linear.weight", []
        )
        g = {k: v for k, v in t.items() if k.startswith(f"blocks.{i}.")}
        if deltas:
            dtype, w = g[native]
            if dtype != "BF16":
                raise models.ModelError(f"{native} of MiniMax-H3 is {dtype}")
            rows = vdn_merged_rows(lambda r0, r1: to_f32(dtype, w[r0:r1]), deltas)
            g[native] = ("BF16", np.concatenate(
                [bf16_bits(rows(r0, min(r0 + ROWS, len(w)))) for r0 in range(0, len(w), ROWS)]))  # fmt: skip
        return [safetensors_parts(g)]

    for i in range(BLOCKS):
        path = f".adaln/transformer/base-{i:02d}.safetensors"
        names = [f"blocks.{i}.adaln_proj.linear.{p}" for p in ("weight", "bias")]
        prefix = f"transformer_blocks.{i}.adaln_proj."
        out.append(U(f"adaLN base {i}", (path,), writes((path,), lambda t, i=i: adaln_base(t, i)),
                     lambda names=names, prefix=prefix: base(names) + lora(lambda n, prefix=prefix: n.startswith(prefix)),
                     consumed_by=tables))  # fmt: skip

    def tabulate(_):
        (scratch / dit).mkdir(parents=True, exist_ok=True)
        for name in ("prep.safetensors", "vdn-globals.safetensors"):
            (scratch / dit / name).unlink(missing_ok=True)
            os.link(stage / dit / name, scratch / dit / name)
        # The VDN-H3 DiT's denoising steps, its family row's (8 turbo, 50 default).
        steps = next(v.steps for v in H3.variants.values() if v.dit == variant)
        run_engine(
            [
                "video-adaln",
                str(scratch),
                variant,
                str(scratch / "transformer"),
                str(steps),
            ]
        )
        return {tables: scratch / dit / "adaln.safetensors"}

    out.append(U("adaLN tables", (tables,), tabulate, discard=(".adaln",)))

    def io(t):
        tensors = {k: (dt, np.ascontiguousarray(a)) for k, (dt, a) in t.items()}
        return [safetensors_parts(tensors, {"format": "h3-io-v1"})]

    out += decoders(stage, release)
    out.append(U("patch projections", (f"{dit}/io.safetensors",), writes((f"{dit}/io.safetensors",), io),
                 lambda: base(IO_KEYS)))  # fmt: skip
    # The DiT's blocks, and the QKFIX blocks' files with theirs.
    for i in range(BLOCKS):
        paths = (f"{dit}/vdn-w8-block{i:02d}.safetensors",)
        paths += (f"{dit}/qkfix-block{i:02d}.safetensors",) * (i in QKFIX)
        prefix = f"transformer_blocks.{i}."
        names = [
            n
            for n in index
            if n.startswith(f"blocks.{i}.")
            and n != f"blocks.{i}.adaln_proj.linear.weight"
        ]

        def block(t, i=i):
            src, deltas = BaseSource(t), all_deltas(adapters, t)
            qkv16 = np.empty((3 * H * HD, D), np.float16) if i in QKFIX else None
            tensors = vdn_block(src, t, i, deltas, qkv16)
            files = [safetensors_parts(tensors, {"format": RECIPE, "block": str(i)})]
            if i in QKFIX:
                files.append(qkfix_block(src, i, qkv16))
            return files

        out.append(U(f"block {i}", paths, writes(paths, block),
                     lambda names=names, prefix=prefix: base(names)
                     + [(branch, [n for n in branch.tensors if n.startswith(prefix)])]
                     + lora(lambda n, prefix=prefix: n.startswith(prefix))))  # fmt: skip
    norms = [f"blocks.{b}.attn.{n}_norm.weight" for b in range(BLOCKS) for n in "qk"]

    def checked_manifest(t):
        for i in range(BLOCKS):
            gain = qk_gain(*(t.get(f"blocks.{i}.attn.{n}_norm.weight") for n in "qk"))
            check_qkfix(i, gain, QKFIX, "MiniMax-H3's transformer")
        return [dit_manifest(variant, adapters, t)]

    path = f"{dit}/manifest.json"
    out.append(
        U("manifest", (path,), writes((path,), checked_manifest), lambda: base(norms))
    )
    return out
