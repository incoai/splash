"""FastVideo's FastH3 students of MiniMax-H3, the variants fasth3-8 and
fasth3-4: their recipe (conversion.py), the units that convert one into the
engine's images (runtime/video/model/FastH3.hpp), in an entry's layout:

  dit-<variant>/         per block, the student's to_q / to_k / to_v and VSA
                         compression gate to_gate_compress, to_out and ff
                         (fc1 [gate | value]) as h3.py's int8 linears, its
                         norms, and its adaLN table over the text schedule's
                         distinct timesteps, bf16(W . bf16(silu(temb(t))) +
                         b) in float64 (FASTH3_LAYOUT); the fp16 q / k / v of
                         the QKFIX blocks; the patch projections and final
                         layer (io); the student's time embedder, condition
                         projection, final adaLN and token refiner (prep);
                         manifest.json, the contract (DMD rungs, shifts, VSA
                         tile and sparsity); and adaln.safetensors, the
                         engine's tabulation (splash video-adaln) of the
                         final layer's tables and of the blocks' with the
                         rows of the keyframe timestep (KEYFRAME), which the
                         block units compute into scratch
  vae/, audio/, preview/ h3.py's (h3.decoders), from MiniMax's release

A spec (families.Variant.spec: Pika H3 Flex's pika-6 and pika-ref2va-6)
names the rest of a variant: a checkpoint of MiniMax-H3's DiT, read with no
adapter merged, its compression gate zero at VSA sparsity 0 (dense
attention) and the release's fp32 modules kept fp32; an optional PEFT
adapter, converted for the engine to run at run time into
dit-<variant>/adapter/ (LORA_FORMAT); the sampler's schedule; whether
references condition it (the blocks' adaLN rows of t = 1 in scratch, and
the audio VAE's encoder in audio/); and its QKFIX blocks.

fasth3-8 reads the student FastVideo/FastVideo-FastH3-8-Step-V2 (bf16
diffusers shards) and its contract (CONTRACT: fastvideo_inference.json and
the scheduler configs). fasth3-4 reads MiniMax-H3's FL2VA transformer shards
with the LoRA bundle of FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA
(vsa-datafree, fastvideo-lora-v2: W + B A, then + diff, or set_weight for
the gates the base lacks, rounded to bf16 as the student's checkpoint holds
every tensor), whose base must be the release's (LORA_BASE),
and the contract of the student the bundle was taken from, at a pinned
commit (STUDENT_4). Every file but manifest.json is byte for byte a reference
conversion's (prep.safetensors names its source by repository path), from
fasth3-4's merged tensors rounded to bf16.
"""

from __future__ import annotations

import functools
import io
import json
import os
import re

import numpy as np

from . import conversion, h3, hub, models
from .families import H3

RECIPE = "h3-w8a8-fasth3-v1"
# Each variant's repository the recipe reads with the release.
COMPANIONS = {v: H3.variants[v].companion for v in ("fasth3-8", "fasth3-4")}
STUDENT_8, LORA_4 = COMPANIONS["fasth3-8"], COMPANIONS["fasth3-4"]
LORA_FILE = "vsa-datafree/adapter_model.safetensors"
# The commit of MiniMax-H3 the bundle's deltas apply to (its metadata's
# base_revision); a release of other transformer shards is refused.
LORA_BASE = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
# The four-step student the bundle was taken from, whose contract it is.
STUDENT_4 = ("FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
             "5ea076f35b84da4c3c82217112fa733d8eea2ae1")  # fmt: skip
CONTRACT = ("fastvideo_inference.json", "scheduler/scheduler_config.json",
            "audio_scheduler/scheduler_config.json")  # fmt: skip
TRANSFORMER = "transformer/"
H, HD, D, F = h3.H, h3.HD, h3.D, h3.F
BLOCKS = h3.BLOCKS
# The recipe's QKFIX blocks, which the student's QK gains must give
# (h3.check_qkfix).
QKFIX = (45, 48, 49)
# FastVideo MINIMAX_H3_KEYFRAME_NOISE_AUG: the keyframe rows' timestep.
KEYFRAME = 0.999
FASTH3_LAYOUT = {
    "attn.qkv": ("q8", "[21504, 5376] rows [q | k | v] (to_q / to_k / to_v), head-major; rotated (signs 0..5375)"),
    "attn.gate": ("q8", "[7168, 5376] attn.to_gate_compress (VSA compression gate); rotated (signs 0..5375)"),
    "attn.out": ("q8", "[5376, 7168] to_out.0; rotated (signs 0..7167 of the 14336 sequence)"),
    "mlp.fc1": ("q8", "[28672, 5376] rows [gate | value] (ff.net.0.proj swapped); rotated"),
    "mlp.fc2": ("q8", "[5376, 14336] ff.net.2; rotated"),
    "norm1": ("BF16", "[5376]"), "norm2": ("BF16", "[5376]"),
    "attn.q_norm": ("BF16", "[128] attn.norm_q"), "attn.k_norm": ("BF16", "[128] attn.norm_k"),
    "adaln.table": ("BF16", "[T][96768] bf16(W_bf16 . bf16(silu(temb(t))) + b) in float64 over the manifest's "
                            "adaln_timesteps (the schedule's distinct timesteps); rows [3 modalities][6][5376]"),
}  # fmt: skip
IO = {
    "video_patch_proj.weight": "proj_in.weight", "video_patch_proj.bias": "proj_in.bias",
    "audio_patch_proj.weight": "audio_proj_in.weight", "audio_patch_proj.bias": "audio_proj_in.bias",
    "final_layer.norm.weight": "norm_out.norm.weight",
    "final_layer.video_out.weight": "proj_out.weight", "final_layer.video_out.bias": "proj_out.bias",
    "final_layer.audio_out.weight": "audio_proj_out.weight", "final_layer.audio_out.bias": "audio_proj_out.bias",
}  # fmt: skip
TIME = ("time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight", "time_embedder.linear_2.bias")  # fmt: skip
PREP = ("time_embedder.", "context_embedder.", "norm_out.linear.", "token_refiner.")
# A spec's fields, and its schedule's: the sampler's rungs and shifts, euler
# or renoise (fresh noise after each forward but the last, from the
# request's seed + renoise_seed_offset), its sigmas in float32 or float64,
# and the source the manifest names. The engine runs one forward per step
# (no CFG) on its rungs' shifted sigmas.
SPEC = ("name", "checkpoint", "adapter", "schedule", "references", "qkfix")
SCHEDULE = (
    "rungs",
    "video_shift",
    "audio_shift",
    "sampler",
    "sigma_precision",
    "renoise_seed_offset",
    "source",
)
# A runtime adapter: its directory in the DiT's, its format, the rank the
# engine's LoRA kernels take (a lower one zero-padded) and the linears it
# adapts in each block and in the token refiner.
ADAPTER, LORA_FORMAT, LORA_RANK = "adapter", "h3-lora-v1", 128
LORA_TARGETS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def pinned(variant, spec=None):
    """The sources the variant's conversion reads at a pinned commit."""
    checkpoint = (spec or {}).get("checkpoint", {})
    if checkpoint.get("revision"):
        return h3.PINNED | {checkpoint["repo"]: checkpoint["revision"]}
    return h3.PINNED | (dict([STUDENT_4]) if variant == "fasth3-4" else {})


# ---------------------------------------------------------------- the students' tensors
def from_bf16_bits(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def stored_rows(tensors, name):
    """name's row count and rows(r0, r1), its fp32 rows r0:r1, as stored."""
    dtype, array = tensors[name]
    return len(array), lambda r0, r1: h3.to_f32(dtype, array[r0:r1])


def swapped_rows(n, rows):
    """rows(r0, r1) of swap_halves of an n-row weight whose rows rows gives
    (diffusers ff.net.0.proj [value | gate] <-> fc1 [gate | value])."""
    half = n // 2
    return h3.stacked([lambda r0, r1: rows(half + r0, half + r1), rows], half)


def bf16_rows(n, rows, step=h3.ROWS, into=None):
    """bf16 bits of an n-row weight whose fp32 rows rows gives, step rows at
    a time, into a new array or over into (each step's rows read before they
    are written)."""
    out = into
    for r0 in range(0, n, step):
        bits = h3.bf16_bits(rows(r0, min(r0 + step, n)))
        if out is None:
            out = np.empty((n, *bits.shape[1:]), np.uint16)
        out[r0 : r0 + len(bits)] = bits
    return out


class Student:
    """A FastH3 student's tensors under diffusers names, as fp32 by get, or
    by rows (rows: the row count and rows(r0, r1))."""

    def __init__(self, tensors, keep_f32=False):
        self.tensors = h3.Tensors(tensors)
        # keep_f32: a tensor the base stores in fp32 and nothing modifies (the
        # release's time embedder, patch projections and output heads) keeps
        # its fp32 values, as a runtime adapter's base runs them (Pika H3 Flex:
        # the release's fp32 modules), instead of the student's bf16 rounding.
        self.keep_f32 = keep_f32

    def get(self, name):
        return self.tensors.get(name)

    def rows(self, name):
        return stored_rows(self.tensors, name)


class LoraStudent:
    """MiniMax-H3's FL2VA tensors (native names) with a FastH3 LoRA bundle
    (diffusers names, fastvideo-lora-v2), under diffusers names: W = W_base +
    lora_B @ lora_A, then + .diff (weights) / .diff_b (biases); .set_weight
    replaces the tensor. Merged in fp32, then rounded to bf16 (RNE), the dtype
    of the student the bundle was taken from: the low-rank delta is below
    half a bf16 ulp of most weights and lands on the student's bf16 values
    only once rounded (unrounded, the step-0 velocity is 49 % off the
    student's, rounded 17 %, at int8 level). Whole (get) or by rows (rows),
    with the same arithmetic."""

    GLOBAL = (("time_embedder.linear_1.", "time_embedder.proj_in."), ("time_embedder.linear_2.", "time_embedder.proj_out."),
              ("context_embedder.", "condition_proj."), ("norm_out.linear.", "final_layer.adaln_proj.linear."),
              ("norm_out.norm.", "final_layer.norm."), ("proj_in.", "video_patch_proj."),
              ("audio_proj_in.", "audio_patch_proj."), ("proj_out.", "final_layer.video_out."),
              ("audio_proj_out.", "final_layer.audio_out."), ("token_refiner.final_norm.", "token_refiner.final_norm."))  # fmt: skip
    REFINER = {"norm1.weight": "norm1.weight", "norm2.weight": "norm2.weight", "attn.norm_q.weight": "attn.q_norm.weight",
               "attn.norm_k.weight": "attn.k_norm.weight", "attn.to_out.0.weight": "attn.out_proj.weight",
               "ff.net.2.weight": "mlp.fc2.weight"}  # fmt: skip
    BLOCK = h3.NATIVE | {"adaln_proj.linear.weight": "adaln_proj.linear.weight"}

    def __init__(self, tensors, keep_f32=False):
        self.tensors = h3.Tensors(tensors)
        # keep_f32: a tensor the base stores in fp32 and nothing modifies (the
        # release's time embedder, patch projections and output heads) keeps
        # its fp32 values, as a runtime adapter's base runs them (Pika H3 Flex:
        # the release's fp32 modules), instead of the student's bf16 rounding.
        self.keep_f32 = keep_f32

    def _base(self, name):
        """The base's row count and rows(r0, r1) of name."""
        native = functools.partial(stored_rows, self.tensors)
        if m := re.match(r"^transformer_blocks\.(\d+)\.(.*)$", name):
            p, rest, names = f"blocks.{m.group(1)}.", m.group(2), self.BLOCK
        elif m := re.match(r"^token_refiner\.refiner_blocks\.(\d+)\.(.*)$", name):
            p, rest, names = (
                f"token_refiner.blocks.{m.group(1)}.",
                m.group(2),
                self.REFINER,
            )
        else:
            for diffusers, nat in self.GLOBAL:
                if name.startswith(diffusers):
                    return native(nat + name[len(diffusers) :])
            raise KeyError(name)
        if rest in names:
            return native(p + names[rest])
        if rest == "ff.net.0.proj.weight":  # native [gate | value] -> [value | gate]
            n, rows = native(p + "mlp.fc1.weight")
            return n, swapped_rows(n, rows)
        part = "qkv".index(re.fullmatch(r"attn\.to_([qkv])\.weight", rest).group(1))
        dtype, qkv = self.tensors[p + "attn.qkv_proj.weight"]

        def rows(r0, r1):  # of [head][128] -> rows [head][q | k | v][128]
            r = np.arange(r0, r1)
            return h3.to_f32(dtype, qkv[r // HD * 3 * HD + part * HD + r % HD])

        return H * HD, rows

    def rows(self, name):
        n, rows = self._merged(name)
        if self.keep_f32 and self._f32_base(name):
            return n, rows
        return n, lambda r0, r1: h3.to_f32("BF16", h3.bf16_bits(rows(r0, r1)))

    def _f32_base(self, name):
        """Whether name is a global the base stores in fp32 and no bundle
        tensor modifies."""
        stem, kind = name.rsplit(".", 1)
        extra = stem + (".diff" if kind == "weight" else ".diff_b")
        if any(
            k in self.tensors
            for k in (stem + ".set_weight", stem + ".lora_A.weight", extra)
        ):
            return False
        for diffusers, nat in self.GLOBAL:
            if name.startswith(diffusers):
                native = nat + name[len(diffusers) :]
                return native in self.tensors and self.tensors[native][0] == "F32"
        return False

    def _merged(self, name):
        stem, kind = name.rsplit(".", 1)
        lora = self.tensors
        if kind == "weight" and stem + ".set_weight" in lora:
            return stored_rows(lora, stem + ".set_weight")
        n, rows = self._base(name)
        extra = stem + (".diff" if kind == "weight" else ".diff_b")
        if kind == "weight" and stem + ".lora_A.weight" in lora:
            ab = (
                lora.get(stem + ".lora_A.weight"),
                lora.get(stem + ".lora_B.weight"),
                1.0,
            )
            rows = h3.merged_rows(rows, [ab])
        if extra in lora:
            _, diff = stored_rows(lora, extra)
            return n, lambda r0, r1, rows=rows: rows(r0, r1) + diff(r0, r1)
        return n, rows

    def get(self, name):
        n, rows = self.rows(name)
        return np.asarray(rows(0, n), dtype=np.float32)


class DiffusersStudent(LoraStudent):
    """LoraStudent over a diffusers-named trunk: a name it stores is read as stored (the diffusers layout LoraStudent
    otherwise builds from the native names: to_q / to_k / to_v, ff.net.0.proj [value | gate]); an fp32 global it stores
    stays fp32 under keep_f32."""

    def _base(self, name):
        if name in self.tensors:
            return stored_rows(self.tensors, name)
        return super()._base(name)

    def _f32_base(self, name):
        if name in self.tensors:
            stem, kind = name.rsplit(".", 1)
            extra = stem + (".diff" if kind == "weight" else ".diff_b")
            if any(
                k in self.tensors
                for k in (stem + ".set_weight", stem + ".lora_A.weight", extra)
            ):
                return False
            return self.tensors[name][0] == "F32"
        return super()._f32_base(name)


# ---------------------------------------------------------------- conversion
def contract(files):
    """The schedule and VSA settings of a student's fastvideo_inference.json
    and scheduler configs, files: name -> parsed JSON."""
    c = files[CONTRACT[0]]
    shifts = {}
    for key, name in (
        ("video_scheduler_shift", CONTRACT[1]),
        ("audio_scheduler_shift", CONTRACT[2]),
    ):
        shift = float(files[name]["shift"])
        if key in c and float(c[key]) != shift:
            raise models.ModelError(
                f"the FastH3 contract's {key} disagrees with {name}"
            )
        shifts[key] = float(c.get(key, shift))
    rungs = [int(x) for x in c["dmd_denoising_steps"]]
    if (
        c.get("transformer_forwards") != len(rungs)
        or c.get("num_inference_steps") != len(rungs) + 1
    ):
        raise models.ModelError(
            "the FastH3 contract's rungs disagree with its forwards"
        )
    if (
        c.get("attention_backend") != "VIDEO_SPARSE_ATTN_H3"
        or int(c.get("vsa_tile_size", 0)) != 64
    ):
        raise models.ModelError(
            "the FastH3 contract is not VIDEO_SPARSE_ATTN_H3 on 64-token tiles"
        )
    return {"dmd_rungs": rungs, "video_shift": shifts["video_scheduler_shift"],
            "audio_shift": shifts["audio_scheduler_shift"], "vsa_sparsity": float(c["vsa_sparsity"]),
            "model_id": c.get("model_id"), "checkpoint_step": c.get("checkpoint_step"),
            "guidance_scale": c.get("guidance_scale")}  # fmt: skip


def check_spec(spec):
    """Refuses a spec the recipe or the engine cannot run."""
    s = spec["schedule"]
    if extra := sorted(set(spec) - set(SPEC)) + sorted(
        set(s) - set(SCHEDULE) - {"guidance_scale", "vsa_sparsity"}
    ):
        raise models.ModelError(f"the spec has no field {extra[0]}")
    if s.get("guidance_scale", 1.0) != 1.0:
        raise models.ModelError(
            "the engine runs one forward per step: a spec's guidance_scale is 1 (no CFG)"
        )
    if s.get("vsa_sparsity", 0.0) != 0.0:
        raise models.ModelError(
            "a spec's DiT runs dense attention: its VSA sparsity is 0"
        )
    c, adapter = spec["checkpoint"], spec["adapter"] or {}
    if any(
        type(v) not in (int, float)
        for v in (s["video_shift"], s["audio_shift"], adapter.get("alpha", 0))
    ):
        raise models.ModelError("a spec's shifts and its adapter's alpha are numbers")
    texts = [spec["name"], c["repo"], c["prefix"]] + [
        adapter[k] for k in ("repo", "file", "name") if k in adapter
    ]
    if (
        any(type(t) is not str for t in texts)
        or c.get("revision") is not None
        and type(c["revision"]) is not str
    ):
        raise models.ModelError(
            "a spec's name, repo, revision, prefix, file and adapter name are strings"
        )
    if s["sampler"] not in ("euler", "renoise") or s["sigma_precision"] not in (
        "float32",
        "float64",
    ):
        raise models.ModelError(
            "a spec's sampler is euler or renoise, its sigma_precision float32 or float64"
        )
    rungs = s["rungs"]
    if (
        type(rungs) is not list
        or not rungs
        or any(type(r) is not int or not 0 < r <= 1000 for r in rungs)
        or any(a <= b for a, b in zip(rungs, rungs[1:]))
    ):
        raise models.ModelError(
            "a spec's rungs are whole timesteps strictly decreasing in (0, 1000]; the engine shifts them into its "
            "sigmas, so a list of sigmas is none"
        )
    if not spec["qkfix"] or any(
        type(b) is not int or not 0 <= b < BLOCKS for b in spec["qkfix"]
    ):
        raise models.ModelError(f"a spec's qkfix names blocks 0 to {BLOCKS - 1}")


def spec_contract(spec, source):
    """The contract (contract's fields) of a spec's schedule, whose DiT is
    source's with no adapter merged."""
    s = spec["schedule"]
    return {"dmd_rungs": list(s["rungs"]), "video_shift": float(s["video_shift"]),
            "audio_shift": float(s["audio_shift"]), "vsa_sparsity": 0.0,
            "model_id": f"{source} (base, no adapter merged)", "checkpoint_step": None, "guidance_scale": 1.0}  # fmt: skip


def timesteps(rungs, video_shift, audio_shift, sigma64=False):
    """model::Prep::scheduleRungs in numpy fp32: the sorted distinct transformer
    timesteps (1 - sigma) of both modalities. sigma64: Pika H3 Flex's
    H3SigmaSchedule, q = rungs / 1000 in float64, the shifted sigma in
    float64, then fp32."""
    base = [np.float32(r / 1000.0) for r in rungs] + [np.float32(0.0)]

    def sigmas(shift):
        if sigma64:
            q = np.array([*rungs, 0], dtype=np.float64) / 1000.0
            return (shift * q / (1.0 + (shift - 1.0) * q)).astype(np.float32)
        s32, sm1 = np.float32(shift), np.float32(shift - 1.0)
        return [
            np.float32(
                np.float32(s32 * b) / np.float32(np.float32(1.0) + np.float32(sm1 * b))
            )
            for b in base
        ]

    ts = set()
    for shift in (video_shift, audio_shift):
        ts.update(np.float32(1.0 - float(x)) for x in sigmas(shift)[:-1])
    return np.array(sorted(ts), dtype=np.float32)


def silu_temb(src, table):
    """bf16(silu(temb(t))) [T][2688] for the adaLN projection: the sinusoid of
    kernels/video/prep (fp32: cos | sin of t exp(-ln(1e4) i / 128)), then the time
    embedder's linear_1 -> silu -> linear_2 in float64, silu, bf16."""
    i = np.arange(128, dtype=np.float32)
    e = np.exp((np.float32(-9.210340371976184) * i) / np.float32(128.0)).astype(
        np.float32
    )
    a = (np.asarray(table)[:, None].astype(np.float32) * e[None, :]).astype(np.float32)
    sin = np.concatenate([np.cos(a), np.sin(a)], axis=1).astype(np.float64)
    w1, b1, w2, b2 = (src.get(n) for n in TIME)
    h = sin @ w1.astype(np.float64).T + b1
    h = h / (1 + np.exp(-h))
    temb = h @ w2.astype(np.float64).T + b2
    st = temb / (1 + np.exp(-temb))
    return h3.bf16_bits(st.astype(np.float32))


def qk_gain(src, i):
    t = "transformer_blocks.%d." % i
    gq, gk = src.get(t + "attn.norm_q.weight"), src.get(t + "attn.norm_k.weight")
    return h3.qk_gain(gq, gk), gq, gk


def block(src, i, st, keyframe, qkv16=None):
    """Block i's tensors, canonical name -> (dtype, array): the rotated int8
    linears and their scales, the norms, the adaLN table of st (bf16 bits of
    silu(temb) [T][2688]); and its adaLN projection of keyframe. qkv16, of a
    QKFIX block, gets its fp16 q / k / v. Each linear is merged, rotated and
    quantized h3.ROWS rows at a time (h3.q8_rows)."""
    t = "transformer_blocks.%d." % i
    out = {}
    sig5, sig14 = h3.had_signs(D), h3.had_signs(F)

    def q8(canon, n, rows, signs, keep=None):
        q, s = h3.q8_rows(n, rows, signs, keep)
        out["blocks.%d.%s.qweight" % (i, canon)] = ("I8", q)
        out["blocks.%d.%s.wscale" % (i, canon)] = ("F32", s)

    def keep(r0, w):
        qkv16[r0 : r0 + len(w)] = w.astype(np.float16)

    (n, q), (_, k), (_, v) = (src.rows(t + "attn.to_%s.weight" % p) for p in "qkv")
    q8(
        "attn.qkv",
        3 * n,
        h3.stacked([q, k, v], n),
        sig5,
        keep if qkv16 is not None else None,
    )
    q8("attn.gate", *src.rows(t + "attn.to_gate_compress.weight"), sig5)
    q8("attn.out", *src.rows(t + "attn.to_out.0.weight"), sig14[: H * HD])
    n, fc1 = src.rows(t + "ff.net.0.proj.weight")
    q8("mlp.fc1", n, swapped_rows(n, fc1), sig5)
    q8("mlp.fc2", *src.rows(t + "ff.net.2.weight"), sig14)
    for canon, name in (("norm1", "norm1.weight"), ("norm2", "norm2.weight"), ("attn.q_norm", "attn.norm_q.weight"),
                        ("attn.k_norm", "attn.norm_k.weight")):  # fmt: skip
        out["blocks.%d.%s" % (i, canon)] = ("BF16", h3.bf16_bits(src.get(t + name)))
    table, keyframe = adaln(src, i, st, keyframe)
    out["blocks.%d.adaln.table" % i] = ("BF16", table)
    return out, keyframe


def adaln(src, i, *sts):
    """bf16 bits of block i's adaLN projection of each of sts, in float64,
    its weight read h3.ROWS rows at a time."""
    t = "transformer_blocks.%d." % i
    n, rows = src.rows(t + "adaln_proj.linear.weight")
    b = src.get(t + "adaln_proj.linear.bias").astype(np.float64)
    xs = [from_bf16_bits(st).astype(np.float64) for st in sts]
    outs = [np.empty((len(x), n), np.uint16) for x in xs]
    for r0 in range(0, n, h3.ROWS):
        r1 = min(r0 + h3.ROWS, n)
        w = rows(r0, r1).astype(np.float64)
        for x, out in zip(xs, outs, strict=True):
            out[:, r0:r1] = h3.bf16_bits((x @ w.T + b[r0:r1]).astype(np.float32))
    return outs


def block_names(i):
    t = "transformer_blocks.%d." % i
    return [t + n for n in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.to_gate_compress.weight",
                            "attn.to_out.0.weight", "ff.net.0.proj.weight", "ff.net.2.weight", "norm1.weight",
                            "norm2.weight", "attn.norm_q.weight", "attn.norm_k.weight", "adaln_proj.linear.weight",
                            "adaln_proj.linear.bias")]  # fmt: skip


def prep(src, variant, source):
    """prep.safetensors: the student's time embedder (F32), condition
    projection, final adaLN and token refiner (BF16, FL2VA's names: to_q /
    to_k / to_v interleaved per head into qkv_proj, ff.net.0.proj [value;
    gate] -> fc1 [gate; value]), converted by rows over the tensors read
    where a row reads only the same row of one: FL2VA's of the output's name
    (fasth3-4's base), else the student's of the same layout."""

    def f32(n):
        return np.asarray(src.get(n), dtype=np.float32)

    def over(*names):
        for name in names:
            if name in src.tensors and src.tensors[name][0] == "BF16":
                array = src.tensors[name][1]
                return array if array.flags.writeable else None
        return None

    def bf(n, out=None):
        return bf16_rows(*src.rows(n), into=over(out, n) if out else None)

    t = {}
    for nat, dif in (("proj_in", "linear_1"), ("proj_out", "linear_2")):
        for q in ("weight", "bias"):
            t[f"time_embedder.{nat}.{q}"] = ("F32", f32(f"time_embedder.{dif}.{q}"))
    for q in ("weight", "bias"):
        for out, n in ((f"condition_proj.{q}", f"context_embedder.{q}"),
                       (f"final_layer.adaln_proj.linear.{q}", f"norm_out.linear.{q}")):  # fmt: skip
            t[out] = ("BF16", bf(n, out))
    name = "token_refiner.final_norm.weight"
    t[name] = ("BF16", bf(name, name))
    for blk in range(2):
        s, n = f"token_refiner.refiner_blocks.{blk}.", f"token_refiner.blocks.{blk}."
        for nat, dif in (("norm1", "norm1"), ("norm2", "norm2"), ("attn.q_norm", "attn.norm_q"),
                         ("attn.k_norm", "attn.norm_k"), ("attn.out_proj", "attn.to_out.0"), ("mlp.fc2", "ff.net.2")):  # fmt: skip
            t[n + nat + ".weight"] = (
                "BF16",
                bf(s + dif + ".weight", n + nat + ".weight"),
            )
        parts = [src.rows(s + f"attn.to_{p}.weight")[1] for p in "qkv"]

        def heads(r0, r1, parts=parts):  # whole heads [q | k | v][128]
            h0, h1 = r0 // (3 * HD), r1 // (3 * HD)
            return np.concatenate(
                [p(h * HD, (h + 1) * HD) for h in range(h0, h1) for p in parts]
            )

        # FL2VA's qkv_proj and fc1 rows are the same rows of the output's
        qkv, fc1 = n + "attn.qkv_proj.weight", n + "mlp.fc1.weight"
        t[qkv] = ("BF16", bf16_rows(3 * H * HD, heads, 3 * HD, over(qkv)))
        rows = src.rows(s + "ff.net.0.proj.weight")
        t[fc1] = ("BF16", bf16_rows(rows[0], swapped_rows(*rows), into=over(fc1)))
    return h3.safetensors_parts(t, {"variant": variant, "source": source})


def patches(src):
    """io.safetensors (FL2VA's names): the patch projections and final layer,
    F32 but the final norm (BF16)."""
    out = {}
    for k, n in IO.items():
        v = src.get(n)
        out[k] = (
            ("BF16", h3.bf16_bits(v)) if k == "final_layer.norm.weight" else ("F32", v)
        )
    return h3.safetensors_parts(out, {"format": "h3-io-v1", "source": "fasth3"})


def dit_manifest(variant, ctr, table, gains, src, source, qkfix=QKFIX):
    qk = {"blocks": {}, "qk": "fp16 x fp16 -> fp32 q/k/v GEMM (student weights, not rotated), fp32 QK-norm + RoPE epilogue",
          "attn": "fp16 Q.K + exact fp32 dot of the split dims (zeroed in the fp16 Q/K)",
          "rule": "blocks with max|norm_q * norm_k| > %g (base FL2VA: 45, 48, 49 at 144 / 97 / 1249, next 8.4)"
                  % h3.QKFIX_GAIN}  # fmt: skip
    for b in qkfix:
        gain, gq, gk = qk_gain(src, b)
        qk["blocks"][str(b)] = {"file": "qkfix-block%02d.safetensors" % b, "split": h3.qkfix_split_dims(gq, gk),
                                "gain": gain, "gq_max": float(np.abs(gq).max()), "gk_max": float(np.abs(gk).max())}  # fmt: skip
    return h3.manifest({
        "format": RECIPE, "variant": variant, "source": source, "model_id": ctr["model_id"],
        "checkpoint_step": ctr["checkpoint_step"],
        "hadamard": {"group": h3.HAD_GROUP, "seed": h3.HAD_SEED, "signs": "splitmix64",
                     "out_signs": "0..7167 of the 14336 sequence"},
        "quant": "symmetric int8, per-output-row fp32 scale = absmax/127 (after rotation)",
        "layout": {k: list(v) for k, v in FASTH3_LAYOUT.items()},
        "vsa": {"tile": [4, 4, 4], "sparsity": ctr["vsa_sparsity"], "prefix": "exempt (text, keyframes, audio dense)",
                "compression_gate": "attn.to_gate_compress"},
        "schedule": {"dmd_rungs": ctr["dmd_rungs"], "video_shift": ctr["video_shift"],
                     "audio_shift": ctr["audio_shift"], "guidance_scale": ctr["guidance_scale"]},
        "adaln_timesteps": [float(x) for x in table],
        "adaln_keyframe_timestep": KEYFRAME,
        "files": {str(i): "fasth3-w8-block%02d.safetensors" % i for i in range(BLOCKS)},
        "qk_gain": {str(i): g for i, g in enumerate(gains)},
        "qkfix": qk,
    })  # fmt: skip


def dense_manifest(data, spec):
    """A spec's DiT manifest from dit_manifest's data: dense attention (no
    compression gate, sparsity 0), its sampler, and its runtime adapter."""
    m = json.loads(data)
    m["vsa"]["compression_gate"] = (
        "none (zero rows: the base has no to_gate_compress; sparsity 0 = dense attention)"
    )
    # the engine's dense path (ops::VsaSpec::dense): no gate rows, coarse stage or selection run
    m["vsa"]["attention"] = "dense"
    s = spec["schedule"]
    m["schedule"].update({"knots": [*s["rungs"], 0], "sampler": s["sampler"], "sigma_precision": s["sigma_precision"]}
                         | {k: s[k] for k in ("renoise_seed_offset", "source") if k in s})  # fmt: skip
    if spec.get("adapter"):
        m["adapters"] = [{"path": ADAPTER, "scale": 1.0}]
    return h3.manifest(m)


def check_shapes(shards, diffusers):
    """Refuses a checkpoint (its shards, conversion.Source each) whose
    blocks are not MiniMax-H3's DiT's: BLOCKS blocks of H heads of HD, width
    D, MLP width F."""
    shapes = {n: e["shape"] for s in shards.values() for n, e in s.tensors.items()}
    if diffusers:
        want = {"transformer_blocks.%d.attn.to_q.weight": [H * HD, D], "transformer_blocks.%d.attn.to_out.0.weight": [D, H * HD],
                "transformer_blocks.%d.ff.net.0.proj.weight": [2 * F, D], "transformer_blocks.%d.ff.net.2.weight": [D, F]}  # fmt: skip
    else:
        want = {"blocks.%d.attn.qkv_proj.weight": [3 * H * HD, D], "blocks.%d.attn.out_proj.weight": [D, H * HD],
                "blocks.%d.mlp.fc1.weight": [2 * F, D], "blocks.%d.mlp.fc2.weight": [D, F]}  # fmt: skip
    for i in range(BLOCKS + 1):
        for name, shape in want.items():
            if shapes.get(name % i) != (shape if i < BLOCKS else None):
                raise models.ModelError(
                    f"the spec's checkpoint is not MiniMax-H3's DiT: {name % i} is {shapes.get(name % i)}, not "
                    f"{shape if i < BLOCKS else 'absent'} ({BLOCKS} blocks)"
                )


def lora_units(out, source, adapter, qkfix, name, revision):
    """The units converting a PEFT adapter (adapter: the spec's file, its
    adapter name and alpha; source, its conversion.Source, of tensors
    X.lora_A.<name>.weight [r, in] and X.lora_B.<name>.weight [out, r])
    into the engine's runtime adapter (runtime/video/model/Adapter.hpp,
    LORA_FORMAT) in the entry's directory out, whose manifest names it name
    and the adapter's repository at revision. All bf16 (RNE of the source,
    as PEFT's bf16 master copy holds them), A and B zero-padded to rank
    LORA_RANK:
      lora-blockNN.safetensors   blocks.N.lora.{qkv,out,fc1,fc2}.{a,b} of transformer block N:
          qkv.a  [3r, 5376]  [A_q; A_k; A_v], Hadamard-rotated as the int8 weights' inputs (signs 0..5375)
          qkv.b  [21504, r]  [B_q; B_k; B_v] (rows head-major, the int8 qkv weight's)
          out.a  [r, 7168]   to_out.0, rotated (signs 0..7167 of the 14336 sequence);  out.b [5376, r]
          fc1.a  [r, 5376]   ff.net.0.proj, rotated;  fc1.b [28672, r] rows [gate | value] (diffusers' halves swapped)
          fc2.a  [r, 14336]  ff.net.2, rotated;  fc2.b [5376, r]
          qkv.a_plain [3r, 5376]  QKFIX blocks only: A unrotated, for their fp16 q / k / v GEMM
      lora-refiner.safetensors   refiner.N.{qkv,out,fc1,fc2}.{a,b} of token-refiner block N (the prep's bf16 linears,
          unrotated): qkv.a [3r, 5376]; qkv.b [21504, 3r] block-diagonal on qkv_proj's per-head interleaved rows
          [head][q | k | v][128] (zeros elsewhere); fc1.b [gate | value]
    The rotation is of the bf16-rounded A in float64, rounded to bf16 once
    more. Each module's B is scaled by its alpha / r over the adapter's
    scale when the modules' ranks differ."""
    file, alpha = adapter["file"], float(adapter["alpha"])
    pattern = re.compile(
        r"^(.*)\.lora_([AB])\." + re.escape(adapter["name"]) + r"\.weight$"
    )
    found = {}
    for t in source.tensors:
        if not (m := pattern.match(t)):
            raise models.ModelError(
                f"{file}: not a LoRA tensor of adapter {adapter['name']!r}: {t}"
            )
        found.setdefault(m.group(1), {})[m.group(2)] = t
    if lacking := [m for m, ab in found.items() if set(ab) != {"A", "B"}]:
        raise models.ModelError(f"{file}: {lacking[0]} lacks lora_A or lora_B")
    ranks = {m: source.tensors[ab["A"]]["shape"][0] for m, ab in found.items()}
    if over := [m for m, r in ranks.items() if r > LORA_RANK]:
        raise models.ModelError(
            f"{file}: {over[0]} has rank {ranks[over[0]]}, above the engine's {LORA_RANK}"
        )
    nref = 0
    while f"token_refiner.refiner_blocks.{nref}.attn.to_q" in found:
        nref += 1
    blocks = {
        i: [f"transformer_blocks.{i}.{t}" for t in LORA_TARGETS] for i in range(BLOCKS)
    }
    refiner = [
        f"token_refiner.refiner_blocks.{b}.{t}"
        for b in range(nref)
        for t in LORA_TARGETS
    ]
    wanted = [m for ms in blocks.values() for m in ms] + refiner
    if missing := [m for m in wanted if m not in found]:
        raise models.ModelError(f"{file}: missing LoRA module {missing[0]}")
    if unused := sorted(set(found) - set(wanted)):
        raise models.ModelError(
            f"{file}: LoRA modules with no engine linear (the blocks' and the token refiner's q / k / v, out, fc1 "
            f"and fc2; no adaLN or compression gate): {unused[:8]}"
        )
    scales = {alpha / r for r in ranks.values()}
    # one scale for the adapter when the modules share it; else each module's folded into its B (then scale 1)
    scale = next(iter(scales)) if len(scales) == 1 else 1.0
    r = LORA_RANK

    def ab(t, module):
        """bf16 A [128, K] and B [N, 128] of module (B times its scale over
        the adapter's when they differ), zero-padded to rank 128."""
        a_, b_ = t.get(found[module]["A"]), t.get(found[module]["B"])
        n, sc = a_.shape[0], alpha / a_.shape[0]
        A = np.zeros((r, a_.shape[1]), np.float32)
        B = np.zeros((b_.shape[0], r), np.float32)
        A[:n] = bf(a_)
        B[:, :n] = bf(b_) if sc == scale else bf(b_ * np.float32(sc / scale))
        return A, B

    def bf(x):
        """fp32 values of x rounded to bf16 (RNE)."""
        return h3.to_f32("BF16", h3.bf16_bits(np.asarray(x, np.float32)))

    def rot(a_, signs):
        return h3.bf16_bits(h3.had_rotate(a_, signs).astype(np.float32))

    def reads(modules):
        return lambda: [(source, [found[m][k] for m in modules for k in "AB"])]

    def lora_block(t, i):
        sig5, sig14 = h3.had_signs(D), h3.had_signs(F)
        pre = f"transformer_blocks.{i}."
        (aq, bq), (ak, bk), (av, bv) = (ab(t, pre + f"attn.to_{x}") for x in "qkv")
        ao, bo = ab(t, pre + "attn.to_out.0")
        a1, b1 = ab(t, pre + "ff.net.0.proj")
        a2, b2 = ab(t, pre + "ff.net.2")
        aqkv = np.concatenate([aq, ak, av])
        tensors = {
            f"blocks.{i}.lora.qkv.a": ("BF16", rot(aqkv, sig5)),
            f"blocks.{i}.lora.qkv.b": (
                "BF16",
                h3.bf16_bits(np.concatenate([bq, bk, bv])),
            ),
            f"blocks.{i}.lora.out.a": ("BF16", rot(ao, sig14[: H * HD])),
            f"blocks.{i}.lora.out.b": ("BF16", h3.bf16_bits(bo)),
            f"blocks.{i}.lora.fc1.a": ("BF16", rot(a1, sig5)),
            f"blocks.{i}.lora.fc1.b": ("BF16", h3.bf16_bits(h3.swap_halves(b1))),
            f"blocks.{i}.lora.fc2.a": ("BF16", rot(a2, sig14)),
            f"blocks.{i}.lora.fc2.b": ("BF16", h3.bf16_bits(b2)),
        }
        if i in qkfix:
            tensors[f"blocks.{i}.lora.qkv.a_plain"] = ("BF16", h3.bf16_bits(aqkv))
        return [h3.safetensors_parts(tensors, {"format": LORA_FORMAT, "block": str(i)})]

    def lora_refiner(t):
        tensors = {}
        for blk in range(nref):
            pre = f"token_refiner.refiner_blocks.{blk}."
            parts = [ab(t, pre + f"attn.to_{x}") for x in "qkv"]
            bqkv = np.zeros(
                (3 * H * HD, 3 * r), np.float32
            )  # rows [head][q | k | v][128]
            for x, (_, b) in enumerate(parts):
                rows = (
                    np.arange(H)[:, None] * 3 * HD + x * HD + np.arange(HD)[None, :]
                ).reshape(-1)
                bqkv[rows, x * r : (x + 1) * r] = b
            ao, bo = ab(t, pre + "attn.to_out.0")
            a1, b1 = ab(t, pre + "ff.net.0.proj")
            a2, b2 = ab(t, pre + "ff.net.2")
            s = f"refiner.{blk}."
            tensors |= {
                s + "qkv.a": ("BF16", h3.bf16_bits(np.concatenate([p[0] for p in parts]))),
                s + "qkv.b": ("BF16", h3.bf16_bits(bqkv)),
                s + "out.a": ("BF16", h3.bf16_bits(ao)), s + "out.b": ("BF16", h3.bf16_bits(bo)),
                s + "fc1.a": ("BF16", h3.bf16_bits(a1)), s + "fc1.b": ("BF16", h3.bf16_bits(h3.swap_halves(b1))),
                s + "fc2.a": ("BF16", h3.bf16_bits(a2)), s + "fc2.b": ("BF16", h3.bf16_bits(b2)),
            }  # fmt: skip
        return [h3.safetensors_parts(tensors, {"format": LORA_FORMAT})]

    def manifest(_):
        sha = (
            source.repo.sizes[file][1]
            if file in source.repo.sizes
            else source.repo.file(file).resolve().name
        )
        files = {str(i): f"lora-block{i:02d}.safetensors" for i in range(BLOCKS)}
        return [h3.manifest({
            "format": LORA_FORMAT, "name": name,
            "source": {"repo": source.repo.name, "revision": revision,
                       "files": [{"file": file, "sha256": sha, "adapter": adapter["name"], "alpha": alpha,
                                  "metadata": source.metadata}]},
            "rank": r, "source_ranks": sorted(set(ranks.values())), "scale": scale,
            "combined": "per linear, the sources' A rows and B columns in order, zero-padded to rank 128"
                        + ("" if len(scales | {scale}) == 1 else "; B scaled by each source's alpha / r"),
            "left_out": [], "left_out_count": 0,
            "dtype": "BF16 (RNE of the FP32 source, PEFT's bf16 master copy); A rotated in float64 from the bf16 values",
            "hadamard": {"group": h3.HAD_GROUP, "seed": h3.HAD_SEED, "signs": "splitmix64",
                         "a_inputs": {"qkv": "0..5375", "out": "0..7167 of the 14336 sequence", "fc1": "0..5375",
                                      "fc2": "0..14335"}},
            "blocks": BLOCKS, "refiner_blocks": nref, "files": files,
            "refiner_file": "lora-refiner.safetensors" if nref else None,
            "targets": {
                "transformer_blocks.N.attn.to_q/to_k/to_v": "attn.qkv rows [q | k | v] (int8 GEMM; QKFIX blocks "
                                                            + ", ".join(map(str, qkfix)) + ": the fp16 q / k / v GEMM)",
                "transformer_blocks.N.attn.to_out.0": "attn.out",
                "transformer_blocks.N.ff.net.0.proj": "mlp.fc1 (rows [gate | value])",
                "transformer_blocks.N.ff.net.2": "mlp.fc2",
                "token_refiner.refiner_blocks.N.attn.to_q/to_k/to_v": "prep token_refiner.blocks.N.attn.qkv_proj "
                                                                        "(per-head interleaved rows)",
                "token_refiner.refiner_blocks.N.attn.to_out.0": "prep attn.out_proj",
                "token_refiner.refiner_blocks.N.ff.net.0.proj": "prep mlp.fc1 ([gate | value])",
                "token_refiner.refiner_blocks.N.ff.net.2": "prep mlp.fc2",
            },
            "engine_linears_without_lora": [
                "blocks.N.attn.gate (VSA compression gate; zero in the dense base)",
                "blocks.N.adaln_proj (tabulated adaLN; not a target)",
                "final_layer.adaln_proj, final_layer.video_out / audio_out, video / audio patch projections",
                "condition_proj, time_embedder",
            ],
        })]  # fmt: skip

    U, writes = conversion.Unit, h3.writes
    units = []
    for i, modules in blocks.items():
        path = (f"{out}/lora-block{i:02d}.safetensors",)
        units.append(
            U(
                f"adapter block {i}",
                path,
                writes(path, functools.partial(lora_block, i=i)),
                reads(modules),
            )
        )
    if nref:
        path = (f"{out}/lora-refiner.safetensors",)
        units.append(
            U("adapter token refiner", path, writes(path, lora_refiner), reads(refiner))
        )
    path = (f"{out}/{H3.dit_manifest}",)
    return units + [U("adapter manifest", path, writes(path, manifest))]


# ---------------------------------------------------------------- units
def _transformer_files(repo, prefix, index):
    return [
        n
        for n in repo.files
        if n.startswith(prefix) and n.endswith((".safetensors", index))
    ]


def _student_4():
    return hub.Repository.resolve(*STUDENT_4)


def _require_base(release):
    """The release's FL2VA transformer shards must be the bundle's base's."""
    if release.revision == LORA_BASE:
        return
    base = hub.Repository.resolve(release.name, LORA_BASE)
    for name in _transformer_files(release, h3.TRANSFORMER, ".index.json"):
        if base.sizes.get(name) != release.sizes.get(name):
            raise models.ModelError(
                f"{release.name}'s {name} is not the one at {LORA_BASE[:12]}, which "
                f"{LORA_4} applies to"
            )


def _checkpoint(spec, companion):
    """The repository a spec's checkpoint is read from: its repo at its
    revision, or else the companion at its commit."""
    c = spec["checkpoint"]
    if c.get("revision"):
        return hub.Repository.resolve(c["repo"], c["revision"])
    if c["repo"] != companion.name:
        raise models.ModelError(f"the spec's checkpoint {c['repo']} names no revision")
    return companion


def inputs(variant, release, companion, spec=None):
    """The files the variant's conversion reads, REPO/NAME -> the content ID
    the Hub reports, and TAEH3's SHA-256; release and companion as the Hub
    listed them, spec the variant's (families.Variant.spec), if any."""
    decoders = [
        f"{d}{n}"
        for d in (h3.VIDEO_VAE, h3.AUDIO_VAE)
        for n in ("model.safetensors", "config.json")
    ]
    if spec:
        check_spec(spec)
        checkpoint = _checkpoint(spec, companion)
        names = {release: decoders}
        names.setdefault(checkpoint, []).extend(
            _transformer_files(checkpoint, spec["checkpoint"]["prefix"], ".index.json")
        )
        if spec.get("adapter"):
            names.setdefault(companion, []).append(spec["adapter"]["file"])
    elif variant == "fasth3-8":
        names = {release: decoders,
                 companion: _transformer_files(companion, TRANSFORMER, ".index.json") + list(CONTRACT)}  # fmt: skip
    else:
        _require_base(release)
        names = {release: _transformer_files(release, h3.TRANSFORMER, ".index.json") + decoders,
                 companion: [LORA_FILE], _student_4(): list(CONTRACT)}  # fmt: skip
    files = {f"{h3.TAEH3_REPO}/{h3.TAEH3}": h3.TAEH3_SHA256}
    for repo, wanted in names.items():
        if missing := sorted(n for n in wanted if n not in repo.sizes):
            raise models.ModelError(
                f"{repo.name} is missing files FastH3 needs: {', '.join(missing)}"
            )
        files |= {f"{repo.name}/{n}": repo.sizes[n][1] for n in sorted(wanted)}
    return files


def units(stage, variant, release, companion, run_engine, spec=None):
    """The units converting variant's entry into the staging folder stage
    from MiniMax's release and the variant's companion (hub.Repository each,
    listed) and TAEH3 (h3.PINNED), and the spec's sources; run_engine(arguments)
    runs the engine."""
    U, dit, writes = conversion.Unit, f"dit-{variant}", h3.writes
    # A spec's switches: a zero compression gate, the base's fp32 modules
    # kept fp32, the sigmas in float64.
    zero_gate = keep_f32 = sigma64 = False
    qkfix, what, extra, lora_source = QKFIX, "the FastH3 student", [KEYFRAME], None
    if spec:
        check_spec(spec)
        checkpoint, prefix = _checkpoint(spec, companion), spec["checkpoint"]["prefix"]
        source = f"{checkpoint.name}@{checkpoint.revision} {prefix.rstrip('/')}"
        found = [n for n in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json")
                 if prefix + n in checkpoint.files]  # fmt: skip
        if not found:
            raise models.ModelError(
                f"{checkpoint.name} has no {prefix} shards with an index"
            )
        index = models.read_json(checkpoint.file(prefix + found[0]))["weight_map"]
        shards = {
            f: conversion.Source(checkpoint, prefix + f)
            for f in sorted(set(index.values()))
        }
        # A diffusers trunk (Pika's transformer_ref/) names its blocks so.
        diffusers = "transformer_blocks.0.attn.to_q.weight" in index
        check_shapes(shards, diffusers)
        named_natively, Base = (
            not diffusers,
            DiffusersStudent if diffusers else LoraStudent,
        )
        zero_gate = keep_f32 = True
        sigma64 = spec["schedule"]["sigma_precision"] == "float64"
        qkfix, what = tuple(spec["qkfix"]), "the base DiT"
        extra += [1.0] * bool(spec["references"])
        ctr, prep_source = spec_contract(spec, source), source + " (base)"
    elif variant == "fasth3-8":
        reads_contract, source = companion, companion.name
        index = models.read_json(
            companion.file(
                TRANSFORMER + "diffusion_pytorch_model.safetensors.index.json"
            )
        )["weight_map"]
        shards = {
            f: conversion.Source(companion, TRANSFORMER + f)
            for f in sorted(set(index.values()))
        }
        named_natively, Base = False, Student
    else:
        reads_contract, source = _student_4(), f"{companion.name}/{LORA_FILE}"
        index = models.read_json(
            release.file(h3.TRANSFORMER + "model.safetensors.index.json")
        )["weight_map"]
        shards = {
            f: conversion.Source(release, h3.TRANSFORMER + f)
            for f in sorted(set(index.values()))
        }
        lora_source = conversion.Source(companion, LORA_FILE)
        named_natively, Base = True, LoraStudent
    if not spec:
        ctr = contract({n: models.read_json(reads_contract.file(n)) for n in CONTRACT})
        prep_source = source
    table = timesteps(ctr["dmd_rungs"], ctr["video_shift"], ctr["audio_shift"], sigma64)
    # the base lacks the VSA compression gate: a zero set_weight (LoraStudent)
    zeros = (
        ("BF16", np.zeros((H * HD, D), np.uint16)) if zero_gate else None
    )  # one array, every block's
    gates = {f"transformer_blocks.{i}.attn.to_gate_compress.set_weight": zeros for i in range(BLOCKS)} if zero_gate else {}  # fmt: skip

    def Src(t):
        # the release's fp32 globals (time embedder, patch projections, output heads) stay fp32 under keep_f32, as
        # Pika's runtime runs them: rounded to bf16 they move the adaLN tables by a bf16 ulp in half their entries,
        # and the student's step 0 by ~20 %
        return Base(t | gates, keep_f32=keep_f32)

    def by_shard(names):
        found = {}
        for name in names:
            found.setdefault(index[name], []).append(name)
        return [(shards[f], n) for f, n in found.items()]

    def reads(diffusers=(), prefixes=(), native=()):
        """The reads of tensors under diffusers names (the student's, those
        the trunk holds) and prefixes of them; for a checkpoint of native
        (FL2VA's) names, of native names and, for fasth3-4, the bundle's
        tensors of those names and prefixes."""
        if not named_natively:
            names = [*diffusers, *(n for n in index if n.startswith(tuple(prefixes)))]
            return lambda: by_shard([n for n in dict.fromkeys(names) if n in index])
        if lora_source is None:
            return lambda: by_shard(native)
        stems = tuple(n.rsplit(".", 1)[0] + "." for n in diffusers)
        bundle = [
            n for n in lora_source.tensors if n.startswith(tuple(prefixes) + stems)
        ]
        return lambda: by_shard(native) + [(lora_source, bundle)]

    def native(match):
        return [n for n in index if match(n)] if named_natively else []

    temb, tables = ".adaln/temb.npy", f"{dit}/{H3.adaln}"

    def embedding(t):
        """bf16 silu(temb) of the text schedule's timesteps, batched as a
        reference conversion batches them, then of the keyframe timestep's."""
        src = Src(t)
        st = np.concatenate(
            [silu_temb(src, table), silu_temb(src, np.array(extra, np.float32))]
        )
        data = io.BytesIO()
        np.save(data, st)
        return {temb: data.getvalue()}

    time_native = native(lambda n: n.startswith("time_embedder."))
    out = [
        U("prep inputs", (f"{dit}/prep.safetensors",),
          writes((f"{dit}/prep.safetensors",), lambda t: [prep(Src(t), variant, prep_source)]),
          reads(prefixes=PREP, native=native(lambda n: n.startswith(h3.PREP)))),
        U("time embedding", (temb,), embedding, reads(TIME, native=time_native), consumed_by=tables),
        U("patch projections", (f"{dit}/io.safetensors",),
          writes((f"{dit}/io.safetensors",), lambda t: [patches(Src(t))]),
          reads(list(IO.values()), native=native(lambda n: n in h3.IO_KEYS))),
    ]  # fmt: skip
    out += h3.decoders(stage, release)
    for i in range(BLOCKS):
        paths = (f"{dit}/fasth3-w8-block{i:02d}.safetensors",)
        paths += (f"{dit}/qkfix-block{i:02d}.safetensors",) * (i in qkfix)
        paths += (f".adaln/keyframe/block{i:02d}.safetensors",)

        def convert(t, i=i):
            src = Src(t)
            gain, _, _ = qk_gain(src, i)
            h3.check_qkfix(i, gain, qkfix, what)
            st = np.load(stage / temb)
            qkv16 = np.empty((3 * H * HD, D), np.float16) if i in qkfix else None
            tensors, rows = block(src, i, st[: -len(extra)], st[-len(extra) :], qkv16)
            files = [h3.safetensors_parts(tensors, {"format": RECIPE, "block": str(i)})]
            if qkv16 is not None:
                files.append(h3.qkfix_block(src, i, qkv16))
            # the rows past the table: the keyframe timestep's, and t = 1 of
            # a reference's (the references' tables, splash video-adaln)
            scratch = {f"blocks.{i}.adaln.keyframe": ("BF16", rows[:1])}
            if len(extra) > 1:
                scratch[f"blocks.{i}.adaln.clean"] = ("BF16", rows[1:2])
            return files + [h3.safetensors_parts(scratch)]

        out.append(U(f"block {i}", paths, writes(paths, convert),
                     reads(block_names(i), prefixes=(f"transformer_blocks.{i}.",),
                           native=native(lambda n, i=i: n.startswith(f"blocks.{i}."))),
                     consumed_by=tables))  # fmt: skip
    norms = [
        f"transformer_blocks.{i}.attn.norm_{p}.weight"
        for i in range(BLOCKS)
        for p in "qk"
    ]

    def manifest(t):
        src = Src(t)
        gains = [qk_gain(src, i)[0] for i in range(BLOCKS)]
        for i, gain in enumerate(gains):
            h3.check_qkfix(i, gain, qkfix, what)
        data = dit_manifest(variant, ctr, table, gains, src, source, qkfix)
        return [dense_manifest(data, spec) if spec else data]

    path = f"{dit}/{H3.dit_manifest}"
    out.append(U("manifest", (path,), writes((path,), manifest),
                 reads(norms, native=native(lambda n: re.fullmatch(r"blocks\.\d+\.attn\.[qk]_norm\.weight", n))),
                 ))  # fmt: skip
    if spec and spec.get("adapter"):
        adapter = conversion.Source(companion, spec["adapter"]["file"])
        out += lora_units(
            f"{dit}/{ADAPTER}",
            adapter,
            spec["adapter"],
            qkfix,
            variant,
            companion.revision,
        )
    if spec and spec["references"]:
        # the references' audio through the audio VAE's encoder (model::References)
        audio_vae = conversion.Source(release, h3.AUDIO_VAE + "model.safetensors")
        encoder = ("audio/audio_enc_f16.bin", "audio/manifest_enc.json")
        out.append(U("audio VAE encoder", encoder,
                     writes(encoder, lambda t: h3.audio_encoder(t, models.read_json(release.file(h3.AUDIO_VAE + "config.json")))),
                     h3.single(audio_vae, lambda n: n.startswith(("encoder.", "pre_block.", "mean_proj.")))))  # fmt: skip
    scratch = stage / ".adaln"

    def tabulate(_):
        (scratch / dit).mkdir(parents=True, exist_ok=True)
        linked = ["prep.safetensors", H3.dit_manifest] + [
            f"fasth3-w8-block{i:02d}.safetensors" for i in range(BLOCKS)
        ]
        for name in linked:
            (scratch / dit / name).unlink(missing_ok=True)
            os.link(stage / dit / name, scratch / dit / name)
        run_engine(["video-adaln", str(scratch), variant, str(scratch / "keyframe")])
        return {tables: scratch / dit / "adaln.safetensors"}

    out.append(U("adaLN tables", (tables,), tabulate, discard=(".adaln",)))
    return out
