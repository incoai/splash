#!/usr/bin/env python3
"""VDN-H3 DiT weights -> the engine's weight directory (<weights>/dit-<variant>).

  convert_dit.py blocks --stage DIR [--turbo ADAPTER_DIR] [--blocks 0-49] --out DIR
      Per block: base weights (MiniMaxAI/MiniMax-H3 FL2VA/transformer shards, read under diffusers names) + the
      stage's default LoRA (+ the turbo adapter: attn, ff.net.0/2, adaln_proj rank 16, norm_out rank 16), all
      adapters summed onto the base in fp32 before any layout change; Hadamard(128) rotation of every int8 linear
      (splitmix64 signs; to_out_linear uses signs 7168..14335 of the 14336-wide sequence); per-output-row int8 with
      an fp32 scale (absmax / 127). Gate projections are one int8 [256, 5376] block: rows [beta_proj 56 |
      softmax_gate.up 56 | zero 16 | output_gate.down 128]. Small tensors in fp32 / fp16 kernel layouts (VDN_LAYOUT).
      One safetensors file per block + manifest.json; the adaLN base stays in the FL2VA shards (prep reads it), its
      LoRA factors go into the block files; the final-layer LoRA factors go into vdn-globals.safetensors.
  convert_dit.py qkfix --stage DIR [--turbo DIR] --blocks 45,48,49 [--split 2] --out DIR
      fp16 q / k / v weights [21504, 5376] (merged, not rotated) of the QKFIX blocks and their fp32 split dims (the
      --split dims with the largest |q_norm * k_norm| gain product plus their RoPE partners, int32[4], -1 = unused).
  convert_dit.py io --out DIR
      patch projections and final layer (FL2VA transformer shards) -> io.safetensors
  convert_dit.py stream --out DIR [--image PATH]
      contiguous per-block image of the int8 matrices for SSD weight streaming (default DIR/stream.bin; ~22 GB)

Quantization (fp32): s[n] = max_k |W[n, k]| / 127, q[n, k] = rint(clamp(W[n, k] * (127 / max_k |W[n, k]|), -127, 127)).
Hadamard: W'[n, g] = (1/sqrt(128)) H128 (s_g (.) W[n, g]) per group g of 128 input columns, in float64;
H128[i][j] = (-1)^popcount(i & j); s[k] = -1 if splitmix64((0x48335138 << 32) + k) >> 63 else +1. The activation
side (kernels/dit/producer.metal) applies the same transform, so x'.w' = x.w.
"""
import argparse
import fcntl
import glob
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3.common import fl2va  # noqa: E402
from h3.safetensors import STFile, Source, f32_to_bf16, lora_deltas, save_st, to_f32  # noqa: E402

H, HD, D, F = 56, 128, 5376, 14336
HAD_GROUP = 128
HAD_SEED = 0x48335138  # "H3Q8"; must match model/Hadamard.hpp

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
    "adaln_proj.weight": ("BF16", "[96768, 2688] base weight, NOT merged (a bf16 merge loses the rank-16 turbo delta)"),
    "adaln_proj.bias": ("BF16", "[96768]; ada table = (W silu(temb) + lora_B (lora_A silu(temb)) + b).view(3, 6, 5376)"),
    "adaln_proj.lora_A": ("F32", "[16, 2688] turbo only"),
    "adaln_proj.lora_B": ("F32", "[96768, 16] turbo only, LoRA scale (alpha / rank) folded in"),
    "vdn.sg_bias": ("F32", "[56] softmax_gate.up.bias"),
    "vdn.og_up": ("F16", "[7168, 128] output_gate.up.weight"), "vdn.og_bias": ("F32", "[7168]"),
    "vdn.alpha_down": ("F32", "[128, 5376] alpha.down.weight, rotated like x"),
    "vdn.alpha_up": ("F32", "[7168, 128]"), "vdn.dt_bias": ("F32", "[7168]"), "vdn.A_log": ("F32", "[56]"),
    "vdn.k_sp": ("F16", "[25, 7168] short_conv.k_sp tap-major"), "vdn.k_tm": ("F16", "[5, 7168]"),
    "vdn.v_sp": ("F16", "[25, 7168]"), "vdn.v_tm": ("F16", "[5, 7168]"),
    "vdn.norm": ("F32", "[128] linear_attention.norm.weight"),
}


# ---------------------------------------------------------------- base weights
class BaseSource:
    """Base DiT weights under diffusers names, read from the MiniMaxAI FL2VA/transformer shards (native names, fused
    per-head interleaved qkv_proj)."""
    NATIVE = {"attn.norm_q.weight": "attn.q_norm.weight", "attn.norm_k.weight": "attn.k_norm.weight",
              "norm1.weight": "norm1.weight", "norm2.weight": "norm2.weight",
              "attn.to_out.0.weight": "attn.out_proj.weight", "adaln_proj.linear.weight": "adaln_proj.linear.weight",
              "adaln_proj.linear.bias": "adaln_proj.linear.bias", "ff.net.2.weight": "mlp.fc2.weight"}

    def __init__(self):
        self.native = Source(sorted(glob.glob(str(fl2va() / "transformer" / "model-*.safetensors"))))

    def _native(self, n):
        m = re.match(r"^transformer_blocks\.(\d+)\.(.*)$", n)
        if not m:
            return None
        b, rest = int(m.group(1)), m.group(2)
        if rest in self.NATIVE and self.native.has("blocks.%d.%s" % (b, self.NATIVE[rest])):
            return ("plain", "blocks.%d.%s" % (b, self.NATIVE[rest]))
        if rest == "ff.net.0.proj.weight" and self.native.has("blocks.%d.mlp.fc1.weight" % b):
            return ("swap", "blocks.%d.mlp.fc1.weight" % b)  # native [gate | value] -> diffusers [value | gate]
        mm = re.match(r"^attn\.to_([qkv])\.weight$", rest)
        if mm and self.native.has("blocks.%d.attn.qkv_proj.weight" % b):
            return ("qkv", b, "qkv".index(mm.group(1)))
        return None

    def has(self, n):
        return self._native(n) is not None

    def get(self, n):
        r = self._native(n)
        if r is None:
            raise KeyError(n)
        if r[0] == "plain":
            return self.native.get(r[1])
        if r[0] == "swap":
            w = self.native.get(r[1])
            k = w.shape[0] // 2
            return np.concatenate([w[k:], w[:k]], axis=0)
        _, b, p = r
        st, k = self.native.loc["blocks.%d.attn.qkv_proj.weight" % b]
        arr = st.get(k)[1]  # memmap [H*3*128, D] rows [head][q|k|v][128]
        rows = (np.arange(H)[:, None] * 3 * HD + p * HD + np.arange(HD)[None, :]).reshape(-1)
        return to_f32("BF16", np.asarray(arr[rows]))


# ---------------------------------------------------------------- Hadamard rotation + int8
def splitmix64(x):
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def had_signs(K, seed=HAD_SEED):
    x = (np.uint64(seed) << np.uint64(32)) + np.arange(K, dtype=np.uint64)
    return np.where((splitmix64(x) >> np.uint64(63)) != 0, -1, 1).astype(np.int8)


def hadamard128():
    i = np.arange(HAD_GROUP)
    a = i[:, None] & i[None, :]
    pc = np.zeros_like(a)
    for b in range(7):
        pc += (a >> b) & 1
    return np.where(pc % 2 == 0, 1, -1).astype(np.int64)


def had_rotate_signs(w, s):
    """W'[n, g] = (1/sqrt(128)) H (s_g (.) W[n, g]) per 128-group of input columns (float64 in and out)."""
    w = np.asarray(w, dtype=np.float64)
    N, K = w.shape
    s = np.asarray(s, dtype=np.float64).reshape(K // HAD_GROUP, HAD_GROUP)
    Hn = hadamard128().astype(np.float64) / np.sqrt(128.0)
    out = np.empty_like(w)
    for r0 in range(0, N, 4096):  # bounded temporaries for the 28672-row fc1
        g = w[r0:r0 + 4096].reshape(-1, K // HAD_GROUP, HAD_GROUP)
        out[r0:r0 + 4096] = ((g * s) @ Hn.T).reshape(-1, K)
    return out


def had_rotate(w, seed=HAD_SEED):
    return had_rotate_signs(w, had_signs(np.asarray(w).shape[1], seed))


def quant_rows(w):
    w = np.ascontiguousarray(w, dtype=np.float32)
    am = np.abs(w).max(axis=1)
    inv = np.where(am > 0, np.float32(127.0) / np.where(am > 0, am, 1), np.float32(0)).astype(np.float32)
    q = np.rint(np.clip(w * inv[:, None], np.float32(-127), np.float32(127))).astype(np.int8)
    return q, (am / np.float32(127.0)).astype(np.float32)


def swap_value_gate(w):
    """Diffusers ff.net.0.proj rows [value | gate] -> native [gate | value]."""
    n = w.shape[0] // 2
    return np.concatenate([w[n:], w[:n]], axis=0)


# ---------------------------------------------------------------- LoRA merge
def adapter_dirs(args):
    return [os.path.join(args.stage, "adapters", "default")] + ([args.turbo] if args.turbo else [])


def all_deltas(dirs):
    """base weight name -> list of (A, B, scale) over all adapters (summed, not overwritten)."""
    out = {}
    for ad in dirs:
        for k, v in lora_deltas(ad).items():
            out.setdefault(k, []).append(v)
    return out


def merged(src, name, deltas, used):
    w = src.get(name)
    for a, b, sc in deltas.get(name, []):
        w = (w + np.float32(sc) * (b.astype(np.float32) @ a.astype(np.float32))).astype(np.float32)
        used.add(name)
    return w


# ---------------------------------------------------------------- blocks
def vdn_block(src, br, i, deltas, used):
    """-> dict canonical -> (dtype, array) for block i (rotated int8 + scales, small tensors in kernel layouts)."""
    t = "transformer_blocks.%d." % i
    la = t + "attn.linear_attention."
    out = {}
    sig5 = had_signs(D)
    sig14 = had_signs(F)

    def q8(canon, w, signs):
        q, s = quant_rows(had_rotate_signs(w, signs).astype(np.float32))
        out["blocks.%d.%s.qweight" % (i, canon)] = ("I8", q)
        out["blocks.%d.%s.wscale" % (i, canon)] = ("F32", s)

    def put(canon, dt, a):
        a = np.asarray(a, dtype=np.float32)
        if dt == "BF16":
            out["blocks.%d.%s" % (i, canon)] = ("BF16", f32_to_bf16(a))
        elif dt == "F16":
            out["blocks.%d.%s" % (i, canon)] = ("F16", a.astype(np.float16))
        else:
            out["blocks.%d.%s" % (i, canon)] = ("F32", a)

    qkv = np.concatenate([merged(src, t + "attn.to_%s.weight" % p, deltas, used) for p in "qkv"], axis=0)
    q8("attn.qkv", qkv, sig5)
    del qkv
    g = np.zeros((256, D), dtype=np.float32)
    g[:56] = br.get(la + "beta_proj.weight")
    g[56:112] = br.get(t + "attn.softmax_gate.up.weight")
    g[128:] = br.get(la + "output_gate.down.weight")
    q8("attn.gates", g, sig5)
    q8("attn.out", merged(src, t + "attn.to_out.0.weight", deltas, used), sig14[:H * HD])
    q8("attn.out_linear", br.get(t + "attn.to_out_linear.weight"), sig14[H * HD:])
    q8("mlp.fc1", swap_value_gate(merged(src, t + "ff.net.0.proj.weight", deltas, used)), sig5)
    q8("mlp.fc2", merged(src, t + "ff.net.2.weight", deltas, used), sig14)
    put("norm1", "BF16", src.get(t + "norm1.weight"))
    put("norm2", "BF16", src.get(t + "norm2.weight"))
    put("attn.q_norm", "BF16", src.get(t + "attn.norm_q.weight"))
    put("attn.k_norm", "BF16", src.get(t + "attn.norm_k.weight"))
    adl = deltas.get(t + "adaln_proj.linear.weight", [])
    if adl:  # low-rank factors kept separately (summed over adapters by concatenating ranks)
        put("adaln_proj.lora_A", "F32", np.concatenate([a_ for a_, b_, sc in adl], axis=0))
        put("adaln_proj.lora_B", "F32", np.concatenate([b_ * np.float32(sc) for a_, b_, sc in adl], axis=1))
        used.add(t + "adaln_proj.linear.weight")
    put("adaln_proj.bias", "BF16", src.get(t + "adaln_proj.linear.bias"))
    put("vdn.sg_bias", "F32", br.get(t + "attn.softmax_gate.up.bias"))
    put("vdn.og_up", "F16", br.get(la + "output_gate.up.weight"))
    put("vdn.og_bias", "F32", br.get(la + "output_gate.up.bias"))
    put("vdn.alpha_down", "F32", had_rotate_signs(br.get(la + "alpha.down.weight"), sig5))
    put("vdn.alpha_up", "F32", br.get(la + "alpha.up.weight"))
    put("vdn.dt_bias", "F32", br.get(la + "alpha.dt_bias"))
    put("vdn.A_log", "F32", br.get(la + "alpha.A_log"))
    C = H * HD
    for kv in "kv":
        put("vdn.%s_sp" % kv, "F16", br.get(la + "short_conv.%s_sp.weight" % kv).reshape(C, 25).T)
        put("vdn.%s_tm" % kv, "F16", br.get(la + "short_conv.%s_tm.weight" % kv).reshape(C, 5).T)
    put("vdn.norm", "F32", br.get(la + "norm.weight"))
    return out


def parse_blocks(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load_manifest(out):
    p = os.path.join(out, "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else {"format": "h3-w8a8-vdn-v1", "files": {}}


def cmd_blocks(args):
    src = BaseSource()
    br = Source([os.path.join(args.stage, "linear_branch", "model.safetensors")])
    ads = adapter_dirs(args)
    deltas = all_deltas(ads)
    used = set()
    os.makedirs(args.out, exist_ok=True)
    blocks = parse_blocks(args.blocks)
    old = load_manifest(args.out)
    man = {"format": "h3-w8a8-vdn-v1", "base": "FL2VA", "stage": args.stage, "adapters": ads,
           "hadamard": {"group": HAD_GROUP, "seed": HAD_SEED, "signs": "splitmix64",
                        "out_linear_signs": "7168..14335 of the 14336 sequence"},
           "quant": "symmetric int8, per-output-row fp32 scale = absmax/127 (after rotation)",
           "layout": {k: list(v) for k, v in VDN_LAYOUT.items()}, "files": dict(old.get("files", {}))}
    if "qkfix" in old:
        man["qkfix"] = old["qkfix"]
    man["adaln_base_in_blocks"] = False
    man["adaln_base_source"] = ("base shards: transformer_blocks.N.adaln_proj.linear.weight (bf16, = FL2VA "
                                "blocks.N.adaln_proj.linear.weight)")
    mp = os.path.join(args.out, "manifest.json")
    for i in blocks:
        t0 = time.time()
        t = vdn_block(src, br, i, deltas, used)
        fn = "vdn-w8-block%02d.safetensors" % i
        save_st(os.path.join(args.out, fn), t, {"format": "h3-w8a8-vdn-v1", "block": str(i)})
        man["files"][str(i)] = fn
        print("block %d -> %s (%d tensors) %.1fs" % (i, fn, len(t), time.time() - t0), flush=True)
        json.dump(man, open(mp, "w"), indent=1)
    g = {}  # final-layer LoRA as factors (fp32, scale folded into B); prep adds it to the FL2VA final_layer table
    nl = deltas.get("norm_out.linear.weight", [])
    if nl:
        g["norm_out.linear.lora_A"] = ("F32", np.concatenate([a_ for a_, b_, sc in nl], axis=0))
        g["norm_out.linear.lora_B"] = ("F32", np.concatenate([b_ * np.float32(sc) for a_, b_, sc in nl], axis=1))
        used.add("norm_out.linear.weight")
    if g:
        save_st(os.path.join(args.out, "vdn-globals.safetensors"), g, {"format": "h3-w8a8-vdn-v1"})
        man["files"]["globals"] = "vdn-globals.safetensors"
    tgt = {k for k in deltas if any(k.startswith("transformer_blocks.%d." % b) for b in blocks) or k.startswith("norm_out")}
    man["lora_merged"] = sorted(set(old.get("lora_merged", [])) | used)
    man["lora_targets_not_merged"] = sorted(tgt - used)
    json.dump(man, open(mp, "w"), indent=1)
    print("BLOCKS_OK merged=%d unmerged(in selected blocks)=%d -> %s" % (len(used), len(tgt - used), args.out))


# ---------------------------------------------------------------- QKFIX
def qkfix_split_dims(gq, gk, n):
    out = []
    for d in np.argsort(-np.abs(np.asarray(gq, np.float64) * np.asarray(gk, np.float64)), kind="stable")[:n]:
        d = int(d)
        out.append(d)
        if d < 48:
            out.append(d + 48)
        elif d < 96:
            out.append(d - 48)
    out = sorted(set(out))[:4]
    return out + [-1] * (4 - len(out))


def cmd_qkfix(args):
    src = BaseSource()
    deltas = all_deltas(adapter_dirs(args))
    used = set()
    os.makedirs(args.out, exist_ok=True)
    man = load_manifest(args.out)
    qf = man.setdefault("qkfix", {"blocks": {}})
    qf["qk"] = "fp16 x fp16 -> fp32 q/k/v GEMM (merged weights, not rotated), fp32 QK-norm + RoPE epilogue"
    qf["attn"] = "fp16 Q.K + exact fp32 dot of the split dims (zeroed in the fp16 Q/K)"
    qf["base"] = "FL2VA"
    qf["stage"] = args.stage
    qf["adapters"] = ["default"] + (["turbo"] if args.turbo else [])
    for i in parse_blocks(args.blocks):
        t = "transformer_blocks.%d." % i
        w = np.concatenate([merged(src, t + "attn.to_%s.weight" % p, deltas, used) for p in "qkv"], axis=0)
        gq, gk = src.get(t + "attn.norm_q.weight"), src.get(t + "attn.norm_k.weight")
        sd = qkfix_split_dims(gq, gk, args.split)
        fn = "qkfix-block%02d.safetensors" % i
        save_st(os.path.join(args.out, fn), {"blocks.%d.qkfix.wqkv" % i: ("F16", w.astype(np.float16)),
                                            "blocks.%d.qkfix.split" % i: ("I32", np.array(sd, dtype=np.int32))},
                {"format": "h3-qkfix-v1", "block": str(i)})
        qf["blocks"][str(i)] = {"file": fn, "split": sd, "gq_max": float(np.abs(gq).max()),
                                "gk_max": float(np.abs(gk).max())}
        print("qkfix block %d -> %s split %s (|w| max %.4g, fp16 overflow %d)" % (
            i, fn, sd, float(np.abs(w).max()), int((np.abs(w) > 65504).sum())), flush=True)
    qf["lora_merged"] = sorted(used)
    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print("QKFIX_OK blocks=%s -> %s" % (args.blocks, args.out))


# ---------------------------------------------------------------- io
IO_KEYS = ["video_patch_proj.weight", "video_patch_proj.bias", "audio_patch_proj.weight", "audio_patch_proj.bias",
           "final_layer.norm.weight", "final_layer.video_out.weight", "final_layer.video_out.bias",
           "final_layer.audio_out.weight", "final_layer.audio_out.bias"]


def cmd_io(args):
    tdir = fl2va() / "transformer"
    idx = json.load(open(tdir / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for k in IO_KEYS:
        dt, a = STFile(tdir / idx[k]).get(k)
        out[k] = (dt, np.ascontiguousarray(a))
    save_st(os.path.join(args.out, "io.safetensors"), out, {"format": "h3-io-v1"})
    print("IO_OK", {k: v[0] for k, v in out.items()})


# ---------------------------------------------------------------- stream image
PAGE = 16384
F_NOCACHE = 48
STREAM_TENSORS = [("qkvg", [("attn.qkv.qweight", (21504, D)), ("attn.gates.qweight", (256, D))]),
                  ("out", [("attn.out.qweight", (D, H * HD))]),
                  ("fc1", [("mlp.fc1.qweight", (2 * F, D))]),
                  ("fc2", [("mlp.fc2.qweight", (D, F))]),
                  ("olin", [("attn.out_linear.qweight", (D, H * HD))])]
QKF_BYTES = 3 * H * HD * D * 2


def cmd_stream(args):
    """Header (16 KiB) | block 0 | ... | block 49 | QKFIX tensors; tensors 16 KiB aligned (engine/WeightStream.hpp)."""
    def al(n):
        return (n + PAGE - 1) // PAGE * PAGE

    wdir = Path(args.out)
    image = args.image or str(wdir / "stream.bin")
    man = json.load(open(wdir / "manifest.json"))
    nb = 50
    tbytes = [sum(int(np.prod(s)) for _, s in parts) for _, parts in STREAM_TENSORS]
    toff, o = [], 0
    for b in tbytes:
        toff.append(o)
        o += al(b)
    blk = o
    qk = sorted(int(b) for b in man.get("qkfix", {}).get("blocks", {}) if int(b) < nb)
    blocks_off = PAGE
    qkf_off = blocks_off + nb * blk
    qstride = al(QKF_BYTES)
    hdr = b"H3WSTRM1" + struct.pack("<IIQII", 1, nb, blk, len(STREAM_TENSORS), len(qk))
    hdr += struct.pack("<QQQ", blocks_off, qkf_off, qstride)
    hdr += struct.pack("<5Q", *toff) + struct.pack("<5Q", *tbytes)
    for i, b in enumerate(qk):
        hdr += struct.pack("<iiQ", b, 0, qkf_off + i * qstride)
    assert len(hdr) <= PAGE
    total = qkf_off + len(qk) * qstride
    tmp = image + ".part"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    t0 = time.perf_counter()

    def wr(off, buf):
        mv = memoryview(buf).cast("B")
        n = 0
        while n < len(mv):
            n += os.pwrite(fd, mv[n:n + (64 << 20)], off + n)

    wr(0, hdr.ljust(PAGE, b"\0"))
    for i in range(nb):
        st = STFile(wdir / man["files"][str(i)])
        base = blocks_off + i * blk
        for (name, parts), off, nbytes in zip(STREAM_TENSORS, toff, tbytes):
            buf = bytearray(al(nbytes))
            p = 0
            for k, shp in parts:
                dt, arr = st.get(f"blocks.{i}.{k}")
                assert tuple(arr.shape) == shp and arr.dtype == np.int8, (i, k, arr.shape, arr.dtype)
                buf[p:p + arr.nbytes] = memoryview(np.ascontiguousarray(arr)).cast("B")
                p += arr.nbytes
            wr(base + off, buf)
        print(f"block {i} done {time.perf_counter() - t0:.1f}s", flush=True)
    for j, b in enumerate(qk):
        st = STFile(wdir / man["qkfix"]["blocks"][str(b)]["file"])
        dt, arr = st.get(f"blocks.{b}.qkfix.wqkv")
        assert arr.nbytes == QKF_BYTES, arr.shape
        buf = bytearray(qstride)
        buf[:QKF_BYTES] = memoryview(np.ascontiguousarray(arr)).cast("B")
        wr(qkf_off + j * qstride, buf)
    os.ftruncate(fd, total)
    os.fsync(fd)
    os.close(fd)
    os.replace(tmp, image)
    dt = time.perf_counter() - t0
    print(f"STREAM_OK {image} blocks={nb} block_bytes={blk} qkfix={qk} total={total / 1e9:.2f} GB in {dt:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("blocks")
    b.add_argument("--stage", required=True, help="OpenVDN stage dir (linear_branch/, adapters/default/)")
    b.add_argument("--turbo", help="extra adapter dir (stage-dmd-step-250/adapters/turbo)")
    b.add_argument("--blocks", default="0-49")
    b.add_argument("--out", required=True)
    q = sp.add_parser("qkfix")
    q.add_argument("--stage", required=True)
    q.add_argument("--turbo")
    q.add_argument("--blocks", default="45,48,49")
    q.add_argument("--split", type=int, default=2, help="gain-product dims kept fp32 in attention (+ RoPE partners)")
    q.add_argument("--out", required=True)
    i = sp.add_parser("io")
    i.add_argument("--out", required=True)
    s = sp.add_parser("stream")
    s.add_argument("--out", required=True, help="converted weight dir (manifest.json)")
    s.add_argument("--image", help="default <out>/stream.bin")
    a = ap.parse_args()
    {"blocks": cmd_blocks, "qkfix": cmd_qkfix, "io": cmd_io, "stream": cmd_stream}[a.cmd](a)


if __name__ == "__main__":
    main()
