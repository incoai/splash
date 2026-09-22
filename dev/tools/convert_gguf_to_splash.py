#!/usr/bin/env python3
"""Convert a Qwen3.8-27B GGUF (llama.cpp K-quants) into a splash 'gguf-kquant' target package.

Package layout (16 KiB-aligned sections, header 16 B = magic 'MDKQ0001' + layer u32 + type u32):
  layer-N.bin (GDN):  input_norm bf16[H] | KQ qkv | KQ z | KQ ab(Q8_0, 256 rows: beta 48, alpha 48, zeros) | conv bf16[10240][4] |
                      decay f32[48] | dt_bias bf16[48] | mixer_norm bf16[128] | KQ out | post_norm bf16[H] | KQ gate | KQ up | KQ down
  layer-N.bin (attn): input_norm | KQ q | KQ k | KQ v | q_norm bf16[256] | k_norm bf16[256] | KQ attn_out | post_norm | KQ gate | KQ up | KQ down
  head.bin:           final_norm bf16[H] | KQ lm_head
  embedding.bin:      KQ-desc(native) | native block_q4_K rows [V][2880]
A 'KQ' projection = descriptor section (64 B) + plane0 + plane1 (if any) + meta, each 16 KiB aligned.
Head order: splash uses HF grouped value heads; GGUF stores llama.cpp's tiled order -> rows of v/z/alpha/beta/conv/decay/dt_bias are
un-reordered here; out_proj keeps its tiled K order (the runtime permutes the activation before that GEMM).
"""

import argparse
import json
import os
import struct

import gguf
import numpy as np

H, VOCAB, LAYERS = 5120, 248320, 64
NK, NV, HD = 16, 48, 128  # GDN key heads, value heads, head dim
ALIGN = 16384
MAGIC = b"MDKQ0001"
TYPE_ID = {
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "Q3_K": 11,
    "Q8_0": 8,
    "IQ4_XS": 23,
    "IQ4_NL": 20,
    "IQ3_S": 21,
}
FMT = {  # blockK, blockBytes, p0, p1, metaBytes, metaGroups, interleave nibbles
    "Q4_K": (256, 144, 16, 0, 16, 8, True),
    "IQ4_XS": (256, 136, 16, 0, 8, 8, False),
    "IQ4_NL": (32, 18, 16, 0, 2, 1, True),
    "Q5_K": (256, 176, 16, 4, 16, 8, True),
    "Q6_K": (256, 210, 16, 8, 20, 8, True),
    "Q3_K": (256, 110, 8, 4, 16, 8, True),
    "Q8_0": (32, 34, 32, 0, 2, 1, True),
    "IQ3_S": (256, 110, 16, 0, 2, 8, True),
}


def bf16(a):
    a = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((a + 0x7FFF + ((a >> 16) & 1)) >> 16).astype(np.uint16)


def unreorder_rows(x):
    """tiled (j*NK + i) -> grouped (i*3 + j) for value-head-major rows; x: [NV*hd, ...] with hd = x.shape[0] // NV"""
    hd = x.shape[0] // NV
    v = x.reshape(3, NK, hd, *x.shape[1:])
    return np.ascontiguousarray(
        v.transpose(1, 0, 2, *range(3, v.ndim)).reshape(x.shape)
    )


def pack_words(codes, interleave):
    """codes: [..., 32] uint8 (values < 16) -> [..., 16] bytes"""
    c = codes.astype(np.uint32).reshape(*codes.shape[:-1], 4, 8)
    if interleave:
        lanes = np.array([(k & 1) * 16 + 4 * (k // 2) for k in range(8)], np.uint32)
    else:
        lanes = np.array([4 * k for k in range(8)], np.uint32)
    words = (c << lanes).sum(-1, dtype=np.uint32)
    return words.astype("<u4").view(np.uint8).reshape(*codes.shape[:-1], 16)


def bits_word(bits, even_shift, odd_shift):
    """bits: [..., 32] -> one uint32 word per 32 with bit positions given by per-weight shift arrays"""
    b = bits.astype(np.uint32)
    k = np.arange(32)
    sh = np.where(k % 2 == 0, even_shift(k), odd_shift(k)).astype(np.uint32)
    return (b << sh).sum(-1, dtype=np.uint32)


def repack(ty, raw, N, K):
    """raw: native rows [N, rowbytes] -> (plane0 [N,G,p0], plane1 [N,G,p1] or None, meta [N,units,mb])"""
    blockK, bb, p0, p1, mb, mg, inter = FMT[ty]
    nb = K // blockK
    G = K // 32
    blk = raw.reshape(N, nb, bb)
    if ty in ("Q4_K", "Q5_K"):
        qs = blk[:, :, 48:176] if ty == "Q5_K" else blk[:, :, 16:144]
        codes = np.stack(
            [
                (qs[:, :, (j // 2) * 32 : (j // 2) * 32 + 32] >> (4 * (j % 2))) & 15
                for j in range(8)
            ],
            2,
        )  # [N,nb,8,32]
        plane0 = pack_words(codes, True).reshape(N, G, 16)
        meta = np.ascontiguousarray(blk[:, :, :16])
        if ty == "Q5_K":
            qh = blk[:, :, 16:48]
            hb = np.stack([(qh >> j) & 1 for j in range(8)], 2)  # [N,nb,8,32]
            w = bits_word(
                hb,
                lambda k: 4 * (k // 8) + (k % 8) // 2,
                lambda k: 16 + 4 * (k // 8) + (k % 8) // 2,
            )
            plane1 = w.astype("<u4").view(np.uint8).reshape(N, G, 4)
        else:
            plane1 = None
    elif ty == "IQ4_XS":
        qs = blk[:, :, 8:136].reshape(N, nb, 8, 16)
        codes = np.concatenate([qs & 15, qs >> 4], -1)  # [N,nb,8,32]
        plane0 = pack_words(codes, inter).reshape(N, G, 16)
        plane1 = None
        meta = np.ascontiguousarray(blk[:, :, :8])
    elif ty == "IQ4_NL":
        qs = blk[:, :, 2:18]
        codes = np.concatenate([qs & 15, qs >> 4], -1)  # [N,G,32]
        plane0 = pack_words(codes, inter).reshape(N, G, 16)
        plane1 = None
        meta = np.ascontiguousarray(blk[:, :, :2])
    elif ty == "Q6_K":
        ql, qh, sc, d = (
            blk[:, :, 0:128],
            blk[:, :, 128:192],
            blk[:, :, 192:208],
            blk[:, :, 208:210],
        )
        lo, hi = [], []
        for j in range(8):
            n, r = j // 4, j % 4
            lo.append(
                (
                    ql[:, :, 64 * n + 32 * (r & 1) : 64 * n + 32 * (r & 1) + 32]
                    >> (4 * (r >> 1))
                )
                & 15
            )
            hi.append((qh[:, :, 32 * n : 32 * n + 32] >> (2 * r)) & 3)
        lo = np.stack(lo, 2)
        hi = np.stack(hi, 2)  # [N,nb,8,32]
        plane0 = pack_words(lo, True).reshape(N, G, 16)
        # two words: weights 0-15 and 16-31; within a half, pair index ip = (k%16)//2: even -> bit 2ip, odd -> bit 16+2ip
        words = []
        for hh in range(2):
            part = hi[..., 16 * hh : 16 * hh + 16].astype(np.uint32)
            k = np.arange(16)
            sh = np.where(k % 2 == 0, 2 * (k // 2), 16 + 2 * (k // 2)).astype(np.uint32)
            words.append((part << sh).sum(-1, dtype=np.uint32))
        plane1 = np.stack(words, -1).astype("<u4").view(np.uint8).reshape(N, G, 8)
        meta = np.concatenate([sc, d, np.zeros((N, nb, 2), np.uint8)], -1)
    elif ty == "Q3_K":
        hm, qs, scales, d = (
            blk[:, :, 0:32],
            blk[:, :, 32:96],
            blk[:, :, 96:108],
            blk[:, :, 108:110],
        )
        c2, hb = [], []
        for j in range(8):
            n, jj = j // 4, j % 4
            c2.append((qs[:, :, 32 * n : 32 * n + 32] >> (2 * jj)) & 3)
            hb.append((hm >> j) & 1)
        c2 = np.stack(c2, 2)
        hb = np.stack(hb, 2)
        words = []
        for hh in range(2):
            part = c2[..., 16 * hh : 16 * hh + 16].astype(np.uint32)
            k = np.arange(16)
            sh = np.where(k % 2 == 0, 2 * (k // 2), 16 + 2 * (k // 2)).astype(np.uint32)
            words.append((part << sh).sum(-1, dtype=np.uint32))
        plane0 = np.stack(words, -1).astype("<u4").view(np.uint8).reshape(N, G, 8)
        hw = bits_word(hb, lambda k: k // 2, lambda k: 16 + k // 2)
        plane1 = hw.astype("<u4").view(np.uint8).reshape(N, G, 4)
        meta = np.concatenate([d, np.zeros((N, nb, 2), np.uint8), scales], -1)
    elif ty == "Q8_0":
        plane0 = np.ascontiguousarray(blk[:, :, 2:34])
        plane1 = None
        meta = np.ascontiguousarray(blk[:, :, 0:2])
    elif ty == "IQ3_S":
        d, qs, qh, signs, scales = (
            blk[:, :, 0:2],
            blk[:, :, 2:66].reshape(N, nb, 8, 8),
            blk[:, :, 66:74],
            blk[:, :, 74:106].reshape(N, nb, 8, 4),
            blk[:, :, 106:110],
        )
        scn = np.stack(
            [(scales[:, :, j // 2] >> (4 * (j % 2))) & 15 for j in range(8)], 2
        )  # [N,nb,8]
        plane0 = np.concatenate(
            [
                qs,
                signs,
                qh[..., None],
                scn[..., None],
                np.zeros((N, nb, 8, 2), np.uint8),
            ],
            -1,
        ).reshape(N, G, 16)
        plane1 = None
        meta = np.ascontiguousarray(d)
    else:
        raise ValueError(ty)
    return plane0, plane1, meta


def to_tile_order(arr):
    """[N, G, P] -> [tiles, G, 256, P] flattened"""
    N, G, P = arr.shape
    return np.ascontiguousarray(
        arr.reshape(N // 256, 256, G, P).transpose(0, 2, 1, 3)
    ).reshape(-1)


class Writer:
    def __init__(self, path, layer, ltype):
        self.f = open(path, "wb")
        self.f.write(MAGIC + struct.pack("<II", layer, ltype))
        self.off = 16
        self.records = []

    def section(self, data, label=""):
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        start = (self.off + ALIGN - 1) // ALIGN * ALIGN
        self.f.write(b"\0" * (start - self.off))
        self.f.write(data.tobytes())
        self.off = start + data.nbytes
        self.records.append((label, start, data.nbytes))

    def kq(self, ty, raw, N, K, label):
        blockK, bb, p0, p1, mb, mg, inter = FMT[ty]
        plane0, plane1, meta = repack(ty, raw, N, K)
        w0 = to_tile_order(plane0)
        w1 = to_tile_order(plane1) if plane1 is not None else np.zeros(0, np.uint8)
        units = K // 32 // mg
        mt = np.ascontiguousarray(
            meta.reshape(N // 256, 256, units, mb).transpose(0, 2, 1, 3)
        ).reshape(-1)
        desc = (
            struct.pack("<8I", TYPE_ID[ty], N, K, p0, p1, mb, mg, 1 if inter else 0)
            + struct.pack("<3Q", w0.nbytes, w1.nbytes, mt.nbytes)
            + b"\0" * 8
        )
        self.section(np.frombuffer(desc, np.uint8), label + "-desc")
        self.section(w0, label + "-plane0")
        if w1.nbytes:
            self.section(w1, label + "-plane1")
        self.section(mt, label + "-meta")

    def close(self):
        end = (self.off + ALIGN - 1) // ALIGN * ALIGN
        self.f.write(b"\0" * (end - self.off))
        self.f.close()
        return end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--skip-embedding", action="store_true")
    ap.add_argument("--skip-head", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "target"), exist_ok=True)
    r = gguf.GGUFReader(a.gguf)
    T = {t.name: t for t in r.tensors}

    def raw(name):
        t = T[name]
        return (
            np.ascontiguousarray(t.data).reshape(int(t.shape[1]), -1),
            t.tensor_type.name,
            int(t.shape[1]),
            int(t.shape[0]),
        )

    def f32(name):
        return np.asarray(T[name].data).astype(np.float32).reshape(-1)

    layers = (
        range(LAYERS) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    )
    types_used = {}
    for L in layers:
        p = f"blk.{L}."
        full = (L + 1) % 4 == 0
        w = Writer(os.path.join(a.out, "target", f"layer-{L}.bin"), L, 1 if full else 0)
        w.section(bf16(f32(p + "attn_norm.weight")), "input-norm")
        if not full:
            qkv, ty, N, K = raw(p + "attn_qkv.weight")
            assert (N, K) == (10240, H)
            qkv = np.concatenate([qkv[:4096], unreorder_rows(qkv[4096:])], 0)
            w.kq(ty, qkv, N, K, "gdn-qkv")
            types_used[p + "attn_qkv"] = ty
            z, ty, N, K = raw(p + "attn_gate.weight")
            w.kq(ty, unreorder_rows(z), N, K, "gdn-z")
            types_used[p + "attn_gate"] = ty
            al, tya, _, _ = raw(p + "ssm_alpha.weight")
            be, tyb, _, _ = raw(p + "ssm_beta.weight")
            assert tya == tyb == "Q8_0"
            ab = np.zeros((256, al.shape[1]), np.uint8)
            ab[:48] = unreorder_rows(be)
            ab[48:96] = unreorder_rows(al)
            w.kq("Q8_0", ab, 256, H, "gdn-ab")
            conv = f32(p + "ssm_conv1d.weight").reshape(10240, 4)
            conv = np.concatenate([conv[:4096], unreorder_rows(conv[4096:])], 0)
            w.section(bf16(conv), "gdn-conv")
            w.section(
                unreorder_rows(f32(p + "ssm_a").reshape(48, 1))
                .reshape(-1)
                .astype(np.float32),
                "gdn-decay",
            )
            w.section(
                bf16(unreorder_rows(f32(p + "ssm_dt.bias").reshape(48, 1)).reshape(-1)),
                "gdn-time-bias",
            )
            w.section(bf16(f32(p + "ssm_norm.weight")), "gdn-norm")
            o, ty, N, K = raw(p + "ssm_out.weight")
            assert (N, K) == (H, 6144)
            w.kq(ty, o, N, K, "gdn-output")
            types_used[p + "ssm_out"] = ty
        else:
            for nm, lab in (
                ("attn_q", "attn-q"),
                ("attn_k", "attn-k"),
                ("attn_v", "attn-v"),
            ):
                x, ty, N, K = raw(p + nm + ".weight")
                w.kq(ty, x, N, K, lab)
                types_used[p + nm] = ty
            w.section(bf16(f32(p + "attn_q_norm.weight")), "query-norm")
            w.section(bf16(f32(p + "attn_k_norm.weight")), "key-norm")
            o, ty, N, K = raw(p + "attn_output.weight")
            w.kq(ty, o, N, K, "attn-output")
            types_used[p + "attn_output"] = ty
        w.section(bf16(f32(p + "post_attention_norm.weight")), "post-norm")
        for nm, lab in (
            ("ffn_gate", "mlp-gate"),
            ("ffn_up", "mlp-up"),
            ("ffn_down", "mlp-down"),
        ):
            x, ty, N, K = raw(p + nm + ".weight")
            w.kq(ty, x, N, K, lab)
            types_used[p + nm] = ty
        size = w.close()
        print(f"layer {L} {'attn' if full else 'gdn'} {size / 1e6:.1f} MB", flush=True)
    if not a.skip_head:
        w = Writer(os.path.join(a.out, "target", "head.bin"), LAYERS, 2)
        w.section(bf16(f32("output_norm.weight")), "final-norm")
        x, ty, N, K = raw("output.weight")
        w.kq(ty, x, N, K, "logits")
        types_used["output"] = ty
        print("head", w.close() / 1e6, "MB", ty)
    if not a.skip_embedding:
        w = Writer(os.path.join(a.out, "target", "embedding.bin"), VOCAB, H)
        x, ty, N, K = raw("token_embd.weight")
        # Native rows, gathered by kq_embed_<type>; the descriptor carries the type.
        assert ty in ("Q4_K", "Q6_K", "Q8_0"), f"unsupported embedding type {ty}"
        types_used["token_embd"] = ty
        desc = (
            struct.pack("<8I", TYPE_ID[ty], N, K, 0, 0, 0, 0, 0)
            + struct.pack("<3Q", x.nbytes, 0, 0)
            + b"\0" * 8
        )
        w.section(np.frombuffer(desc, np.uint8), "embedding-desc")
        w.section(x, "embedding-native")
        print("embedding", w.close() / 1e6, "MB")
    json.dump(
        types_used, open(os.path.join(a.out, "target", "types.json"), "w"), indent=1
    )


if __name__ == "__main__":
    main()
