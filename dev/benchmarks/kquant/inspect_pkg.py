# Reverse-engineer splash package conventions from target/layer-0.bin (GDN layer) vs the GGUF's tensors.
import glob
import struct

import gguf
import numpy as np

snap = glob.glob(
    "/Users/liang2kl/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-Splash/snapshots/*/"
)[0]
r = gguf.GGUFReader("/Users/liang2kl/dev/q4k-m5/models/Qwen3.8-27B-UD-Q4_K_M.gguf")
T = {t.name: t for t in r.tensors}


def deq(name):
    t = T[name]
    if t.tensor_type.name in ("F32", "F16"):
        return np.asarray(t.data).astype(np.float32)
    return gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)


def bf16(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


ALIGN = 16384


class Reader:
    def __init__(self, path):
        self.m = np.memmap(path, dtype=np.uint8, mode="r")
        self.off = 16
        print(
            path,
            len(self.m),
            "header",
            bytes(self.m[:8]),
            struct.unpack("<II", bytes(self.m[8:16])),
        )

    def section(self, nbytes):
        start = (self.off + ALIGN - 1) // ALIGN * ALIGN
        self.off = start + nbytes
        return self.m[start : start + nbytes]


def q4_packed_bytes(n, k):
    return n * k // 2 + 2 * (n * (k // 64) * 2)


def deq_q4(sec, n, k, rows):
    """dequantize selected rows of a splash-packed-q4 projection (n x k) -> float32 [len(rows), k]"""
    G = k // 64
    wb = n * k // 2
    pb = n * G * 2
    w = np.asarray(sec[:wb])
    sc = bf16(np.frombuffer(bytes(sec[wb : wb + pb]), dtype=np.uint16))
    bi = bf16(np.frombuffer(bytes(sec[wb + pb : wb + 2 * pb]), dtype=np.uint16))
    out = np.zeros((len(rows), k), np.float32)
    for i, row in enumerate(rows):
        tile, c = row // 256, row % 256
        idx = (tile * G + np.arange(G)) * 256 + c  # per group
        blocks = w.reshape(-1, 32)[idx]  # [G, 32 bytes]
        q = np.stack([blocks & 15, blocks >> 4], -1).reshape(G, 64).astype(np.float32)
        out[i] = (q * sc[idx][:, None] + bi[idx][:, None]).reshape(-1)
    return out


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-30))


rd = Reader(snap + "target/layer-0.bin")
H, PG, CONV, VH, HD, AW, INTER = 5120, 16640, 10240, 48, 128, 6144, 17408
norm = bf16(np.frombuffer(bytes(rd.section(H * 2)), dtype=np.uint16))
gn = deq("blk.0.attn_norm.weight")
print(
    "input norm: splash vs gguf   maxdiff",
    np.abs(norm - gn).max(),
    "| vs gguf-1",
    np.abs(norm - (gn - 1)).max(),
    "| splash mean",
    norm.mean(),
    "gguf mean",
    gn.mean(),
)
gdn_in = rd.section(q4_packed_bytes(PG, H))
conv = bf16(np.frombuffer(bytes(rd.section(CONV * 4 * 2)), dtype=np.uint16)).reshape(
    CONV, 4
)
gc = deq("blk.0.ssm_conv1d.weight").reshape(CONV, 4)
print("conv1d: direct maxdiff", np.abs(conv - gc).max(), "rms", np.sqrt((gc**2).mean()))


def unreorder_rows(
    x, n_k=16, n_v_per_k=3, hd=HD
):  # inverse of llama.cpp grouped->tiled: tiled index (j, i) -> grouped (i, j)
    v = x.reshape(n_v_per_k, n_k, hd, *x.shape[1:])
    return v.transpose(1, 0, 2, *range(3, v.ndim)).reshape(x.shape)


gc2 = np.concatenate([gc[:4096], unreorder_rows(gc[4096:])], 0)
print("conv1d: v-unreordered maxdiff", np.abs(conv - gc2).max())
decay = np.frombuffer(bytes(rd.section(VH * 4)), dtype=np.float32)
ga = deq("blk.0.ssm_a").reshape(-1)
for label, cand in [
    ("ssm_a as is", ga),
    ("log(-ssm_a)=A_log", np.log(-ga)),
    ("-ssm_a", -ga),
    ("unreordered ssm_a", unreorder_rows(ga.reshape(-1, 1), hd=1).reshape(-1)),
    ("unreordered A_log", np.log(-unreorder_rows(ga.reshape(-1, 1), hd=1).reshape(-1))),
]:
    print(f"decay vs {label:<20} maxdiff {np.abs(decay - cand).max():.4g}")
print("decay[:6]", decay[:6], "ssm_a[:6]", ga[:6])
tb = bf16(np.frombuffer(bytes(rd.section(VH * 2)), dtype=np.uint16))
gdt = deq("blk.0.ssm_dt.bias").reshape(-1)
print(
    "time bias: direct maxdiff",
    np.abs(tb - gdt).max(),
    "unreordered",
    np.abs(tb - unreorder_rows(gdt.reshape(-1, 1), hd=1).reshape(-1)).max(),
)
mn = bf16(np.frombuffer(bytes(rd.section(HD * 2)), dtype=np.uint16))
gmn = deq("blk.0.ssm_norm.weight")
print(
    "mixer norm: vs gguf maxdiff",
    np.abs(mn - gmn).max(),
    "vs gguf-1",
    np.abs(mn - (gmn - 1)).max(),
    "vs gguf+1",
    np.abs(mn - (gmn + 1)).max(),
)
gdn_out = rd.section(q4_packed_bytes(H, AW))
post = bf16(np.frombuffer(bytes(rd.section(H * 2)), dtype=np.uint16))
gp = deq("blk.0.post_attention_norm.weight")
print(
    "post norm: vs gguf maxdiff",
    np.abs(post - gp).max(),
    "vs gguf-1",
    np.abs(post - (gp - 1)).max(),
)
gate = rd.section(q4_packed_bytes(INTER, H))
up = rd.section(q4_packed_bytes(INTER, H))
down = rd.section(q4_packed_bytes(H, INTER))
print("consumed", (rd.off + ALIGN - 1) // ALIGN * ALIGN, "file", len(rd.m))
# gdn input projection rows: q (0..2047), k (2048..4095), v (4096..10239), z (10240..16383), a/b (16384..16639)
qkv = deq("blk.0.attn_qkv.weight")
z = deq("blk.0.attn_gate.weight")
a = deq("blk.0.ssm_alpha.weight")
b = deq("blk.0.ssm_beta.weight")
print("gguf shapes qkv", qkv.shape, "z", z.shape, "a", a.shape, "b", b.shape)
rows = [
    0,
    1,
    2048,
    4096,
    4096 + 128,
    4096 + 256,
    4096 + 3 * 128,
    10240,
    10240 + 128,
    10240 + 3 * 128,
    16384,
    16385,
    16384 + 48,
    16384 + 64,
    16384 + 128,
    16384 + 176,
]
sp = deq_q4(gdn_in, PG, H, rows)
v_tiled = qkv[4096:]
v_grouped = unreorder_rows(v_tiled)
z_grouped = unreorder_rows(z)
for i, row in enumerate(rows):
    s = sp[i]
    cands = {}
    if row < 4096:
        cands["qkv same row"] = qkv[row]
    elif row < 10240:
        cands["v tiled"] = v_tiled[row - 4096]
        cands["v grouped(HF)"] = v_grouped[row - 4096]
    elif row < 16384:
        cands["z tiled"] = z[row - 10240]
        cands["z grouped(HF)"] = z_grouped[row - 10240]
    else:
        j = row - 16384
        for name, mat, base in (("a", a, 0), ("b", b, 0)):
            for jj in (j, j - 48, j - 64, j - 128, j - 176, j - 192):
                if 0 <= jj < 48:
                    cands[f"{name}[{jj}]"] = mat[jj]
            for jj in (j, j - 48, j - 64, j - 128, j - 176, j - 192):
                if 0 <= jj < 48:
                    cands[f"{name} grouped[{jj}]"] = unreorder_rows(
                        mat.reshape(48, 1, -1), hd=1
                    ).reshape(48, -1)[jj]
    best = sorted(((corr(s, c), n) for n, c in cands.items()), reverse=True)[:2]
    print(
        f"gdn_in row {row:5d}: |row|rms {np.sqrt((s**2).mean()):.4g}  best",
        ", ".join(f"{n} corr {c:.3f}" for c, n in best),
    )
# gdn output projection: K order (columns): compare dequantized splash rows vs gguf ssm_out rows, with column reorder hypotheses
go = deq("blk.0.ssm_out.weight")  # [5120, 6144] HF-shaped rows x K(tiled)
so = deq_q4(gdn_out, H, AW, [0, 1, 7])


def unreorder_cols(x):
    return unreorder_rows(x.T).T


for i, row in enumerate([0, 1, 7]):
    print(
        f"gdn_out row {row}: corr tiled {corr(so[i], go[row]):.3f}  grouped(HF) {corr(so[i], unreorder_cols(go)[row]):.3f}"
    )
