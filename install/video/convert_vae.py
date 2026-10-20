"""MiniMax-H3 video VAE decoder (FL2VA/video_vae/source/model.safetensors, fp32) -> the engine's weights (<weights>/vae).

  convert_vae.py --src <FL2VA/video_vae dir> --out weights/vae --tag w8 --i8 qkv,o,1,2     (shipping, W8A8)
  convert_vae.py --src <FL2VA/video_vae dir> --out weights/vae --tag f16                   (all linears fp16)

One flat little-endian binary per variant (vae_dec_<tag>.bin) + manifest_<tag>.json (byte offsets; every
tensor 256-byte aligned). Decoder tensors only (post_quant_conv is folded into x_embedder in fp64).
Per layer (36):
  f16: wqkv f16 [6144, 2048] (rows reordered from per-head interleaved (h, part, d) to (part, h, d)), bqkv f32,
       wo f16 [2048, 2048], bo, s1 (scale1), w1 f16 [16384, 2048] (rows [gate; up], native), b1, w2 f16 [2048, 8192],
       b2, s2 (scale2), n1, n2 (RMSNorm weights)   -- all small tensors fp32
  W8 (linears named in --i8): the same with int8 weights + per-output-row fp32 scales (sqkv, so, sw1, sw2) after the QuaRot-style
       Hadamard(128) input rotation of tools/convert_dit.py (had_rotate / quant_rows / had_signs, imported).
Globals: we f32 [2048, 24] (= We Wpq), be f32 [2048] (= We bpq + be), reg f32 [4, 2048], lnw, lnb f32 [2048],
  wp f16 [3072, 2048] (proj_out, fp16 in both precisions), bp f32 [3072], cos / sin f32 [1797, 48] (RoPE table of a
  7 x 16 x 16 tile-clip + 5 suffix rows at position 0, computed with the MLX port's ViTRotaryPosEmbed on the CPU),
  sg2048 / sg8192 int8 (Hadamard sign vectors).
"""
import argparse, json, struct, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_dit import had_rotate, quant_rows, had_signs, HAD_SEED  # noqa: E402

L, D, H, HD, F = 36, 2048, 32, 64, 8192
S = 7 * 16 * 16 + 5


class ST:
    def __init__(self, path):
        f = open(path, "rb")
        n = struct.unpack("<Q", f.read(8))[0]
        self.h = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, k):
        e = self.h[k]
        assert e["dtype"] == "F32", (k, e["dtype"])
        a, b = e["data_offsets"]
        return np.frombuffer(self.mm[self.base + a:self.base + b], dtype=np.float32).reshape(e["shape"])


class Blob:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.off = 0

    def put(self, a):
        a = np.ascontiguousarray(a)
        pad = (-self.off) % 256
        if pad:
            self.f.write(b"\0" * pad)
            self.off += pad
        o = self.off
        self.f.write(a.tobytes())
        self.off += a.nbytes
        return o


def qkv_perm():
    return np.array([h * 192 + part * 64 + d for part in range(3) for h in range(H) for d in range(HD)])


def f16(w, name):
    am = float(np.abs(w).max())
    assert am < 60000, (name, am)
    return w.astype(np.float16)


def mlx_port() -> Path:
    """The MiniMax-H3 MLX port checkout (H3_MLX_PORT, else h3/config.py's default); exits when it is missing."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from h3.config import MLX_PORT
    if not (MLX_PORT / "minimax_h3_mlx").is_dir():
        sys.exit(f"convert_vae.py: the MLX port (minimax_h3_mlx) is not at {MLX_PORT}; set H3_MLX_PORT to a "
                 "checkout of https://github.com/PipeNetwork/minimax-h3-mlx (needed for the RoPE tables)")
    return MLX_PORT


def rope_tables():
    port = str(mlx_port())
    if port not in sys.path:
        sys.path.insert(0, port)
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from minimax_h3_mlx.video_vae import ViTRotaryPosEmbed
    rope = ViTRotaryPosEmbed(48, theta=100.0)
    d, h, w = 7, 16, 16
    grids = [2.0 * ((mx.arange(s, dtype=mx.float32) + 0.5) / s) - 1.0 for s in (d, h, w)]
    tt = mx.broadcast_to(grids[0].reshape(d, 1, 1), (d, h, w))
    hh = mx.broadcast_to(grids[1].reshape(1, h, 1), (d, h, w))
    ww = mx.broadcast_to(grids[2].reshape(1, 1, w), (d, h, w))
    pos = mx.concatenate([mx.stack([tt, hh, ww], axis=-1).reshape(1, d * h * w, 3), mx.zeros((1, 5, 3))], axis=1)
    c, s = rope(pos)
    return np.array(c[0, :, 0]).astype(np.float32), np.array(s[0, :, 0]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True, help="output name: vae_dec_<tag>.bin / manifest_<tag>.json")
    ap.add_argument("--i8", default="", help="comma list of linears stored W8 (Hadamard + int8): qkv,o,1,2; "
                    "empty = all fp16")
    a = ap.parse_args()
    st = ST(Path(a.src) / "source" / "model.safetensors")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    i8set = set(x for x in a.i8.split(",") if x)
    assert i8set <= {"qkv", "o", "1", "2"}, i8set
    blob = Blob(out / f"vae_dec_{a.tag}.bin")
    man = {"tag": a.tag, "i8": sorted(i8set), "layers": [], "globals": {}, "S": S, "had_seed": HAD_SEED,
           "src": str(Path(a.src).resolve() / "source" / "model.safetensors")}
    t0 = time.time()
    g = man["globals"]
    we = st.get("decoder.x_embedder.weight").astype(np.float64)
    be = st.get("decoder.x_embedder.bias").astype(np.float64)
    wpq = st.get("post_quant_conv.weight").astype(np.float64).reshape(24, 24)
    bpq = st.get("post_quant_conv.bias").astype(np.float64)
    g["we"] = blob.put((we @ wpq).astype(np.float32))
    g["be"] = blob.put((we @ bpq + be).astype(np.float32))
    g["reg"] = blob.put(st.get("decoder.register_tokens").reshape(4, D))
    g["lnw"] = blob.put(st.get("decoder.norm_out.weight"))
    g["lnb"] = blob.put(st.get("decoder.norm_out.bias"))
    g["wp"] = blob.put(f16(st.get("decoder.proj_out.weight"), "proj_out"))
    g["bp"] = blob.put(st.get("decoder.proj_out.bias"))
    cs, sn = rope_tables()
    assert cs.shape == (S, 48)
    g["cos"] = blob.put(cs)
    g["sin"] = blob.put(sn)
    g["sg2048"] = blob.put(had_signs(2048))
    g["sg8192"] = blob.put(had_signs(8192))
    perm = qkv_perm()
    stats = []
    for i in range(L):
        p = f"decoder.transformer_blocks.{i}."
        rec = {}
        wqkv = st.get(p + "attn.to_qkv.weight")[perm]
        bqkv = st.get(p + "attn.to_qkv.bias")[perm]
        wo, bo = st.get(p + "attn.to_out.weight"), st.get(p + "attn.to_out.bias")
        w1, b1 = st.get(p + "ff.w1.weight"), st.get(p + "ff.w1.bias")
        w2, b2 = st.get(p + "ff.w2.weight"), st.get(p + "ff.w2.bias")
        for nm, w, b in (("qkv", wqkv, bqkv), ("o", wo, bo), ("1", w1, b1), ("2", w2, b2)):
            if nm not in i8set:
                rec["w" + nm] = blob.put(f16(w, p + nm))
            else:
                q, s = quant_rows(had_rotate(w).astype(np.float32))
                rec["w" + nm] = blob.put(q)
                rec["q" + nm] = blob.put(s)  # per-row int8 weight scale (not s1 / s2 = scale1 / scale2)
            rec["b" + nm] = blob.put(b)
        rec["s1"] = blob.put(st.get(p + "scale1"))
        rec["s2"] = blob.put(st.get(p + "scale2"))
        rec["n1"] = blob.put(st.get(p + "norm1.weight"))
        rec["n2"] = blob.put(st.get(p + "norm2.weight"))
        man["layers"].append(rec)
        stats.append(float(max(np.abs(wqkv).max(), np.abs(wo).max(), np.abs(w1).max(), np.abs(w2).max())))
        print(f"layer {i} done {time.time() - t0:.0f}s  w absmax {stats[-1]:.3f}", flush=True)
    man["bytes"] = blob.off
    blob.f.close()
    man["layer_keys"] = list(man["layers"][0].keys())
    json.dump(man, open(out / f"manifest_{a.tag}.json", "w"), indent=1)
    print("CONVERT_OK", a.tag, blob.off / 1e9, "GB", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
