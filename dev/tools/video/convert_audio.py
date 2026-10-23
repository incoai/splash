"""MiniMax-H3 audio VAE decoder (FL2VA/audio_vae/model.safetensors: dec_in_proj + BigVGAN decoder) -> the engine's
weights (<weights>/audio: audio_dec_f16.bin + manifest.json). The encoder half is not converted.

  convert_audio.py <FL2VA>/audio_vae weights/audio

Weight norm is folded (w = g v / ||v||, norm over every axis but the first, fp64). Every conv weight is stored as an
fp16 hi / lo pair (.w = fp16(w), .wl = fp16(w - .w)): the engine accumulates hi*hi + lo*hi + hi*lo in fp32 against
hi / lo activations. Measured on the 3 s golden latents against the MLX fp32 decoder: fp16 weights and fp16
activations 41.9 dB, fp16 weights with exact activations 60.7 dB, both split 101 dB (the port's default TF32 path:
42.9 dB). Layouts (every tensor 256-byte aligned; fp32 biases and activation parameters):
  conv (dec_in_proj, conv_pre, AMP convs1 / convs2, conv_post)   W[tap][Cout][Cin]   (torch (Cout, Cin, k))
  transposed conv (ups.i)                                         W[tap][Cout][Cin]   (torch (Cin, Cout, k))
  SnakeBeta of every Activation1d                                  [2][C]: exp(alpha), 1 / (exp(beta) + 1e-9)
  filter                                                           [12] Kaiser-windowed sinc of the 2x up/down
                                                                   samplers (the port's kaiser_sinc_filter1d)
  latents_mean / latents_std                                       [32] de-normalisation (config.json)
Manifest keys: dec_in, conv_pre, ups<i>, rb<r>.c1.<d>, rb<r>.c2.<d> (r = stage * 3 + kernel index, d = dilation
index) with .w / .wl / .b, rb<r>.act<a> (a = 0..5), post.act, conv_post.w, filter, latents_mean, latents_std; "config"
holds the decoder shape. Pure numpy safetensors parsing (no code from the download is executed).
"""
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

RATES = (5, 5, 2, 2, 2, 2, 2)
KERNELS = (9, 9, 4, 4, 4, 4, 4)
RES_KERNELS = (3, 7, 11)
DILATIONS = (1, 3, 5)
DECODER_DIM = 1024


def load_st(path):
    b = Path(path).read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    hdr = json.loads(b[8:8 + n])
    hdr.pop("__metadata__", None)
    base = 8 + n
    out = {}
    for k, v in hdr.items():
        if v["dtype"] != "F32":
            raise ValueError(f"{k}: unexpected dtype {v['dtype']}")
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(b[base + s:base + e], dtype=np.float32).reshape(v["shape"])
    return out


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """minimax_h3_mlx.audio_vae.kaiser_sinc_filter1d (alias-free-torch), float32 [kernel_size]."""
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
    filt = filt / filt.sum()
    return filt.astype(np.float32)


def main(src, dst_dir):
    src = Path(src)
    w = load_st(src / "model.safetensors")
    wrapper = json.loads((src / "config.json").read_text())
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    man = {}

    def put(name, arr):
        while len(blob) % 256:
            blob.append(0)
        man[name] = {"off": len(blob), "shape": list(arr.shape), "dtype": str(arr.dtype)}
        blob.extend(np.ascontiguousarray(arr).tobytes())

    def folded(key):
        if key + ".weight" in w:
            return w[key + ".weight"].astype(np.float64)
        g = w[key + ".weight_g"].astype(np.float64)
        v = w[key + ".weight_v"].astype(np.float64)
        norm = np.sqrt((v.reshape(v.shape[0], -1) ** 2).sum(axis=1)).reshape(-1, 1, 1)
        return g * v / norm

    def conv(name, key, transposed=False, bias=True):
        k = folded(key)
        k = k.transpose(2, 1, 0) if transposed else k.transpose(2, 0, 1)  # -> [tap][Cout][Cin]
        hi = k.astype(np.float16)
        put(name + ".w", hi)
        put(name + ".wl", (k - hi.astype(np.float64)).astype(np.float16))
        if bias:
            put(name + ".b", w[key + ".bias"].astype(np.float32))
        return k.shape

    def snake(name, key):
        a = np.exp(w[key + ".act.alpha"].astype(np.float32))
        b = np.float32(1.0) / (np.exp(w[key + ".act.beta"].astype(np.float32)) + np.float32(1e-9))
        put(name, np.stack([a, b]).astype(np.float32))

    filt = kaiser_sinc_filter1d(0.5 / 2, 0.6 / 2, 12)
    stored = [v.reshape(-1) for k, v in w.items() if k.startswith("decoder.") and k.endswith(".filter")]
    worst = max(float(np.abs(f - filt).max()) for f in stored)
    put("filter", filt)
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
    man["config"] = {"decoder_dim": DECODER_DIM, "rates": list(RATES), "kernels": list(KERNELS),
                     "res_kernels": list(RES_KERNELS), "dilations": list(DILATIONS), "latent_channels": 32,
                     "latent_dim": 2048, "sample_rate": int(wrapper.get("sample_rate", 32000)), "hop": 800}
    (dst / "audio_dec_f16.bin").write_bytes(bytes(blob))
    (dst / "manifest.json").write_text(json.dumps(man, indent=1))
    n_dec = sum(v.size for k, v in w.items() if k.startswith(("decoder.", "dec_in_proj.")) and not k.endswith("filter"))
    print(f"CONVERT_OK {len(man) - 1} tensors, {len(blob)} bytes; decoder params {n_dec}; "
          f"filter vs checkpoint buffers max |diff| {worst:.2e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
