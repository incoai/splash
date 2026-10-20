"""TAEH3 preview decoder (madebyollin/taehv taeh3.safetensors, MIT; decoder only) -> the engine's weights
(<weights>/preview: taeh3_dec_f16.bin + manifest.json). The decoder structure follows FastVideo's TAEH3 port.

  convert_preview.py <taeh3.safetensors> weights/preview

Layouts (fp16 weights, fp32 biases; every tensor 256-byte aligned):
  conv3x3 (decoder.1, MemBlock convs, decoder.22): W[src][tap 9][Cout][Cin]   tap = ky*3 + kx
      MemBlock conv.0 has two sources: src 0 = x (in-channels [0, C)), src 1 = past (in-channels [C, 2C)).
      decoder.1: Cin 24 -> 32 (zero pad). decoder.22: Cout 12 -> 16 (zero pad).
  tgrow 1x1 (decoder.7/13/19, no bias): W[Cout][Cin]
  up (decoder.8/14/20, no bias; nearest 2x upsample then 3x3 conv) as 4 output phases x 2x2 taps on the low-res
      input: W[phase = py*2 + px][tap = a*2 + b][Cout][Cin], tap (a, b) reads low-res offset (py + a - 1, px + b - 1);
      rows: py = 0: a=0 <- k0, a=1 <- k1 + k2; py = 1: a=0 <- k0 + k1, a=1 <- k2 (same for columns). Sums in fp32.
Pure numpy safetensors parsing (no code from the download is executed).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_st(path):
    b = Path(path).read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    hdr = json.loads(b[8:8 + n])
    hdr.pop("__metadata__", None)
    base = 8 + n
    dt = {"F16": np.float16, "F32": np.float32, "BF16": None}
    out = {}
    for k, v in hdr.items():
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(b[base + s:base + e], dtype=dt[v["dtype"]]).reshape(v["shape"]).astype(np.float32)
    return out


def main(src, dst_dir):
    w = load_st(src)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    man = {}

    def put(name, arr):
        while len(blob) % 256:
            blob.append(0)
        man[name] = {"off": len(blob), "shape": list(arr.shape), "dtype": str(arr.dtype)}
        blob.extend(np.ascontiguousarray(arr).tobytes())

    def conv3(name, key, nsrc=1, cin_pad=None, cout_pad=None):
        k = w[key + ".weight"]  # (Cout, Cin_total, 3, 3)
        co, ci_t = k.shape[:2]
        ci = ci_t // nsrc
        k = k.reshape(co, nsrc, ci, 3, 3).transpose(1, 3, 4, 0, 2).reshape(nsrc, 9, co, ci)  # [src][tap][co][ci]
        if cin_pad and cin_pad > ci:
            k = np.concatenate([k, np.zeros(k.shape[:3] + (cin_pad - ci,), np.float32)], axis=3)
        if cout_pad and cout_pad > co:
            k = np.concatenate([k, np.zeros((nsrc, 9, cout_pad - co, k.shape[3]), np.float32)], axis=2)
        put(name + ".w", k.astype(np.float16))
        b = w.get(key + ".bias")
        if b is not None:
            if cout_pad and cout_pad > co:
                b = np.concatenate([b, np.zeros(cout_pad - co, np.float32)])
            put(name + ".b", b.astype(np.float32))

    def up(name, key):
        k = w[key + ".weight"]  # (Cout, Cin, 3, 3)
        comb = {0: [[0], [1, 2]], 1: [[0, 1], [2]]}
        ph = np.zeros((4, 4) + k.shape[:2], np.float32)
        for py in range(2):
            for px in range(2):
                for a in range(2):
                    for bb in range(2):
                        acc = sum(k[:, :, ky, kx] for ky in comb[py][a] for kx in comb[px][bb])
                        ph[py * 2 + px, a * 2 + bb] = acc
        put(name + ".w", ph.astype(np.float16))

    conv3("d1", "decoder.1", cin_pad=32)
    for idx in (3, 4, 5, 9, 10, 11, 15, 16, 17):
        conv3(f"m{idx}.0", f"decoder.{idx}.conv.0", nsrc=2)
        conv3(f"m{idx}.2", f"decoder.{idx}.conv.2")
        conv3(f"m{idx}.4", f"decoder.{idx}.conv.4")
    for idx in (7, 13, 19):
        k = w[f"decoder.{idx}.conv.weight"][:, :, 0, 0]
        put(f"g{idx}.w", k.astype(np.float16))
    for idx in (8, 14, 20):
        up(f"u{idx}", f"decoder.{idx}")
    conv3("d22", "decoder.22", cout_pad=16)
    (dst / "taeh3_dec_f16.bin").write_bytes(bytes(blob))
    (dst / "manifest.json").write_text(json.dumps(man, indent=1))
    n_dec = sum(v.size for k, v in w.items() if k.startswith("decoder."))
    print("CONVERT_OK", len(man), "tensors", len(blob), "bytes; decoder params", n_dec)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
