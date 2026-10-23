#!/usr/bin/env python3
"""Request-independent inputs of the native prep (h3 prep, runtime/model/Prep) -> <weights>/dit-<variant>/prep.safetensors,
plus the link <weights>/fl2va -> the MiniMax-H3 FL2VA snapshot (the adaLN bases of the 50 blocks, 26 GB bf16, are read
from its transformer shards at prep time).

  convert_prep.py --variant turbo|default [--weights DIR]

prep.safetensors (numpy only, no code from the downloads is executed):
  time_embedder.proj_in.{weight,bias}, time_embedder.proj_out.{weight,bias}      F32, as in the FL2VA shards
  condition_proj.{weight,bias}, final_layer.adaln_proj.linear.{weight,bias}       BF16, as in the FL2VA shards
  token_refiner.*                                                                  BF16, the checkpoint's refiner LoRAs
      merged the OpenVDN way (merge_lora_state: W.add_(bf16(B @ A * scale)) in bf16, adapters in order: default,
      then turbo for the turbo variant): W = bf16(W + bf16(scale (B @ A))), fp32 numpy, delta by delta. q / k / v
      deltas land on the per-head interleaved qkv_proj rows; ff.net.0.proj [value; gate] -> fc1 [gate; value].
The rank-16 adaLN LoRA factors stay where convert_dit.py put them (vdn-w8-blockNN.safetensors, vdn-globals).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3.common import bf16_bits, fl2va, from_bf16_bits  # noqa: E402
from h3.config import WEIGHTS  # noqa: E402
from h3.safetensors import STFile, lora_deltas, save_st  # noqa: E402

H, HD = 56, 128


def adapters_for(vdn: Path, variant: str):
    if variant == "turbo":
        st = vdn / "stage-dmd-step-250"
        return [st / "adapters" / "default", st / "adapters" / "turbo"]
    return [vdn / "stage-b-step-2000" / "adapters" / "default"]


def refiner_lora(vdn: Path, variant: str):
    """native token_refiner weight name -> fp32 deltas [out, in] (adapter order)."""
    out = {}
    for ad in adapters_for(vdn, variant):
        for name, (a, b, sc) in lora_deltas(str(ad)).items():
            if not name.startswith("token_refiner.refiner_blocks."):
                continue
            blk = int(name.split(".")[2])
            rest = ".".join(name.split(".")[3:])
            d = (np.float32(sc) * (b.astype(np.float32) @ a.astype(np.float32))).astype(np.float32)
            if rest in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"):
                p = "qkv".index(rest[8])
                key = f"token_refiner.blocks.{blk}.attn.qkv_proj.weight"
                full = np.zeros((H * 3 * HD, d.shape[1]), np.float32)
                rows = (np.arange(H)[:, None] * 3 * HD + p * HD + np.arange(HD)[None, :]).reshape(-1)
                full[rows] = d
                out.setdefault(key, []).append(full)
            elif rest == "attn.to_out.0.weight":
                out.setdefault(f"token_refiner.blocks.{blk}.attn.out_proj.weight", []).append(d)
            elif rest == "ff.net.0.proj.weight":  # diffusers [value; gate] -> native fc1 [gate; value]
                n = d.shape[0] // 2
                out.setdefault(f"token_refiner.blocks.{blk}.mlp.fc1.weight", []).append(
                    np.concatenate([d[n:], d[:n]], axis=0))
            elif rest == "ff.net.2.weight":
                out.setdefault(f"token_refiner.blocks.{blk}.mlp.fc2.weight", []).append(d)
            else:
                raise KeyError(name)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["turbo", "default"], required=True)
    ap.add_argument("--weights", default=str(WEIGHTS))
    a = ap.parse_args()
    weights = Path(a.weights)
    src = fl2va()
    link = weights / "fl2va"
    if not link.exists():
        os.symlink(src, link)
    tdir = src / "transformer"
    index = json.load(open(tdir / "model.safetensors.index.json"))["weight_map"]
    files = {}

    def get(k):
        f = index[k]
        if f not in files:
            files[f] = STFile(tdir / f)
        dt, arr = files[f].get(k)
        return dt, np.array(arr)

    keys = [k for k in index if k.startswith(("time_embedder.", "condition_proj.", "token_refiner.",
                                              "final_layer.adaln_proj."))]
    tensors = {k: get(k) for k in keys}
    for k, (dt, arr) in tensors.items():
        want = "F32" if k.startswith("time_embedder.") else "BF16"
        if dt != want:
            raise ValueError(f"{k}: {dt}, expected {want}")
    merged = 0
    for k, ds in refiner_lora(weights / "vdn", a.variant).items():
        wb = from_bf16_bits(tensors[k][1])
        for d in ds:
            wb = from_bf16_bits(bf16_bits(wb + from_bf16_bits(bf16_bits(d))))
            merged += 1
        tensors[k] = ("BF16", bf16_bits(wb))
    out = weights / f"dit-{a.variant}"
    out.mkdir(parents=True, exist_ok=True)
    save_st(out / "prep.safetensors", tensors, meta={"variant": a.variant, "refiner_lora_deltas": str(merged)})
    print(f"CONVERT_OK {out / 'prep.safetensors'}: {len(tensors)} tensors, {merged} refiner LoRA deltas merged; "
          f"{link} -> {os.readlink(link)}")


if __name__ == "__main__":
    main()
