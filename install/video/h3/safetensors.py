"""Minimal safetensors IO (numpy only) and PEFT LoRA adapters, shared by prep and the converters in tools/."""
from __future__ import annotations

import json
import os
import re
import struct

import numpy as np

_DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I8": np.int8, "U8": np.uint8, "I32": np.int32}


class STFile:
    """Header index of one safetensors file; get() memory-maps a tensor."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = json.loads(f.read(n))
        self.meta = self.hdr.pop("__metadata__", None)
        self.base = 8 + n

    def keys(self):
        return list(self.hdr)

    def get(self, k):
        e = self.hdr[k]
        a, _ = e["data_offsets"]
        arr = np.memmap(self.path, dtype=_DT[e["dtype"]], mode="r", offset=self.base + a, shape=tuple(e["shape"]))
        return e["dtype"], arr


def to_f32(dtype, arr):
    if dtype == "BF16":
        return (np.asarray(arr, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.asarray(arr, dtype=np.float32)


def f32_to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def save_st(path, tensors, meta=None):
    """tensors: name -> (dtype string, ndarray); keys sorted, header padded to 8 bytes."""
    hdr, off, blobs = {}, 0, []
    for k in sorted(tensors):
        dt, a = tensors[k]
        b = np.ascontiguousarray(a).tobytes()
        hdr[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    if meta:
        hdr["__metadata__"] = meta
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


class Source:
    """name -> fp32 array over a set of safetensors files."""

    def __init__(self, files=(), prefix=""):
        self.loc = {}
        for p in files:
            st = STFile(p)
            for k in st.keys():
                self.loc[prefix + k] = (st, k)

    def names(self):
        return set(self.loc)

    def has(self, n):
        return n in self.loc

    def get(self, n):
        st, k = self.loc[n]
        return to_f32(*st.get(k))


def lora_deltas(adapter_dir):
    """PEFT adapter (adapter_model.safetensors + adapter_spec.json) -> {base weight name: (A [r, in], B [out, r],
    scale)}. Module 'X.attn.orig.to_q' targets base 'X.attn.to_q.weight'."""
    spec = json.load(open(os.path.join(adapter_dir, "adapter_spec.json")))
    src = Source([os.path.join(adapter_dir, "adapter_model.safetensors")])
    cfg = spec["config"]
    alpha = cfg["alpha"]
    apat = cfg.get("alpha_pattern", {}) or {}
    out = {}
    for n in src.names():
        m = re.match(r"^(.*)\.lora_A\.([^.]+)\.weight$", n)
        if not m:
            continue
        mod, tag = m.group(1), m.group(2)
        a = src.get(n)
        b = src.get("%s.lora_B.%s.weight" % (mod, tag))
        r = a.shape[0]
        al = alpha
        for k, v in apat.items():  # PEFT semantics: key matches the module name or its suffix
            pat = re.escape(k).replace("\\*", "[^.]*")
            if re.fullmatch(pat, mod) or re.search(r"(^|\.)" + pat + "$", mod):
                al = v
        base = mod.replace(".attn.orig.", ".attn.") + ".weight"
        out[base] = (a, b, al / r)
    return out
