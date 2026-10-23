"""Helpers shared by the phases and tools: Hugging Face snapshot paths, bf16 bits, phase timer."""
from __future__ import annotations

import glob
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from h3.config import H3_REPO


def hf_hub() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


def snap(repo: str) -> Path:
    d = sorted(glob.glob(str(hf_hub() / ("models--" + repo.replace("/", "--")) / "snapshots" / "*")))
    if not d:
        raise FileNotFoundError(f"{repo} is not in the Hugging Face cache (tools/fetch_weights.py)")
    return Path(d[-1])


def fl2va() -> Path:
    return snap(H3_REPO) / "FL2VA"


def bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16_bits(b: np.ndarray) -> np.ndarray:
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class Phase:
    """Wall time + peak memory of the process (ru_maxrss, and MLX peak if MLX was used)."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        print(f"PHASE {self.name} wall_s={self.dt:.1f} maxrss_GB={maxrss_gb():.2f} mlx_peak_GB={mlx_peak_gb():.2f}",
              flush=True)


def maxrss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def mlx_peak_gb() -> float:
    if "mlx.core" not in sys.modules:
        return 0.0
    import mlx.core as mx
    return mx.get_peak_memory() / 1e9


def save_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
