"""User-facing request options and the paths of the frontend.

The engine (build/libh3.dylib) takes the request as a struct (h3/engine.py); every execution choice (kernels, tiles,
dispatch shapes) is fixed inside it (runtime/engine/DeviceProfile.hpp). Environment variables name paths only:
  H3_WEIGHTS_DIR  weights root (default <repo>/weights): dit-turbo/, dit-default/ (converted DiT + stream.bin),
                  vae/ (converted video VAE), audio/ (converted audio VAE), preview/ (converted TAEH3),
                  fl2va -> the FL2VA snapshot (adaLN bases read by h3 prep), vdn/ (OpenVDN, converters only)
  H3_MLX_PORT     MiniMax-H3 MLX port (request presentation of the standalone text encoder, h3 encode, only)
  H3_GPU_LOCK     shared GPU lock directory
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = Path.home()
BUILD = REPO / "build"
WEIGHTS = Path(os.environ.get("H3_WEIGHTS_DIR") or REPO / "weights").expanduser()
MLX_PORT = Path(os.environ.get("H3_MLX_PORT") or HOME / "exp" / "h3-pipeline" / "ref" / "minimax-h3-mlx").expanduser()
GPU_LOCK = Path(os.environ.get("H3_GPU_LOCK") or HOME / "exp" / "h3-gpu.lock").expanduser()

TEXT_ENCODER_REPO = "mlx-community/Qwen3-VL-32B-Instruct-4bit"  # == H3 FL2VA/text_encoder, MLX 4-bit
H3_REPO = "MiniMaxAI/MiniMax-H3"
CLIP_FRAMES = {3: 73, 10: 243, 15: 362}  # seconds -> video frames (17 n + 5), 24 fps
GOLDEN_PROMPT = ("A golden retriever runs along a beach at sunset, waves crashing, camera tracking from the side. "
                 "Sound of waves and the dog's footsteps on wet sand.")


@dataclass
class Request:
    variant: str = "turbo"          # turbo (8 steps) | default (50 steps)
    seconds: int = 3                # 3 | 10 | 15 (frames 73 / 243 / 362), unless frames is set
    frames: int | None = None       # 17 n + 5
    width: int = 1344
    height: int = 768
    window_chunk: int = 5           # window softmax: frame chunk and radius; anchor frames first and last
    window_radius: int = 1
    qkfix: bool = True              # fp16 Q / K in the QKFIX blocks
    qkfix_blocks: tuple = (45, 48, 49)
    stream: bool = False            # SSD weight streaming (15 s: ~15 GB device memory instead of ~32 GB)
    stream_slots: int = 8
    stream_image: str | None = None  # default <weights>/dit-<variant>/stream.bin
    vae_precision: str = "w8"       # w8 | f16
    preview: bool = False           # also run the TAEH3 preview decoder (generate)
    seed: int = 0

    @property
    def num_frames(self) -> int:
        return self.frames or CLIP_FRAMES[self.seconds]

    def dit_dir(self) -> Path:
        return WEIGHTS / f"dit-{self.variant}"


def add_request_args(ap, *, dit=False, vae=False, length=False):
    """argparse options of the request fields a phase uses."""
    d = Request()
    if length:
        g = ap.add_mutually_exclusive_group()
        g.add_argument("--seconds", type=int, choices=sorted(CLIP_FRAMES), default=d.seconds)
        g.add_argument("--frames", type=int, help="17 n + 5")
    ap.add_argument("--variant", choices=["turbo", "default"], default=d.variant)
    if dit:
        ap.add_argument("--chunk", type=int, default=d.window_chunk, help="window frame chunk")
        ap.add_argument("--radius", type=int, default=d.window_radius, help="window radius (chunks)")
        ap.add_argument("--no-qkfix", action="store_true")
        ap.add_argument("--qkfix-blocks", default=",".join(map(str, d.qkfix_blocks)))
        ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=d.stream, help="SSD weight streaming")
        ap.add_argument("--stream-slots", type=int, default=d.stream_slots)
        ap.add_argument("--stream-image", default=None)
    if vae:
        ap.add_argument("--vae-precision", choices=["w8", "f16"], default=d.vae_precision)


def request_from(args, frames: int | None = None) -> Request:
    r = Request()
    r.variant = getattr(args, "variant", r.variant)
    if frames:
        r.frames = frames
    elif getattr(args, "frames", None):
        r.frames = args.frames
    elif getattr(args, "seconds", None):
        r.seconds = args.seconds
    if hasattr(args, "chunk"):
        r.window_chunk, r.window_radius = args.chunk, args.radius
        r.qkfix = not args.no_qkfix
        r.qkfix_blocks = tuple(int(b) for b in args.qkfix_blocks.split(",") if b)
        r.stream, r.stream_slots, r.stream_image = args.stream, args.stream_slots, args.stream_image
    if hasattr(args, "vae_precision"):
        r.vae_precision = args.vae_precision
    return r
