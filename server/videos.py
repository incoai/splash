"""Decode a video into the temporal patches of the Qwen3-VL video input.

The preprocessing follows the model's processor, transformers'
Qwen3VLVideoProcessor and Qwen3VLProcessor, configured by the repository's
video processor configuration (VideoConfig, read and checked at
installation), with the per-frame cap of Qwen's reference qwen-vl-utils
(``cap_pixels_per_frame``):

- frames are sampled uniformly at ``fps`` per second, at least
  ``min_frames`` and at most ``max_frames`` and never more than the video
  holds;
- an odd count repeats the last frame, so the frames pair into temporal
  patches of two;
- every frame is resized, bicubic, to one patch-aligned size that keeps the
  whole clip within ``size.longest_edge`` pixels (25,165,824 in the published
  config, about 12K tokens) and each frame within 768 tokens and the image
  patch limit the engine is started with;
- each temporal patch is introduced by the mean time of its two frames,
  rendered as ``<{t:.1f} seconds>`` before its vision tokens.

Each temporal patch then reaches the engine like an image whose two frames
differ: the vision tower attends within one temporal patch at a time and
M-RoPE places every patch as an image of one temporal step, exactly as the
model's ``get_rope_index`` splits a video grid.

The engine's ``video-frames`` command decodes, in a disposable worker that
workers.py supervises: AVFoundation reads MP4 and QuickTime files, VideoToolbox
decodes and converts to RGB with the stream's own matrix, and the frames are
resized with Pillow's bicubic filter, byte for byte, and turned upright.
"""

import base64
import hashlib
import json
import math
import os
import struct
import tempfile
import threading
import time
from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

from . import workers
from .errors import APIError
from .images import MAX_PIXELS as MAX_IMAGE_PIXELS
from .images import PATCH, PreparedImage, max_patches
from .lru import LRUCache

MERGE = 2
TEMPORAL_PATCH = 2
FACTOR = PATCH * MERGE
# transformers' Qwen3VLVideoProcessor class defaults, for the keys a
# configuration leaves out.
CLASS_FPS = 2.0
CLASS_MIN_FRAMES = 4
CLASS_MAX_FRAMES = 768
CLASS_MIN_PIXELS = 128 * FACTOR * FACTOR
CLASS_MAX_PIXELS = 768 * FACTOR * FACTOR
# The most frames a video may be sampled to: 384 temporal patches, of the
# 1024 spans a request carries.
MAX_FRAMES = 768
# One request frame carries at most 256 MiB of pixels and tokens; 80 Mi RGB
# pixels (about 40K tokens) leave room for the rest.
MAX_PIXELS = 80 * 1024 * 1024
MIN_PIXELS = FACTOR * FACTOR
# qwen-vl-utils VIDEO_MAX_TOKEN_NUM: the per-frame ceiling, in merged tokens.
MAX_TOKENS_PER_FRAME = 768
MAX_VIDEO_BYTES = 1024 * 1024 * 1024
VIDEO_URL_PREFIX = "data:video/"
VIDEO_PAD_TOKEN = "<|video_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
# The decoding worker's budgets, measured on an M1 Max: a 102 s 4K HEVC clip
# samples 204 frames in 8 s, with about 200 MB resident.
MAX_SECONDS = 120.0
MAX_MEMORY_BYTES = 1024 * 1024 * 1024
MAX_PROBE_BYTES = 64 * 1024 * 1024
WORK = workers.Work(
    subject="video decoding",
    worker="video worker",
    unavailable="video_unavailable",
    oversized="decoded video exceeds the size limit",
)
# The exit status of a video the decoder cannot read (sysexits EX_DATAERR).
UNREADABLE = 65
# The exit status of a decode whose hardware decoder returned no frame for
# seconds (EX_TEMPFAIL): with wired memory near the kernel's limit, as a
# model's weights wired by Metal can bring it, Apple's decoder stalls for
# good. The frames are decoded again in software.
STALLED = 75
# The exit status of a software decode of a codec that has no software decoder
# (EX_UNAVAILABLE): only the hardware decoder can read it.
NO_SOFTWARE_DECODER = 69


class VideoError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VideoConfig:
    """The sampling and size budget of the model's video processor, from its
    repository (install/upstream.py video_config)."""

    fps: float = 2.0
    min_frames: int = 4
    max_frames: int = 768
    min_pixels: int = 4096
    max_pixels: int = 25_165_824


# The configuration Qwen publishes for Qwen3.5, Qwen3.6, Qwen3.8 and
# Qwen3.8-Flash-Next (video_preprocessor_config.json and the class defaults it
# leaves to fps and the frame range), for the GGUF repositories, which carry
# none.
PUBLISHED_CONFIG = VideoConfig()


def _number(value, name, integer):
    if (
        isinstance(value, bool)
        or not isinstance(value, int if integer else (int, float))
        or not math.isfinite(value)
    ):
        raise VideoError(f"{name} must be a {'whole ' if integer else ''}number")
    return value


def video_config(processor):
    """The VideoConfig a video processor configuration describes, the keys it
    leaves out at Qwen3VLVideoProcessor's defaults, as transformers loads
    it; a VideoError for one Splash does not implement."""
    if not isinstance(processor, dict):
        raise VideoError("the video processor configuration is not an object")
    kind = processor.get("video_processor_type", "Qwen3VLVideoProcessor")
    if kind != "Qwen3VLVideoProcessor":
        raise VideoError(f"unsupported video processor {kind}")
    expected = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    if any(processor.get(key, value) != value for key, value in expected.items()):
        raise VideoError("unsupported video preprocessing configuration")
    for key in ("do_sample_frames", "do_resize", "do_convert_rgb"):
        if processor.get(key, True) is not True:
            raise VideoError(f"unsupported video preprocessing: {key} is not true")
    size = processor.get("size") or {}
    if not isinstance(size, dict):
        raise VideoError("the video processor's size is not an object")
    # Older configurations name the budget min_pixels and max_pixels, which
    # transformers reads as size.
    min_pixels = size.get(
        "shortest_edge", processor.get("min_pixels", CLASS_MIN_PIXELS)
    )
    max_pixels = size.get("longest_edge", processor.get("max_pixels", CLASS_MAX_PIXELS))
    config = VideoConfig(
        fps=float(_number(processor.get("fps", CLASS_FPS), "fps", False)),
        min_frames=_number(
            processor.get("min_frames", CLASS_MIN_FRAMES), "min_frames", True
        ),
        max_frames=_number(
            processor.get("max_frames", CLASS_MAX_FRAMES), "max_frames", True
        ),
        min_pixels=_number(min_pixels, "shortest_edge", True),
        max_pixels=_number(max_pixels, "longest_edge", True),
    )
    if not 0 < config.fps <= 60:
        raise VideoError("fps must be in (0, 60]")
    if not 1 <= config.min_frames <= config.max_frames <= MAX_FRAMES:
        raise VideoError(
            f"frame counts must satisfy 1 <= min_frames <= max_frames <= {MAX_FRAMES}"
        )
    if not MIN_PIXELS <= config.min_pixels <= config.max_pixels <= MAX_PIXELS:
        raise VideoError(
            f"the pixel budget must satisfy {MIN_PIXELS} <= shortest_edge <= "
            f"longest_edge <= {MAX_PIXELS}"
        )
    return config


def installed_config(vision):
    """The VideoConfig of the model whose installation's vision directory is
    vision: the repository's configuration the installation links
    (install/upstream.py _video_processor), else the published one."""
    for name, key in (
        ("video_preprocessor_config.json", None),
        ("processor_config.json", "video_processor"),
    ):
        path = Path(vision) / name
        if not path.is_file():
            continue
        try:
            config = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise VideoError(f"cannot read {path}: {error}") from None
        if key is not None:
            config = config.get(key) if isinstance(config, dict) else None
            if config is None:
                continue
        return video_config(config)
    return PUBLISHED_CONFIG


@dataclass(frozen=True, slots=True)
class VideoOptions:
    """One request's frame sampling and size budget, as ``mm_processor_kwargs``
    names them, and the per-frame pixel cap."""

    fps: float = PUBLISHED_CONFIG.fps
    num_frames: int | None = None
    min_frames: int = PUBLISHED_CONFIG.min_frames
    max_frames: int = PUBLISHED_CONFIG.max_frames
    min_pixels: int = PUBLISHED_CONFIG.min_pixels
    max_pixels: int = PUBLISHED_CONFIG.max_pixels
    # The engine's image size limit (--max-image-pixels): a temporal patch is
    # one image span, held to its patch limit.
    frame_pixels: int = MAX_IMAGE_PIXELS

    @classmethod
    def from_config(cls, config, max_pixels=None, frame_pixels=None):
        options = cls(
            fps=config.fps,
            min_frames=config.min_frames,
            max_frames=config.max_frames,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels if max_pixels is None else max_pixels,
        )
        if frame_pixels is not None:
            options = replace(options, frame_pixels=frame_pixels)
        return options

    def key(self) -> bytes:
        return struct.pack(
            "<dqqqqqq",
            self.fps,
            -1 if self.num_frames is None else self.num_frames,
            self.min_frames,
            self.max_frames,
            self.min_pixels,
            self.max_pixels,
            self.frame_pixels,
        )


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise VideoError(f"{name} must be a positive integer")
    return value


def video_options(kwargs, defaults=VideoOptions()) -> VideoOptions:
    """Options from a request's ``mm_processor_kwargs`` over the server's.

    Accepts the video processor's ``fps`` or ``num_frames`` (not both),
    ``min_frames``, ``max_frames``, ``do_sample_frames`` (only true: frames are
    always sampled) and ``size`` with ``shortest_edge`` and ``longest_edge``.
    """
    if kwargs is None:
        return defaults
    if not isinstance(kwargs, dict):
        raise VideoError("mm_processor_kwargs must be an object")
    unknown = set(kwargs) - {
        "fps",
        "num_frames",
        "min_frames",
        "max_frames",
        "do_sample_frames",
        "size",
    }
    if unknown:
        raise VideoError(
            "unsupported mm_processor_kwargs: " + ", ".join(sorted(unknown))
        )
    if kwargs.get("do_sample_frames", True) is not True:
        raise VideoError("do_sample_frames must be true")
    fps, num_frames = kwargs.get("fps"), kwargs.get("num_frames")
    if fps is not None and num_frames is not None:
        raise VideoError("fps and num_frames are mutually exclusive")
    changes = {}
    if fps is not None:
        if (
            not isinstance(fps, (int, float))
            or isinstance(fps, bool)
            or not math.isfinite(fps)
            or not 0 < fps <= 60
        ):
            raise VideoError("fps must be a number in (0, 60]")
        changes["fps"] = float(fps)
    if num_frames is not None:
        changes["num_frames"] = _positive_int(num_frames, "num_frames")
    for name in ("min_frames", "max_frames"):
        if name in kwargs:
            changes[name] = _positive_int(kwargs[name], name)
    size = kwargs.get("size")
    if size is not None:
        if not isinstance(size, dict) or set(size) - {"shortest_edge", "longest_edge"}:
            raise VideoError("size takes shortest_edge and longest_edge")
        if "shortest_edge" in size:
            changes["min_pixels"] = _positive_int(
                size["shortest_edge"], "shortest_edge"
            )
        if "longest_edge" in size:
            changes["max_pixels"] = _positive_int(size["longest_edge"], "longest_edge")
            if changes["max_pixels"] > MAX_PIXELS:
                raise VideoError(f"longest_edge must be at most {MAX_PIXELS}")
    options = replace(defaults, **changes)
    if options.max_frames > MAX_FRAMES or options.min_frames > options.max_frames:
        raise VideoError(
            f"frame counts must satisfy min_frames <= max_frames <= {MAX_FRAMES}"
        )
    if options.num_frames is not None and options.num_frames > options.max_frames:
        raise VideoError("num_frames exceeds max_frames")
    if options.min_pixels > options.max_pixels:
        raise VideoError("shortest_edge exceeds longest_edge")
    return options


def sample_indices(total_frames: int, video_fps: float, options: VideoOptions):
    """Qwen3VLVideoProcessor.sample_frames: uniform indices of displayed frames."""
    if total_frames < 1:
        raise VideoError("the video has no frames")
    if options.num_frames is not None:
        count = options.num_frames
    else:
        count = int(total_frames / video_fps * options.fps)
        count = min(max(count, options.min_frames), options.max_frames, total_frames)
    if count == 1:
        return [0]
    step = (total_frames - 1) / (count - 1)
    # numpy.linspace(...).round(): halves round to even.
    return [round(step * i) for i in range(count)]


def smart_resize(num_frames: int, height: int, width: int, options: VideoOptions):
    """Qwen3VLVideoProcessor.resize with ``cap_pixels_per_frame``: one patch
    aligned (height, width) for every frame of the clip, each frame also
    within the engine's image patch limit."""
    if height <= 0 or width <= 0:
        raise VideoError("video dimensions must be positive")
    span_pixels = max_patches(options.frame_pixels) * PATCH * PATCH
    frame_cap = min(MAX_TOKENS_PER_FRAME * FACTOR * FACTOR, span_pixels)
    pixels_per_frame = max(
        min(frame_cap, options.max_pixels // num_frames), int(options.min_pixels * 1.05)
    )
    max_pixels = pixels_per_frame * num_frames
    if height < FACTOR or width < FACTOR:
        scale = max(FACTOR / height, FACTOR / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise VideoError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    t_bar = round(num_frames / TEMPORAL_PATCH) * TEMPORAL_PATCH
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif t_bar * h_bar * w_bar < options.min_pixels:
        beta = math.sqrt(options.min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    # The engine refuses a span over its patch limit, which rounding up (or a
    # request's shortest_edge) can cross: the longer side gives way.
    while h_bar * w_bar > span_pixels and max(h_bar, w_bar) > FACTOR:
        if h_bar >= w_bar:
            h_bar -= FACTOR
        else:
            w_bar -= FACTOR
    return h_bar, w_bar


def patch_timestamps(seconds):
    """Qwen3VLProcessor._calculate_timestamps: the mean time of the two frames
    of each temporal patch, the last frame repeated for an odd count."""
    seconds = list(seconds)
    if len(seconds) % TEMPORAL_PATCH:
        seconds.append(seconds[-1])
    return [
        (seconds[i] + seconds[i + TEMPORAL_PATCH - 1]) / 2
        for i in range(0, len(seconds), TEMPORAL_PATCH)
    ]


def placeholder(timestamps) -> str:
    """The text Qwen3VLProcessor puts in place of one video's
    ``<|vision_start|><|video_pad|><|vision_end|>``, with a single pad token
    per temporal patch; the frontend widens each pad to the patch's tokens."""
    return "".join(
        f"<{t:.1f} seconds>{VISION_START_TOKEN}{VIDEO_PAD_TOKEN}{VISION_END_TOKEN}"
        for t in timestamps
    )


def read_ranges(frames, keyframes):
    """The ranges the decoder reads frames (increasing times) in, as index
    pairs: a range is decoded from the keyframe before its first frame, so
    one starts where a keyframe lies after a frame and at or before the next,
    and decoding from it is no longer than decoding on."""
    ranges, first = [], 0
    for i in range(1, len(frames)):
        k = bisect_right(keyframes, frames[i - 1])
        if k < len(keyframes) and keyframes[k] <= frames[i]:
            ranges.append([first, i - 1])
            first = i
    ranges.append([first, len(frames) - 1])
    return ranges


@dataclass(frozen=True, slots=True)
class PreparedVideo:
    patches: tuple[PreparedImage, ...]
    timestamps: tuple[float, ...]

    @property
    def tokens(self) -> int:
        return sum(patch.tokens for patch in self.patches)

    @property
    def pixel_bytes(self) -> int:
        return sum(len(patch.pixels) for patch in self.patches)


def decode_data_url(url) -> bytes:
    if not isinstance(url, str) or not url.startswith(VIDEO_URL_PREFIX):
        raise VideoError("only data:video/... base64 URLs are supported")
    header, _, encoded = url.partition(",")
    if ";base64" not in header:
        raise VideoError("video data URLs must be base64-encoded")
    if len(encoded) > MAX_VIDEO_BYTES * 4 // 3 + 4:
        raise VideoError("video exceeds the size limit")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise VideoError("video data URL is not valid base64") from error


def _decoded(finished):
    if finished.returncode == UNREADABLE:
        raise VideoError(finished.error or "the video cannot be read")
    if finished.returncode != 0:
        raise APIError(400, "video decoding failed or exceeded its resource limit")
    return finished.output


def _probe(binary, path, remaining):
    output = _decoded(
        workers.run(
            [binary, "video-frames", "probe", path],
            b"",
            WORK,
            remaining=remaining,
            max_seconds=MAX_SECONDS,
            max_memory_bytes=MAX_MEMORY_BYTES,
            max_output_bytes=MAX_PROBE_BYTES,
        )
    )
    try:
        probe = json.loads(output)
        width, height = int(probe["width"]), int(probe["height"])
        timescale = int(probe["timescale"])
        frames = [int(time) for time in probe["frames"]]
        keyframes = [int(time) for time in probe["keyframes"]]
    except (ValueError, KeyError, TypeError):
        raise APIError(
            500, "video worker returned an invalid result", "video_worker_failed"
        ) from None
    if not frames or timescale < 1 or width < 1 or height < 1:
        raise VideoError("the video has no frames")
    return width, height, timescale, frames, keyframes


def prepare(
    payload: bytes, options: VideoOptions, binary, remaining, software=False
) -> tuple[PreparedVideo, bool]:
    """Sample, decode, resize and pair the frames of one video, within the
    `remaining` seconds of its request, and tell whether the hardware decoder
    stalled. With software, the frames are decoded in software only."""
    with tempfile.TemporaryDirectory(prefix="splash-video-") as directory:
        # AVFoundation chooses the container's reader by the file's
        # extension; the QuickTime one reads MP4 too.
        path = os.path.join(directory, "input.mov")
        with open(path, "wb") as file:
            file.write(payload)
        width, height, timescale, frames, keyframes = _probe(binary, path, remaining())
        span = (frames[-1] - frames[0]) / timescale
        video_fps = (len(frames) - 1) / span if span > 0 else 1.0
        indices = sample_indices(len(frames), video_fps, options)
        # A clip of one frame still fills one temporal patch.
        count = max(len(indices), TEMPORAL_PATCH)
        frame_height, frame_width = smart_resize(count, height, width, options)
        targets = sorted({frames[i] for i in indices})
        frame_bytes = frame_width * frame_height * 3
        request = json.dumps(
            {
                "timescale": timescale,
                "frames": targets,
                "ranges": read_ranges(targets, keyframes),
                "keyframes": keyframes,
            }
        ).encode()
        command = [binary, "video-frames", "decode", path]
        command += [str(frame_width), str(frame_height)]
        stalled = False
        no_software_decoder = None
        for in_software in (True, False) if software else (False, True):
            finished = workers.run(
                command + (["--software"] if in_software else []),
                request,
                WORK,
                remaining=remaining(),
                max_seconds=MAX_SECONDS,
                max_memory_bytes=MAX_MEMORY_BYTES,
                max_output_bytes=len(targets) * frame_bytes,
            )
            if finished.returncode == STALLED:
                stalled = True
            elif finished.returncode == NO_SOFTWARE_DECODER:
                no_software_decoder = finished.error
            else:
                break
        if stalled and no_software_decoder:
            raise VideoError(
                f"{no_software_decoder}, and the hardware decoder has stalled: "
                "wired memory is near the kernel's limit"
            )
        output = _decoded(finished)
    if len(output) % frame_bytes:
        raise APIError(
            500, "video worker returned an invalid result", "video_worker_failed"
        )
    # A file that ends early yields the frames before its cut.
    decoded = {
        time: output[i * frame_bytes : (i + 1) * frame_bytes]
        for i, time in enumerate(targets[: len(output) // frame_bytes])
    }
    indices = [i for i in indices if frames[i] in decoded]
    if not indices:
        raise VideoError("no frame could be decoded")
    pixels = [decoded[frames[i]] for i in indices]
    seconds = [(frames[i] - frames[0]) / timescale for i in indices]
    if len(pixels) % TEMPORAL_PATCH:
        pixels.append(pixels[-1])
        seconds.append(seconds[-1])
    grid_height, grid_width = frame_height // PATCH, frame_width // PATCH
    patches = []
    for first in range(0, len(pixels), TEMPORAL_PATCH):
        patch = b"".join(pixels[first : first + TEMPORAL_PATCH])
        digest = hashlib.sha256(
            struct.pack("<III", grid_height, grid_width, TEMPORAL_PATCH) + patch
        )
        digest_lo, digest_hi = struct.unpack_from("<QQ", digest.digest())
        patches.append(
            PreparedImage(
                grid_height, grid_width, patch, digest_lo, digest_hi, TEMPORAL_PATCH
            )
        )
    prepared = PreparedVideo(tuple(patches), tuple(patch_timestamps(seconds)))
    return prepared, stalled


class VideoCache:
    """Byte-budgeted LRU of prepared videos keyed by the raw bytes and options:
    chat history resends a video with every turn. One video is decoded at a
    time; a request waits for the decoder, and decodes, without its
    preparation slot."""

    BUDGET_BYTES = 512 * 1024 * 1024
    # After the hardware decoder stalls, videos decode in software this long
    # first: what stalled it, the engine's wired memory, stays while it
    # serves.
    SOFTWARE_SECONDS = 600.0

    def __init__(self, binary=None):
        self.binary = binary
        self._lock = threading.Lock()
        self._decoding = threading.Lock()
        self._entries = LRUCache(self.BUDGET_BYTES)
        self._software_until = 0.0

    def _cached(self, key):
        with self._lock:
            return self._entries.get(key)

    def prepare(
        self, payload: bytes, options: VideoOptions, remaining, without_slot=nullcontext
    ) -> PreparedVideo:
        key = hashlib.sha256(options.key() + payload).digest()
        if (cached := self._cached(key)) is not None:
            return cached
        if self.binary is None:
            raise VideoError("video input needs the engine's video decoder")
        with without_slot():
            if not self._decoding.acquire(
                timeout=min(remaining(), threading.TIMEOUT_MAX)
            ):
                raise APIError(504, "request timed out", "request_timeout")
            try:
                # Another request may have decoded the same video meanwhile.
                if (cached := self._cached(key)) is not None:
                    return cached
                software = time.monotonic() < self._software_until
                prepared, stalled = prepare(
                    payload, options, self.binary, remaining, software
                )
                if stalled:
                    self._software_until = time.monotonic() + self.SOFTWARE_SECONDS
            finally:
                self._decoding.release()
        with self._lock:
            self._entries.put(key, prepared, prepared.pixel_bytes)
        return prepared

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._entries.bytes,
                "budget_bytes": self._entries.budget_bytes,
            }
