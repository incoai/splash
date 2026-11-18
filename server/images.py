"""Decode and resize images to patch-aligned uint8 RGB within the pixel budget.

The engine handles normalization and vision inference. Grid geometry and a
content digest identify image prefixes; a byte-budgeted cache reuses images
resent in chat history.
"""

import base64
import hashlib
import io
import json
import math
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

from .lru import LRUCache

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE
# Qwen3-VL's shortest_edge floor; the serving cap below is far below the model's.
MIN_PIXELS = 65_536
# Serving default and largest cap: 16,384 patches, the protocol's per-image ceiling.
MAX_PIXELS = 4_194_304
# Bound codec work before full-resolution decode, independently of the model's
# resized input budget. Ordinary screenshots and up to 32 MP photos fit here.
MAX_SOURCE_PIXELS = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP", "GIF")
# Nemotron Omni's processor (Nemotron 3.5 Bolt): each image's patches
# within these bounds.
NEMOTRON_MIN_PATCHES = 1024
NEMOTRON_MAX_PATCHES = 13312
# Its video frames: about this many patches each, two frames to a tubelet.
VIDEO_TARGET_PATCHES = 1024
VIDEO_TUBELET = 2
# Sampling as vLLM's Nemotron loader does it, at the model card's 1080p limits.
VIDEO_FPS = 1.0
VIDEO_MAX_FRAMES = 128
# EVS: the share of a video's tubelet rows dropped as least changed from the
# tubelet before (config.json video_pruning_rate, which HF applies).
VIDEO_PRUNING = 0.7
MAX_VIDEO_BYTES = 256 * 1024 * 1024


class ImageError(ValueError):
    pass


class ImageCapacityError(ImageError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedImage:
    grid_height: int
    grid_width: int
    pixels: bytes
    digest_lo: int
    digest_hi: int
    # A video's tubelet holds two frames, one after the other, in one grid.
    frames: int = 1

    @property
    def tokens(self) -> int:
        """Merged language tokens the image occupies."""
        return (self.grid_height // MERGE) * (self.grid_width // MERGE)


def decode_data_url(url, kind="image", limit=MAX_IMAGE_BYTES) -> bytes:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ImageError(f"only data: {kind} URLs are supported")
    header, _, encoded = url.partition(",")
    if ";base64" not in header:
        raise ImageError(f"{kind} data URLs must be base64-encoded")
    if len(encoded) > limit * 4 // 3 + 4:
        raise ImageError(f"{kind} exceeds the size limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ImageError(f"{kind} data URL is not valid base64") from error
    if len(raw) > limit:
        raise ImageError(f"{kind} exceeds the size limit")
    return raw


def max_patches(max_pixels: int) -> int:
    """The most patches an image resized within max_pixels can have: its
    sides are whole merge units, so its patch count is a multiple of 4."""
    return max_pixels // (PATCH * PATCH) // 4 * 4


def smart_resize(height: int, width: int, max_pixels: int) -> tuple[int, int]:
    """Upstream Qwen2VL smart-resize: patch/merge-aligned sides inside the
    pixel budget with the aspect ratio preserved as closely as possible."""
    if height <= 0 or width <= 0:
        raise ImageError("image dimensions must be positive")
    if not MIN_PIXELS <= max_pixels <= MAX_PIXELS:
        raise ImageError("invalid image pixel budget")
    if max(height, width) / min(height, width) > 200:
        raise ImageError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    # Upscaling to the minimum can also overshoot a small serving cap after
    # alignment. The hard cap takes precedence over that soft minimum.
    while h_bar * w_bar > max_pixels:
        if h_bar >= w_bar and h_bar > FACTOR:
            h_bar -= FACTOR
        else:
            w_bar -= FACTOR
    return h_bar, w_bar


def nemotron_resize(height: int, width: int, max_pixels: int) -> tuple[int, int]:
    """Nemotron Omni's _compute_target_patches: about one patch per 16 pixels,
    scaled down into the patch budget or up to its minimum, then each side
    rounded to whole merge units. The serving cap bounds the budget and, as
    in smart_resize, takes precedence over the minimum."""
    if height <= 0 or width <= 0:
        raise ImageError("image dimensions must be positive")
    if not MIN_PIXELS <= max_pixels <= MAX_PIXELS:
        raise ImageError("invalid image pixel budget")
    if max(height, width) / min(height, width) > 200:
        raise ImageError("absolute aspect ratio must be smaller than 200")
    budget = min(max_patches(max_pixels), NEMOTRON_MAX_PATCHES)
    # NVIDIA's processor's own rounding, ties to even: a whole odd count of patches
    # gains one.
    rows, cols = round(height / PATCH + 0.5), round(width / PATCH + 0.5)
    factor = min(math.sqrt(budget / (rows * cols)), 1.0)
    rows, cols = math.floor(factor * rows), math.floor(factor * cols)
    if budget > NEMOTRON_MIN_PATCHES and rows * cols < NEMOTRON_MIN_PATCHES:
        up = math.sqrt(NEMOTRON_MIN_PATCHES / (rows * cols))
        rows, cols = math.ceil(up * rows), math.ceil(up * cols)
    if rows % MERGE:
        if (rows + MERGE - rows % MERGE) * cols <= budget:
            rows += MERGE - rows % MERGE
        else:
            rows = max(MERGE, rows - rows % MERGE)
    if cols % MERGE:
        if rows * (cols + MERGE - cols % MERGE) <= budget:
            cols += MERGE - cols % MERGE
        else:
            cols = max(MERGE, cols - cols % MERGE)
    # The minimum's ceilings can overshoot a budget just above it.
    while rows * cols > budget:
        if rows >= cols:
            rows -= MERGE
        else:
            cols -= MERGE
    return rows * PATCH, cols * PATCH


def video_resize(height: int, width: int) -> tuple[int, int]:
    """Nemotron Omni's _compute_target_patches_video: VIDEO_TARGET_PATCHES
    patches at the frame's aspect ratio, each side rounded up to whole merge
    units where that stays within them, else both down."""
    aspect = width / max(height, 1)
    rows = max(round(math.sqrt(VIDEO_TARGET_PATCHES / aspect)), 1)
    cols = max(round(math.sqrt(VIDEO_TARGET_PATCHES * aspect)), 1)
    up_rows, up_cols = rows + rows % MERGE, cols + cols % MERGE
    if up_rows * up_cols <= VIDEO_TARGET_PATCHES:
        rows, cols = up_rows, up_cols
    else:
        rows, cols = max(MERGE, rows - rows % MERGE), max(MERGE, cols - cols % MERGE)
    return rows * PATCH, cols * PATCH


def resize_rgb(image, width: int, height: int):
    from PIL import Image

    return image.resize((width, height), Image.Resampling.BICUBIC)


def resize_float(image, width: int, height: int):
    """Antialiased bicubic on float pixels, rounded once: within 0.03 levels
    of torch's interpolate(antialias=True), which Nemotron Omni's processor
    resizes with. Pillow's RGB path rounds between its passes, up to 4 levels
    off it."""
    from PIL import Image

    return Image.merge(
        "RGB",
        [
            channel.convert("F")
            .resize((width, height), Image.Resampling.BICUBIC)
            .point(lambda value: value + 0.5)
            .convert("L")  # Truncates and clamps: rounds half up.
            for channel in image.split()
        ],
    )


# Each family's resize rule (ChatFormat.image_resize): its target size, and
# how it resizes to it.
RESIZE_RULES = {
    "qwen": (smart_resize, resize_rgb),
    "nemotron": (nemotron_resize, resize_float),
}


def prepare(
    payload: bytes, max_pixels: int = MAX_PIXELS, rule: str = "qwen"
) -> PreparedImage:
    """Decode, convert to RGB, resize into the grid, and digest the pixels."""
    from PIL import Image, ImageOps

    target, resize = RESIZE_RULES[rule]
    try:
        with Image.open(io.BytesIO(payload), formats=IMAGE_FORMATS) as decoded:
            if decoded.height * decoded.width > MAX_SOURCE_PIXELS:
                raise ImageError("source image exceeds the pixel limit")
            stored = decoded.size
            height, width = target(decoded.height, decoded.width, max_pixels)
            try:
                # Cameras store the sensor's orientation and an EXIF tag that
                # turns it upright for display; show the model the upright one.
                ImageOps.exif_transpose(decoded, in_place=True)
            except Exception:
                pass  # A malformed tag leaves the stored orientation.
            if decoded.size != stored:
                height, width = width, height
            if decoded.has_transparency_data:
                # Composite onto white as Qwen's preprocessing does: transparent
                # pixels usually store black, which hides dark content.
                rgba = decoded.convert("RGBA")
                image = Image.new("RGB", rgba.size, (255, 255, 255))
                image.paste(rgba, mask=rgba)
            else:
                image = decoded.convert("RGB")
            if (image.height, image.width) != (height, width):
                image = resize(image, width, height)
            pixels = image.tobytes()
    except ImageError:
        raise
    except Exception as error:
        raise ImageError("image could not be decoded") from error
    grid_height, grid_width = height // PATCH, width // PATCH
    digest = hashlib.sha256(struct.pack("<II", grid_height, grid_width) + pixels)
    digest_lo, digest_hi = struct.unpack_from("<QQ", digest.digest())
    return PreparedImage(grid_height, grid_width, pixels, digest_lo, digest_hi)


@dataclass(frozen=True, slots=True)
class PreparedVideo:
    """A video's tubelets, each one image of two frames, and the label the
    processor writes before each."""

    tubelets: tuple[PreparedImage, ...]
    labels: tuple[str, ...]


def _frame_labels(indices, source_fps):
    """Nemotron Omni's processor's: each tubelet's frames by number and
    time, the time truncated to whole milliseconds of a frame first."""
    frame_ms = int(1000.0 / source_fps)
    labels = []
    for start in range(0, len(indices), VIDEO_TUBELET):
        parts = [
            f"{'frame' if index > start else 'Frame'} {index + 1} sampled at "
            f"{int(indices[index]) * frame_ms / 1000.0:.2f} seconds"
            for index in range(start, min(start + VIDEO_TUBELET, len(indices)))
        ]
        labels.append(" and ".join(parts) + ": ")
    return tuple(labels)


def _video_timeout(seconds: float) -> ImageError:
    return ImageError(
        f"video took longer than {seconds:g} seconds to decode; "
        "send a shorter or lower-resolution clip"
    )


def prepare_video(
    payload: bytes, decoder: str, fps: float, max_frames: int, seconds: float
) -> PreparedVideo:
    """Decode a video in a disposable `decoder` process (splash video-frames),
    resize its sampled frames as the processor does and pair them into
    tubelets, the last frame repeated where their count is odd."""
    from PIL import Image

    deadline = time.monotonic() + seconds
    # AVFoundation reads the container by its extension; .mp4 also opens a
    # QuickTime movie.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
        source.write(payload)
        source.flush()
        process = subprocess.Popen(
            [decoder, "video-frames", source.name, repr(fps), str(max_frames)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # The pipe blocks; a timer ends a decoder that outlives its time.
        expired = threading.Event()
        timer = threading.Timer(seconds, lambda: (expired.set(), process.kill()))
        timer.start()
        try:
            header = json.loads(process.stdout.readline(4096) or b"null")
            if not isinstance(header, dict):
                raise ImageError(
                    "video could not be decoded; MP4 and QuickTime are supported"
                )
            width, height = header["width"], header["height"]
            indices, source_fps = header["indices"], header["fps"]
            if not (
                0 < width * height <= MAX_SOURCE_PIXELS
                and 0 < len(indices) <= max_frames
                and source_fps > 0
            ):
                raise ImageError("video exceeds the decoder's limits")
            grid = video_resize(height, width)
            frames = []
            for _ in indices:
                raw = process.stdout.read(width * height * 3)
                if len(raw) != width * height * 3:
                    raise ImageError("video could not be decoded")
                frame = Image.frombytes("RGB", (width, height), raw)
                if (height, width) != grid:
                    frame = resize_float(frame, grid[1], grid[0])
                frames.append(frame.tobytes())
            if process.wait() != 0:
                raise ImageError("video could not be decoded")
        except (KeyError, TypeError, ValueError) as error:
            # A killed decoder's output ends early: say why.
            if expired.is_set():
                raise _video_timeout(seconds) from error
            if isinstance(error, ImageError):
                raise
            raise ImageError("video could not be decoded") from error
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
        if expired.is_set() or time.monotonic() > deadline:
            raise _video_timeout(seconds)
    if len(frames) % VIDEO_TUBELET:
        frames.append(frames[-1])
    grid_height, grid_width = grid[0] // PATCH, grid[1] // PATCH
    tubelets = []
    for start in range(0, len(frames), VIDEO_TUBELET):
        pixels = b"".join(frames[start : start + VIDEO_TUBELET])
        digest = hashlib.sha256(
            struct.pack("<III", grid_height, grid_width, VIDEO_TUBELET) + pixels
        )
        digest_lo, digest_hi = struct.unpack_from("<QQ", digest.digest())
        tubelets.append(
            PreparedImage(
                grid_height, grid_width, pixels, digest_lo, digest_hi, VIDEO_TUBELET
            )
        )
    return PreparedVideo(tuple(tubelets), _frame_labels(indices, source_fps))


class PreparedImages(list):
    """Append-only request batch, charged against the in-flight image budget
    until it is dropped: the request drops it once its frame is written, the
    last copy of its pixels in the server.

    Charge each occurrence, even cache hits: concatenation and the frame
    repeat those pixels.
    """

    def __init__(self, cache):
        super().__init__()
        self._cache = cache
        self._bytes = 0

    def append(self, image):
        size = len(image.pixels)
        with self._cache._lock:
            if size > self._cache.REQUEST_BUDGET_BYTES - self._cache._request_bytes:
                raise ImageCapacityError("in-flight image memory budget is full")
            self._cache._request_bytes += size
            self._bytes += size
        super().append(image)

    def __del__(self):
        with self._cache._lock:
            self._cache._request_bytes -= self._bytes


class ImageCache:
    """Byte-budgeted LRU of prepared images keyed by the raw image bytes."""

    # Cached prepared images, and the images of requests in flight.
    BUDGET_BYTES = 256 * 1024 * 1024
    REQUEST_BUDGET_BYTES = 256 * 1024 * 1024

    def __init__(self):
        # GC may finalize a cancelled batch during a cache insertion on this
        # same thread; returning its byte charge must not deadlock the cache.
        self._lock = threading.RLock()
        self._entries = LRUCache(self.BUDGET_BYTES)
        self._request_bytes = 0

    def request_batch(self):
        return PreparedImages(self)

    def prepare(
        self, payload: bytes, max_pixels: int, rule: str = "qwen"
    ) -> PreparedImage:
        key = hashlib.sha256(
            struct.pack("<I", max_pixels) + rule.encode() + b"\0" + payload
        ).digest()
        with self._lock:
            cached = self._entries.get(key)
        if cached is not None:
            return cached
        prepared = prepare(payload, max_pixels, rule)
        with self._lock:
            self._entries.put(key, prepared, len(prepared.pixels))
        return prepared

    def prepare_video(
        self, payload: bytes, decoder: str, fps: float, max_frames: int, seconds: float
    ) -> PreparedVideo:
        """A video resent in a conversation's history decodes once."""
        key = hashlib.sha256(
            struct.pack("<dI", fps, max_frames) + b"video\0" + payload
        ).digest()
        with self._lock:
            cached = self._entries.get(key)
        if cached is not None:
            return cached
        prepared = prepare_video(payload, decoder, fps, max_frames, seconds)
        size = sum(len(tubelet.pixels) for tubelet in prepared.tubelets)
        with self._lock:
            self._entries.put(key, prepared, size)
        return prepared

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._entries.bytes,
                "budget_bytes": self._entries.budget_bytes,
                "request_bytes": self._request_bytes,
                "request_budget_bytes": self.REQUEST_BUDGET_BYTES,
            }
