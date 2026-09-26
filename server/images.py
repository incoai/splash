"""Decode and resize images to patch-aligned uint8 RGB within the pixel budget.

The engine handles normalization and vision inference. Grid geometry and a
content digest identify image prefixes; a byte-budgeted cache reuses images
resent in chat history.
"""

import base64
import hashlib
import io
import math
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE
# Qwen3-VL's shortest_edge floor; the serving cap below is far below the model's.
MIN_PIXELS = 65_536
# Serving default: the engine's vision scratch covers 16,384 patches.
MAX_PIXELS = 4_194_304
# Bound codec work before full-resolution decode, independently of the model's
# resized input budget. Ordinary screenshots and up to 32 MP photos fit here.
MAX_SOURCE_PIXELS = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP", "GIF")
DEFAULT_CACHE_BYTES = 256 * 1024 * 1024


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

    @property
    def tokens(self) -> int:
        """Merged language tokens the image occupies."""
        return (self.grid_height // MERGE) * (self.grid_width // MERGE)


def decode_data_url(url) -> bytes:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ImageError("only data: image URLs are supported")
    header, _, encoded = url.partition(",")
    if ";base64" not in header:
        raise ImageError("image data URLs must be base64-encoded")
    if len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise ImageError("image exceeds the size limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ImageError("image data URL is not valid base64") from error
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError("image exceeds the size limit")
    return raw


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


def prepare(payload: bytes, max_pixels: int = MAX_PIXELS) -> PreparedImage:
    """Decode, convert to RGB, resize into the grid, and digest the pixels."""
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(payload), formats=IMAGE_FORMATS) as decoded:
            if decoded.height * decoded.width > MAX_SOURCE_PIXELS:
                raise ImageError("source image exceeds the pixel limit")
            stored = decoded.size
            height, width = smart_resize(decoded.height, decoded.width, max_pixels)
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
                image = image.resize((width, height), Image.Resampling.BICUBIC)
            pixels = image.tobytes()
    except ImageError:
        raise
    except Exception as error:
        raise ImageError("image could not be decoded") from error
    grid_height, grid_width = height // PATCH, width // PATCH
    digest = hashlib.sha256(struct.pack("<II", grid_height, grid_width) + pixels)
    digest_lo, digest_hi = struct.unpack_from("<QQ", digest.digest())
    return PreparedImage(grid_height, grid_width, pixels, digest_lo, digest_hi)


class PreparedImages(list):
    """Append-only request batch; its owner travels with the pixel payload.

    Charge each occurrence, even cache hits: concatenation and native transport
    repeat those pixels. Release only when every holder drops the batch, not
    when HTTP disconnects while native work or a mask task still owns it.
    """

    def __init__(self, cache):
        super().__init__()
        self._cache = cache
        self._bytes = 0

    def append(self, image):
        size = len(image.pixels)
        with self._cache._lock:
            if size > self._cache.request_budget_bytes - self._cache._request_bytes:
                raise ImageCapacityError("in-flight image memory budget is full")
            self._cache._request_bytes += size
            self._bytes += size
        super().append(image)

    def __del__(self):
        with self._cache._lock:
            self._cache._request_bytes -= self._bytes


class ImageCache:
    """Byte-budgeted LRU of prepared images keyed by the raw image bytes."""

    def __init__(
        self,
        budget_bytes: int = DEFAULT_CACHE_BYTES,
        request_budget_bytes: int = DEFAULT_CACHE_BYTES,
    ):
        if budget_bytes < 0 or request_budget_bytes <= 0:
            raise ValueError("invalid image memory budget")
        self.budget_bytes = budget_bytes
        self.request_budget_bytes = request_budget_bytes
        # GC may finalize a cancelled batch during a cache insertion on this
        # same thread; returning its byte charge must not deadlock the cache.
        self._lock = threading.RLock()
        self._entries: OrderedDict[bytes, PreparedImage] = OrderedDict()
        self._bytes = 0
        self._request_bytes = 0

    def request_batch(self):
        return PreparedImages(self)

    def prepare(self, payload: bytes, max_pixels: int) -> PreparedImage:
        key = hashlib.sha256(struct.pack("<I", max_pixels) + payload).digest()
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached
        prepared = prepare(payload, max_pixels)
        if len(prepared.pixels) > self.budget_bytes:
            return prepared
        with self._lock:
            if key not in self._entries:
                self._entries[key] = prepared
                self._bytes += len(prepared.pixels)
                while self._bytes > self.budget_bytes:
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= len(evicted.pixels)
        return prepared

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "budget_bytes": self.budget_bytes,
                "request_bytes": self._request_bytes,
                "request_budget_bytes": self.request_budget_bytes,
            }
