"""Decode images and prepare them as patch-aligned uint8 RGB within the pixel
budget, as the served family's vision tower takes them (ImageFormat): Qwen's
resized into its grid, Inkling's padded to it.

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
from collections.abc import Callable
from dataclasses import dataclass

from .lru import LRUCache

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE
# Inkling 2's tower: 20-pixel patches, merged 2x2 into one token of 40x40.
INKLING_PATCH = 20
INKLING_FACTOR = INKLING_PATCH * MERGE
# The 8-pixel checkerboard of light and darker grey Inkling's preprocessing
# composites a transparent image onto (transformers' Inkling2ImageProcessor).
CHECKERBOARD = 8
CHECKERBOARD_GREYS = (204, 153)
# Qwen3-VL's shortest_edge floor; the serving cap below is far below the model's.
MIN_PIXELS = 65_536
# Serving default and largest cap: 16,384 patches, the protocol's per-image ceiling.
MAX_PIXELS = 4_194_304
# An image's longer side over its shorter one, at most (Qwen2-VL's smart resize).
MAX_ASPECT = 200
# Bound codec work before full-resolution decode, independently of the model's
# resized input budget. Ordinary screenshots and up to 32 MP photos fit here.
MAX_SOURCE_PIXELS = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP", "GIF")


class ImageError(ValueError):
    pass


# What Anthropic's API says of an image it cannot process. Its clients act on
# the words: Claude Code replaces that image in the conversation with a note
# and goes on, where any other 400 fails every later request that resends it.
UNPROCESSABLE = "Could not process image"


class ImageCapacityError(ImageError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedImage:
    grid_height: int
    grid_width: int
    pixels: bytes
    digest_lo: int
    digest_hi: int
    # The side of a grid patch in pixels, the tower's.
    patch: int = PATCH
    # The image's own pixels at the grid's top left, before its padding to
    # whole tokens; zero, the whole grid.
    content_height: int = 0
    content_width: int = 0

    @property
    def tokens(self) -> int:
        """Merged language tokens the image occupies."""
        return (self.grid_height // MERGE) * (self.grid_width // MERGE)


# What standard base64 decoding is lenient about (decode_base64): ASCII
# whitespace, as `base64` on Linux wraps its output at 76 columns, and the
# URL-safe alphabet's two characters.
_BASE64_LENIENCE = str.maketrans(
    {" ": None, "\t": None, "\n": None, "\r": None, "-": "+", "_": "/"}
)


def decode_base64(encoded: str) -> bytes:
    """The bytes of a base64 text, standard or URL-safe, wrapped or not,
    padded or not; a ValueError for any other text."""
    text = encoded.translate(_BASE64_LENIENCE)
    return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)


def base64_limit(size: int) -> int:
    """The longest base64 text of size bytes that decode_base64 takes,
    which bounds a text before it is decoded: padded, and wrapped at 76
    columns with a two-character break after each line."""
    characters = 4 * ((size + 2) // 3)
    return characters + 2 * ((characters + 75) // 76)


def decode_data_url(url) -> bytes:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ImageError("only data: image URLs are supported")
    header, _, encoded = url.partition(",")
    if ";base64" not in header:
        raise ImageError("image data URLs must be base64-encoded")
    if len(encoded) > base64_limit(MAX_IMAGE_BYTES):
        raise ImageError("image exceeds the size limit")
    try:
        raw = decode_base64(encoded)
    except (ValueError, TypeError) as error:
        raise ImageError("image data URL is not valid base64") from error
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError("image exceeds the size limit")
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
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise ImageError(f"absolute aspect ratio must be at most {MAX_ASPECT}")
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


def _upright(decoded):
    """Turns a decoded image upright in place and reports whether its sides
    swapped. Cameras store the sensor's orientation and an EXIF tag that
    turns it upright for display; show the model the upright one."""
    from PIL import ImageOps

    stored = decoded.size
    try:
        ImageOps.exif_transpose(decoded, in_place=True)
    except Exception:
        pass  # A malformed tag leaves the stored orientation.
    return decoded.size != stored


def _digest(grid_height, grid_width, pixels, content=None):
    """The 128-bit digest of a grid's pixels and, for an image padded to it,
    its own pixels' height and width, which the encoder pads past."""
    head = struct.pack("<II", grid_height, grid_width)
    if content is not None:
        head += struct.pack("<II", *content)
    digest = hashlib.sha256(head + pixels)
    return struct.unpack_from("<QQ", digest.digest())


def prepare(payload: bytes, max_pixels: int = MAX_PIXELS) -> PreparedImage:
    """Decode, convert to RGB, resize into the grid, and digest the pixels."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload), formats=IMAGE_FORMATS) as decoded:
            if decoded.height * decoded.width > MAX_SOURCE_PIXELS:
                raise ImageError("source image exceeds the pixel limit")
            height, width = smart_resize(decoded.height, decoded.width, max_pixels)
            if _upright(decoded):
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
        raise ImageError(UNPROCESSABLE) from error
    grid_height, grid_width = height // PATCH, width // PATCH
    digest_lo, digest_hi = _digest(grid_height, grid_width, pixels)
    return PreparedImage(grid_height, grid_width, pixels, digest_lo, digest_hi)


def inkling_size(height: int, width: int, max_pixels: int) -> tuple[int, int]:
    """The size Inkling's preprocessing gives an image: its own, which it
    pads to whole 40-pixel tokens, unless those exceed the patch budget the
    serving cap gives (max_patches, the encoder's bound, in Qwen's 16-pixel
    patches; at the default, 4,096 of Inkling's tokens, llama.cpp's budget).
    Then its sides are scaled alike, a pixel each at least, to the largest
    size whose padded tokens fit the budget: its aspect ratio stays within a
    pixel's rounding, and the padding takes up the rest. The reference
    processor never resizes."""
    if height <= 0 or width <= 0:
        raise ImageError("image dimensions must be positive")
    if not MIN_PIXELS <= max_pixels <= MAX_PIXELS:
        raise ImageError("invalid image pixel budget")
    tokens = max_patches(max_pixels) // (MERGE * MERGE)

    def fits(first, second):
        return (
            math.ceil(first / INKLING_FACTOR) * math.ceil(second / INKLING_FACTOR)
            <= tokens
        )

    if fits(height, width):
        return height, width
    longer, shorter = max(height, width), min(height, width)

    def scaled(side):
        """The shorter side, scaled as the longer one is to side."""
        return max(1, (shorter * side + longer // 2) // longer)

    # The tokens grow with the longer side, so a bisection finds the largest
    # that fits; one pixel always does.
    low, high = 1, longer
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle, scaled(middle)):
            low = middle
        else:
            high = middle - 1
    return (low, scaled(low)) if height >= width else (scaled(low), low)


def composite_on_checkerboard(rgba):
    """An RGBA image as RGB over Inkling's checkerboard, in integers as its
    processor rounds them: (rgb a + grey (255 - a) + 127) // 255, the grey
    204 where the 8-pixel squares' row and column sum to an even number, 153
    elsewhere. A band of rows at a time, of about a megapixel, so an image
    of the source limit costs about its own pixels' memory again."""
    import numpy as np
    from PIL import Image

    values = np.asarray(rgba)
    height, width = values.shape[:2]
    composite = np.empty((height, width, 3), np.uint8)
    columns = (np.arange(width) // CHECKERBOARD)[None, :]
    band = max(1, (1 << 20) // width)
    for top in range(0, height, band):
        rows = (np.arange(top, min(top + band, height)) // CHECKERBOARD)[:, None]
        grey = np.where((rows + columns) % 2 == 0, *CHECKERBOARD_GREYS)
        grey = grey.astype(np.uint32)[..., None]
        part = values[top : top + band].astype(np.uint32)
        alpha = part[..., 3:]
        composite[top : top + band] = (
            part[..., :3] * alpha + grey * (255 - alpha) + 127
        ) // 255
    return Image.fromarray(composite, "RGB")


def prepare_inkling(payload: bytes, max_pixels: int = MAX_PIXELS) -> PreparedImage:
    """Decode an image as Inkling 2's processor prepares it: upright, RGB (a
    transparent one composited onto its checkerboard), at its own size
    (inkling_size), padded right and bottom with zeros to whole 40-pixel
    tokens of 2x2 patches of 20 pixels, and digest the pixels with its own
    size. The reference pads with -1 on the 0-255 scale, which no uint8
    holds: the frame's padding is zero, and the encoder, which the span's
    content size tells where the image ends, takes it as -1."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload), formats=IMAGE_FORMATS) as decoded:
            if decoded.height * decoded.width > MAX_SOURCE_PIXELS:
                raise ImageError("source image exceeds the pixel limit")
            _upright(decoded)
            height, width = inkling_size(decoded.height, decoded.width, max_pixels)
            # A transparent image is resized first, its alpha premultiplied,
            # so its checkerboard keeps 8-pixel squares at any size.
            transparent = decoded.has_transparency_data
            image = decoded.convert("RGBA" if transparent else "RGB")
            if (image.height, image.width) != (height, width):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            if transparent:
                image = composite_on_checkerboard(image)
            grid_height = math.ceil(height / INKLING_FACTOR) * MERGE
            grid_width = math.ceil(width / INKLING_FACTOR) * MERGE
            canvas = Image.new(
                "RGB", (grid_width * INKLING_PATCH, grid_height * INKLING_PATCH)
            )
            canvas.paste(image, (0, 0))
            pixels = canvas.tobytes()
    except ImageError:
        raise
    except Exception as error:
        raise ImageError(UNPROCESSABLE) from error
    digest_lo, digest_hi = _digest(grid_height, grid_width, pixels, (height, width))
    return PreparedImage(
        grid_height,
        grid_width,
        pixels,
        digest_lo,
        digest_hi,
        INKLING_PATCH,
        height,
        width,
    )


@dataclass(frozen=True)
class ImageFormat:
    """How a family's vision tower takes an image: the side of its grid's
    patches in pixels (the engine's ImageSpan counts them), the preparation
    of a payload within a pixel cap, and whether a checkpoint may leave its
    processor's configuration out (defaulted): Inkling2ImageProcessor's is its
    class's defaults alone, which an MLX conversion does not save, while a
    Qwen processor's sizes are its checkpoint's."""

    name: str
    patch: int
    prepare: Callable[[bytes, int], PreparedImage]
    defaulted: bool = False


QWEN_IMAGES = ImageFormat("qwen", PATCH, prepare)
INKLING_IMAGES = ImageFormat("inkling", INKLING_PATCH, prepare_inkling, defaulted=True)


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

    def __init__(self, preparation=prepare):
        # The served family's preparation (ImageFormat.prepare).
        self._prepare = preparation
        # GC may finalize a cancelled batch during a cache insertion on this
        # same thread; returning its byte charge must not deadlock the cache.
        self._lock = threading.RLock()
        self._entries = LRUCache(self.BUDGET_BYTES)
        self._request_bytes = 0

    def request_batch(self):
        return PreparedImages(self)

    def prepare(self, payload: bytes, max_pixels: int) -> PreparedImage:
        key = hashlib.sha256(struct.pack("<I", max_pixels) + payload).digest()
        with self._lock:
            cached = self._entries.get(key)
        if cached is not None:
            return cached
        prepared = self._prepare(payload, max_pixels)
        with self._lock:
            self._entries.put(key, prepared, len(prepared.pixels))
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
