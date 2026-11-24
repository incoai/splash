"""The request options the nodes expose, in one place: ranges, sizes,
variants and the mapping from the nodes' widgets to POST /v1/videos fields.

No ComfyUI imports, so it can be tested on its own.
"""

from __future__ import annotations

# The ranges the server takes: `seconds` 4-15 (videos.clip_seconds) and a
# 32-bit `seed` (videos.parse_request), server/videos.py.
MIN_SECONDS = 4
MAX_SECONDS = 15
SEED_MAX = 2**32 - 1

# One `splash serve` serves every installed variant of MiniMax-H3
# (videos.hold_variants, server/videos.py), so all three nodes share it.
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
# The canvases the nodes offer, landscape then portrait, by resolution
# (docs/video.md, Video canvases). 864x480 exists server-side; the nodes do
# not offer it.
RESOLUTIONS = {"768p": ("1344x768", "768x1344"), "480p": ("832x480", "480x832")}
SIZES = [size for pair in RESOLUTIONS.values() for size in pair]
# The variant a node's server is started with when its variant is "auto".
DEFAULT_VARIANT = "pika-6"
# The variant of every request with references; it takes no other request.
REFERENCE_VARIANT = "pika-ref2va-6"
# "auto" omits the field and lets the server pick among those it serves.
VARIANTS = ["auto", "vdn-8", "vdn-50", "fasth3-8", "fasth3-4", DEFAULT_VARIANT]


def keyframe_size(resolution: str, width: int, height: int) -> str:
    """The `size` field of a resolution for a first frame of width x height:
    landscape when it is at least as wide as tall, else portrait."""
    landscape, portrait = RESOLUTIONS[resolution]
    return landscape if width >= height else portrait


def serve_variant(variant: str) -> str:
    """The variant to start a server with for a node's variant widget."""
    return variant if variant and variant != "auto" else DEFAULT_VARIANT


def video_fields(
    prompt: str,
    seconds: int,
    size: str,
    seed: int,
    audio: bool,
    preview: bool,
    variant: str = "auto",
) -> dict:
    """The text fields of a POST /v1/videos. `model` and `steps` are never
    sent; variant "auto" is omitted."""
    fields = {
        "prompt": prompt,
        "seconds": str(int(seconds)),
        "size": size,
        "seed": int(seed),
        "audio": bool(audio),
        "preview": bool(preview),
    }
    if variant and variant != "auto":
        fields["variant"] = variant
    return fields
