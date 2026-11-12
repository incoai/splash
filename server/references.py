"""Ref2VA references: the media a video request is conditioned on, decoded
and normalized here as Pika H3 Flex's host code does, so the engine receives
pixels and samples only.

A reference is an image (PNG or JPEG), a video (MP4, QuickTime or WebM)
with its soundtrack, or an audio clip (MP3, WAV, M4A, AAC; a voice or music
reference, beside an image or video one). Each is decoded as
pika_h3.core.media does (PyAV: a video's frames as rgb24 for at most 16 s at
its average rate, a soundtrack or an audio clip resampled to 32 kHz stereo
fltp; PIL for an image, converted to RGB, EXIF orientation not applied) and
normalized as pika_h3.core.conditioning.normalize_references does: a video
onto the 24 fps grid (diffusers' MiniMaxH3Ref2VASetupStep.
_normalize_video_condition: each frame held until the slot of the next),
truncated to the clip's frame count and LANCZOS-resized to the canvas its
own aspect ratio resolves to on the clip's tier (resolve_canvas_size), its
soundtrack (and an audio clip) truncated to round(frames / 24 * 32000)
samples; an image
LANCZOS-resized to the tier's reference short edge (1024 at 480p, 2048 at
768p, upscaling, no area cap, both sides rounded to 32). Pika's limits hold:
at most 9 images, 3 videos and 3 audio clips, 12 in all, each video and each
audio clip 2 to 15 s of decoded source, the videos 15 s at most in all and
the audio clips likewise, a video at least 22 frames once normalized, and an
image or a video among them.
"""

from __future__ import annotations

import io
import math
import mimetypes
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from install.families import H3

from .errors import APIError

# A reference file's bytes, and an image's decoded pixels (images.py's).
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_SOURCE_PIXELS = 32 * 1024 * 1024
IMAGE_TYPES = {"image/png": "PNG", "image/jpeg": "JPEG"}
VIDEO_TYPES = ("video/mp4", "video/quicktime", "video/webm")
AUDIO_TYPES = (
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/aac", "audio/x-aac", "audio/aacp",
)  # fmt: skip
# PyAV, which decodes video and audio references: the release leaves it out
# (its wheels bundle FFmpeg), so the user installs this pin into the
# environment Splash runs in (dev/requirements.txt holds the same).
PYAV = "av==19.0.2"


@dataclass(frozen=True)
class ReferenceFile:
    """A reference as the request carries it: its content type and bytes."""

    content_type: str
    data: bytes

    def __repr__(self):
        return f"ReferenceFile({self.content_type!r}, {len(self.data)} bytes)"


@dataclass(frozen=True, eq=False)
class Normalized:
    """A normalized reference: kind "image", "video" or "audio", its uint8
    RGB pixels [frames][height][width][3] (an image's frames are 1, an audio
    clip's 0) and its soundtrack, fp32 [2][samples] at 32 kHz (a video's, or
    None; an audio clip's)."""

    kind: str
    pixels: np.ndarray
    soundtrack: np.ndarray | None = None

    @property
    def frames(self):
        return self.pixels.shape[0]

    @property
    def size(self):
        return self.pixels.shape[2], self.pixels.shape[1]

    @property
    def has_audio(self):
        return self.soundtrack is not None

    def __repr__(self):
        sound = (
            "" if self.soundtrack is None else f" + {self.soundtrack.shape[1]} samples"
        )
        return f"Normalized({self.kind} {self.size[0]}x{self.size[1]} x {self.frames}{sound})"


def kind(content_type):
    """A reference's kind by its content type: "image", "video" or
    "audio"; another type is refused."""
    base = (content_type or "").partition(";")[0].strip().lower()
    if base in IMAGE_TYPES:
        return "image"
    if base in VIDEO_TYPES:
        return "video"
    if base in AUDIO_TYPES:
        return "audio"
    raise APIError(
        400,
        f"a reference of content type {base or 'none'} is not supported: "
        "image/png, image/jpeg, video/mp4, video/quicktime, video/webm, "
        "audio/mpeg, audio/wav, audio/mp4 (M4A) or audio/aac",
    )


def content_type(data: bytes, name: str = ""):
    """A reference file's content type, by its leading bytes, else by its
    name's extension: what `splash generate --reference` sends."""
    head = bytes(data[:16])
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B "):
            return "audio/mp4"
        return "video/quicktime" if brand.startswith(b"qt") else "video/mp4"
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "audio/wav"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if head[:2] in (b"\xff\xf1", b"\xff\xf9"):
        return "audio/aac"
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


def read(path: Path):
    """The ReferenceFile of a path, its type by content_type."""
    data = path.read_bytes()
    return ReferenceFile(content_type(data, path.name), data)


# ---- decoding (pika_h3.core.media)


def pyav():
    """The av module, imported only where a video or audio reference is
    decoded; an APIError(400) naming the command that installs it where it
    is missing. Nothing installs it at request time."""
    try:
        import av
    except ModuleNotFoundError as error:
        raise APIError(
            400,
            "video and audio references need PyAV, which Splash does not ship;"
            f" install it with: {shlex.quote(sys.executable)} -m pip install {PYAV}",
        ) from error
    return av


def decode_image(data):
    """A PIL RGB image, as Reference.from_spec opens one."""
    from PIL import Image

    try:
        with Image.open(
            io.BytesIO(data), formats=tuple(IMAGE_TYPES.values())
        ) as decoded:
            if decoded.width * decoded.height > MAX_SOURCE_PIXELS:
                raise APIError(400, "a reference image exceeds the pixel limit")
            return decoded.convert("RGB")
    except APIError:
        raise
    except Exception as error:
        raise APIError(
            400, "a reference image could not be decoded as PNG or JPEG"
        ) from error


def decode_soundtrack(
    data, sampling_rate=H3.sample_rate, max_seconds=H3.max_source_seconds
):
    """A video's soundtrack, fp32 [2][samples] at sampling_rate, or None
    without one (media.decode_audio_stereo): PyAV's resampler converts the
    rate and the channel layout in one pass."""
    av = pyav()

    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="stereo", rate=sampling_rate
        )
        chunks, samples = [], 0
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunk = resampled.to_ndarray()
                chunks.append(chunk)
                samples += chunk.shape[-1]
            if samples >= max_seconds * sampling_rate:
                break
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())
    if not chunks:
        return None
    waveform = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)
    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise APIError(
            400, f"a reference soundtrack decoded to {waveform.shape}, not stereo"
        )
    if waveform.shape[1] == 0:
        return None
    return waveform[:, : round(max_seconds * sampling_rate)]


def decode_video(data, max_seconds=H3.max_source_seconds):
    """A video's frames uint8 [n][h][w][3], its frame rate and its
    soundtrack (media.decode_video_reference): at most ceil(max_seconds x
    fps) frames."""
    av = pyav()

    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.video:
                raise APIError(400, "a reference video has no video stream")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.base_rate or 0)
            if not np.isfinite(fps) or fps <= 0:
                raise APIError(400, "a reference video has no valid frame rate")
            frames = []
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= int(np.ceil(max_seconds * fps)):
                    break
        if not frames:
            raise APIError(400, "a reference video holds no decoded frame")
        return np.stack(frames), fps, decode_soundtrack(data, max_seconds=max_seconds)
    except APIError:
        raise
    except Exception as error:
        raise APIError(
            400, f"a reference video could not be decoded: {error}"
        ) from error


# ---- normalization (pika_h3.core.conditioning, diffusers' minimax_h3)


def resolve_canvas_size(
    aspect_width, aspect_height, short_edge, max_pixels, multiple=H3.canvas_step
):
    """(height, width) of the canvas an aspect ratio resolves to (diffusers'
    minimax_h3 resolve_canvas_size)."""
    ratio = aspect_width / aspect_height
    if not H3.aspect_range[0] <= ratio <= H3.aspect_range[1]:
        raise APIError(
            400,
            f"a reference video's aspect ratio must be within 1:4 and 4:1, not {ratio:g}",
        )
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    return max(multiple, round(height / multiple) * multiple), max(
        multiple, round(width / multiple) * multiple
    )


def reference_canvas_size(width, height, short_edge, multiple=H3.canvas_step):
    """(height, width) an image reference is encoded at (Pika's
    pipeline_geometry.reference_canvas_size with the tier rule: upscaling,
    no area cap)."""
    if (
        width <= 0
        or height <= 0
        or not H3.aspect_range[0] <= width / height <= H3.aspect_range[1]
    ):
        raise APIError(
            400, f"a reference image must be within 1:4 and 4:1, not {width}x{height}"
        )
    scale = short_edge / min(width, height)
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def normalize_video(frames, fps, num_frames, short_edge, max_pixels):
    """A video's frames on the 24 fps grid, truncated to num_frames, on its
    canvas (MiniMaxH3Ref2VASetupStep._normalize_video_condition)."""
    from PIL import Image

    if fps != H3.fps:
        scale = H3.fps / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        frames = np.repeat(
            frames,
            np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)),
            axis=0,
        )
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(
        frames.shape[2], frames.shape[1], short_edge, max_pixels
    )
    if frames.shape[1:3] == (height, width):
        return np.ascontiguousarray(frames)
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
            )
            for frame in frames
        ]
    )


def decode_audio(data):
    """An audio reference's samples, fp32 [2][samples] at 32 kHz (as
    Reference.from_spec decodes one: decode_audio_stereo)."""
    try:
        wave = decode_soundtrack(data)
    except APIError:
        raise
    except Exception as error:
        raise APIError(
            400, f"an audio reference could not be decoded: {error}"
        ) from error
    if wave is None:
        raise APIError(400, "an audio reference has no audio stream")
    return wave.astype(np.float32)


# An audio reference's pixels: none.
NO_PIXELS = np.zeros((0, 0, 0, 3), np.uint8)


def _soundtrack(wave, max_samples):
    """A soundtrack truncated to the clip (normalize_references)."""
    wave = np.asarray(wave, dtype=np.float32)
    if (
        wave.ndim != 2
        or wave.shape[0] not in (1, 2)
        or wave.shape[1] == 0
        or not np.isfinite(wave).all()
    ):
        raise APIError(
            400,
            "a reference soundtrack must be finite, nonempty mono or stereo samples",
        )
    wave = np.repeat(wave, 2, axis=0) if wave.shape[0] == 1 else wave
    return np.ascontiguousarray(wave[:, :max_samples])


def tier(width, height):
    """The tier rule of a canvas: its short edge's (the canvases Splash
    renders are of the 480p and 768p tiers)."""
    return H3.reference_tiers[min(width, height)]


def normalize(files, num_frames, width, height):
    """The references of a request in presentation order, each decoded and
    normalized for a clip of num_frames frames on a width x height canvas;
    APIError(400) for a reference Pika refuses."""
    from PIL import Image

    kinds = [kind(f.content_type) for f in files]
    if (
        len(kinds) > H3.max_references
        or kinds.count("image") > H3.max_images
        or kinds.count("video") > H3.max_videos
        or kinds.count("audio") > H3.max_audio
    ):
        raise APIError(
            400,
            f"references are at most {H3.max_images} images, {H3.max_videos} videos and "
            f"{H3.max_audio} audio clips, {H3.max_references} in all",
        )
    if "image" not in kinds and "video" not in kinds:
        raise APIError(
            400, "an audio reference needs an image or video reference beside it"
        )
    if any(len(f.data) > MAX_FILE_BYTES for f in files):
        raise APIError(400, f"a reference file holds at most {MAX_FILE_BYTES} bytes")
    short_edge, max_pixels, image_edge = tier(width, height)
    max_samples = round(num_frames / H3.fps * H3.sample_rate)
    decoders = {"image": decode_image, "video": decode_video, "audio": decode_audio}
    decoded = [decoders[k](f.data) for k, f in zip(kinds, files)]
    seconds = [len(d[0]) / d[1] for k, d in zip(kinds, decoded) if k == "video"]
    audio_seconds = [
        d.shape[-1] / H3.sample_rate for k, d in zip(kinds, decoded) if k == "audio"
    ]
    if (
        sum(seconds) > H3.max_total_video_seconds + 1e-6
        or sum(audio_seconds) > H3.max_total_video_seconds + 1e-6
    ):
        raise APIError(
            400,
            f"reference videos take {H3.max_total_video_seconds:g} s at most in all, and audio references likewise",
        )
    if any(not H3.video_seconds[0] <= s <= H3.video_seconds[1] + 1e-6 for s in seconds):
        raise APIError(400, "each reference video must be 2 to 15 seconds")
    if any(
        not H3.video_seconds[0] <= s <= H3.video_seconds[1] + 1e-6
        for s in audio_seconds
    ):
        raise APIError(400, "each audio reference must be 2 to 15 seconds")
    normalized = []
    for k, d in zip(kinds, decoded):
        if k == "audio":
            normalized.append(
                Normalized("audio", NO_PIXELS, _soundtrack(d, max_samples))
            )
            continue
        if k == "image":
            h, w = reference_canvas_size(d.width, d.height, image_edge)
            pixels = np.asarray(d.resize((w, h), Image.Resampling.LANCZOS))[None]
            normalized.append(Normalized("image", np.ascontiguousarray(pixels)))
            continue
        frames, fps, wave = d
        if wave is not None:
            wave = _soundtrack(wave, max_samples)
        frames = normalize_video(frames, fps, num_frames, short_edge, max_pixels)
        if len(frames) < H3.min_video_frames:
            raise APIError(
                400,
                f"a reference video must hold at least {H3.min_video_frames} frames at 24 fps",
            )
        normalized.append(Normalized("video", frames, wave))
    return normalized
