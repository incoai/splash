"""ComfyUI values to the files a Splash request carries.

Duck-typed: an IMAGE is anything np.asarray turns into [B,]H,W,C floats in
[0, 1] (a torch tensor, or numpy in tests); an AUDIO is {"waveform":
[B,]C,T, "sample_rate": int}; a VIDEO has ComfyUI's VideoInput methods.
"""

from __future__ import annotations

import io
import wave

import numpy as np
from PIL import Image

from .splash_client import FilePart


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def image_size(image) -> tuple[int, int]:
    """(width, height) of the first image of an IMAGE batch."""
    array = _numpy(image)
    if array.ndim == 4:
        array = array[0]
    return int(array.shape[1]), int(array.shape[0])


def image_to_png(image) -> bytes:
    """The first image of an IMAGE batch as PNG bytes (RGB, 8 bits)."""
    array = _numpy(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 2:
        array = array[..., None]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = array[..., :3]
    pixels = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(out, format="PNG")
    return out.getvalue()


def audio_to_wav(audio) -> bytes:
    """An AUDIO dict (first item of the batch) as a 16-bit PCM WAV."""
    waveform = _numpy(audio["waveform"])
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform[None]
    channels = int(waveform.shape[0])
    samples = (np.clip(waveform, -1.0, 1.0) * 32767.0).round().astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(int(audio["sample_rate"]))
        writer.writeframes(np.ascontiguousarray(samples.T).tobytes())
    return out.getvalue()


def _is_mp4(video) -> bool:
    try:
        name = video.get_container_format()
    except Exception:
        return False
    return "mp4" in str(name).split(",")


def _untrimmed(video) -> bool:
    window = getattr(video, "get_active_trim_window", None)
    if window is None:
        return True
    try:
        start, duration = window()
    except Exception:
        return False
    return not start and not duration


def video_to_mp4(video) -> bytes:
    """A VIDEO as mp4 bytes: its source file when that is an untrimmed mp4,
    otherwise the video re-saved as mp4 (save_to)."""
    source = video.get_stream_source() if hasattr(video, "get_stream_source") else None
    if source is not None and _is_mp4(video) and _untrimmed(video):
        if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
            with open(source, "rb") as handle:
                return handle.read()
        if hasattr(source, "getvalue"):
            return source.getvalue()
    out = io.BytesIO()
    video.save_to(out, format="mp4")
    return out.getvalue()


def reference_parts(pictures, videos, audio_clips) -> list:
    """The `references` file parts of Autogrow dicts, in order: pictures,
    videos, audio clips, each in slot order."""
    files = []
    for i, image in enumerate((pictures or {}).values(), 1):
        files.append(FilePart("references", f"image_{i}.png", "image/png", image_to_png(image)))
    for i, video in enumerate((videos or {}).values(), 1):
        files.append(FilePart("references", f"video_{i}.mp4", "video/mp4", video_to_mp4(video)))
    for i, audio in enumerate((audio_clips or {}).values(), 1):
        files.append(FilePart("references", f"audio_{i}.wav", "audio/wav", audio_to_wav(audio)))
    return files
