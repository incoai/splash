"""Audio files as the audio server reads and writes them: an upload decoded
to the 16 kHz mono float32 samples the ASR engine takes, split where it is
longer than one request may be, and the engine's 24 kHz int16 output as a
WAV file or stream.

A PCM WAV (8, 16, 24 or 32-bit integer samples) is read with the standard
library; any other upload is decoded in process by PyAV, whose wheel carries
its own FFmpeg libraries: audio files (FLAC, MP3, M4A, AAC, Ogg/Opus, AIFF,
float or compressed WAV...) and the first audio track of a video (MP4, MOV,
MKV, WebM...). No ffmpeg on PATH is needed, so a broken system installation
cannot break transcription.

Resampling quality: a rate that is a whole multiple of 16 kHz (32, 48, 96
kHz) is decimated through a windowed-sinc low-pass filter, which is exact
up to the filter's transition band near 8 kHz. Any other rate above 16 kHz
(44.1, 22.05 kHz...) passes the same low-pass filter and is then linearly
interpolated at the 16 kHz instants, which rolls off the top of the band a
little; a rate below 16 kHz (8 kHz telephone audio) is linearly
interpolated, adding no content above its own half rate. All of it is well
within what the recognizer's log-mel front end resolves.
"""

from __future__ import annotations

import io
import math
import struct
import wave
from dataclasses import dataclass

import av
import numpy as np

from .errors import APIError

ASR_SAMPLE_RATE = 16_000
TTS_SAMPLE_RATE = 24_000
# What one transcription request carries at most, the engine's limit
# (AsrEngineConfig::maxAudioSeconds, from which its audio workspace and its
# smallest KV pool follow): longer audio is cut into segments of at most this,
# at quiet points. And what an upload may decode to in all: longer audio is
# refused rather than held as hours of float32 samples.
MAX_REQUEST_SECONDS = 2 * 60
MAX_UPLOAD_SECONDS = 2 * 60 * 60
# Long audio is cut near each request boundary, at the quietest 100 ms
# window within 5 s before it, so a request ends between words. Audio longer
# than PARALLEL_MINIMUM_SECONDS is cut into segments decoded concurrently
# on the engine's lanes: about one per lane, none shorter than that.
PARALLEL_MINIMUM_SECONDS = 30.0
SPLIT_SEARCH_SECONDS = 5.0
SPLIT_WINDOW_SECONDS = 0.1
# WAV headers: RIFF, then fmt and data chunks.
_RIFF = struct.Struct("<4sI4s")
_FMT = struct.Struct("<4sIHHIIHH")
_DATA = struct.Struct("<4sI")
WAV_HEADER_BYTES = _RIFF.size + _FMT.size + _DATA.size
# The data length of a WAV stream whose length is not known when its header
# is written; players read such a file to its end.
UNKNOWN_LENGTH = 0xFFFFFFFF


class UnsupportedAudio(APIError):
    """An upload in a format the server cannot decode here."""

    def __init__(self, message):
        super().__init__(415, message, "unsupported_media_type")


@dataclass(frozen=True, slots=True)
class Segment:
    """One request's span of the decoded samples."""

    start: int
    end: int

    def seconds(self, rate: int = ASR_SAMPLE_RATE) -> tuple[float, float]:
        return self.start / rate, self.end / rate


def decode_upload(data: bytes, filename: str | None = None) -> np.ndarray:
    """An upload's samples as float32, mono, at ASR_SAMPLE_RATE: a PCM WAV
    decoded here, anything else by PyAV. An APIError says what is wrong
    with an upload that cannot be decoded."""
    if not data:
        raise APIError(400, "the audio file is empty")
    samples = None
    if data[:4] == b"RIFF":
        samples = _decode_wav(data)
    if samples is None:
        samples = _decode_with_av(data)
    if not len(samples):
        raise APIError(400, "the audio file holds no samples")
    if len(samples) > MAX_UPLOAD_SECONDS * ASR_SAMPLE_RATE:
        raise APIError(
            400,
            f"the audio is longer than {MAX_UPLOAD_SECONDS // 3600} hours; "
            "split it before uploading",
        )
    return samples


def _decode_wav(data: bytes) -> np.ndarray | None:
    """A PCM WAV's samples at ASR_SAMPLE_RATE; None for a WAV the standard
    library does not read (float, compressed), which PyAV may."""
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError, ValueError, struct.error):
        return None
    if channels < 1 or rate < 1 or width not in (1, 2, 3, 4):
        raise APIError(400, "the WAV file's format is invalid")
    frames = frames[: len(frames) - len(frames) % (channels * width)]
    if width == 1:
        values = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        values = (values - 128.0) / 128.0
    elif width == 2:
        values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        signed = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        signed = np.where(signed >= 1 << 23, signed - (1 << 24), signed)
        values = signed.astype(np.float32) / float(1 << 23)
    else:
        values = np.frombuffer(frames, dtype="<i4").astype(np.float32) / float(1 << 31)
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return resample(np.ascontiguousarray(values, dtype=np.float32), rate)


def _decode_with_av(data: bytes) -> np.ndarray:
    """The first audio track of any container FFmpeg reads, resampled by
    FFmpeg's resampler and averaged to mono as a WAV is (FFmpeg's own
    downmix adds the channels at -3 dB, which clips loud stereo). Decoding
    stops a second past MAX_UPLOAD_SECONDS, which decode_upload then
    refuses."""
    limit = (MAX_UPLOAD_SECONDS + 1) * ASR_SAMPLE_RATE
    chunks: list[np.ndarray] = []
    total = 0
    try:
        container = av.open(io.BytesIO(data), mode="r")
    except av.error.FFmpegError as error:
        raise UnsupportedAudio(
            f"the file is not in an audio or video format the server reads ({error.strerror})"
        ) from None
    try:
        if not container.streams.audio:
            raise APIError(400, "the file holds no audio track")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", rate=ASR_SAMPLE_RATE)

        def keep(frames) -> None:
            nonlocal total
            for frame in frames:
                samples = frame.to_ndarray().mean(axis=0, dtype=np.float32)
                chunks.append(samples)
                total += len(samples)

        for frame in container.decode(stream):
            keep(resampler.resample(frame))
            if total > limit:
                break
        else:
            keep(resampler.resample(None))
    except av.error.FFmpegError as error:
        raise APIError(
            400, f"the audio file could not be decoded: {error.strerror}"
        ) from None
    finally:
        container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def _low_pass(samples: np.ndarray, rate: int, cutoff: float) -> np.ndarray:
    """samples through a windowed-sinc (Hann) low-pass filter at cutoff Hz:
    the anti-aliasing filter of a resampling down from rate."""
    half = 16 * math.ceil(rate / ASR_SAMPLE_RATE)
    taps = np.arange(-half, half + 1, dtype=np.float64)
    kernel = 2 * cutoff / rate * np.sinc(2 * cutoff / rate * taps)
    kernel *= np.hanning(len(taps))
    kernel /= kernel.sum()
    # The "same" span of the full convolution: np.convolve's own "same"
    # is the kernel's length when the signal is shorter.
    full = np.convolve(samples.astype(np.float64), kernel, mode="full")
    return full[half : half + len(samples)]


def resample(samples: np.ndarray, rate: int, target: int = ASR_SAMPLE_RATE):
    """samples, float32 mono at rate, at target (see the module's note on
    quality)."""
    if rate == target or not len(samples):
        return samples.astype(np.float32, copy=False)
    if rate > target:
        filtered = _low_pass(samples, rate, 0.45 * target)
        if rate % target == 0:
            return filtered[:: rate // target].astype(np.float32)
        source = filtered
    else:
        source = samples.astype(np.float64)
    count = max(1, round(len(source) * target / rate))
    positions = np.arange(count, dtype=np.float64) * (rate / target)
    positions = np.minimum(positions, len(source) - 1)
    return np.interp(positions, np.arange(len(source)), source).astype(np.float32)


def split_segments(
    samples: np.ndarray,
    rate: int = ASR_SAMPLE_RATE,
    *,
    request_seconds: float = MAX_REQUEST_SECONDS,
    search_seconds: float = SPLIT_SEARCH_SECONDS,
    window_seconds: float = SPLIT_WINDOW_SECONDS,
) -> list[Segment]:
    """The spans of samples each transcribed by one request: the whole audio
    when it fits, else spans of at most request_seconds, each cut at the
    quietest window_seconds within search_seconds before its boundary."""
    total = len(samples)
    limit = int(request_seconds * rate)
    if total <= limit:
        return [Segment(0, total)]
    search = max(1, int(search_seconds * rate))
    window = max(1, int(window_seconds * rate))
    segments = []
    start = 0
    while total - start > limit:
        boundary = start + limit
        low = max(start + window, boundary - search)
        candidates = samples[low : boundary - (boundary - low) % window]
        if len(candidates) >= window:
            energies = (candidates.astype(np.float64).reshape(-1, window) ** 2).mean(
                axis=1
            )
            cut = low + int(energies.argmin()) * window + window // 2
        else:
            cut = boundary
        segments.append(Segment(start, cut))
        start = cut
    segments.append(Segment(start, total))
    return segments


def parallel_segment_seconds(
    duration: float, lanes: int, minimum: float = PARALLEL_MINIMUM_SECONDS
) -> float:
    """The segment length that spreads `duration` seconds over at most
    `lanes` concurrent requests (each cut may come up to the split search
    early, which the length allows for), at least `minimum` and at most
    MAX_REQUEST_SECONDS; MAX_REQUEST_SECONDS when minimum is 0 (no parallel
    segments)."""
    if minimum <= 0 or lanes <= 1:
        return float(MAX_REQUEST_SECONDS)
    return float(
        min(MAX_REQUEST_SECONDS, max(minimum, duration / lanes + SPLIT_SEARCH_SECONDS))
    )


def wav_header(
    data_bytes: int | None,
    *,
    rate: int = TTS_SAMPLE_RATE,
    channels: int = 1,
    bits: int = 16,
) -> bytes:
    """The header of a PCM WAV file of data_bytes of samples, or of a stream
    of unknown length (None)."""
    block = channels * bits // 8
    if data_bytes is None:
        riff_size = data_size = UNKNOWN_LENGTH
    else:
        data_size = data_bytes
        # The RIFF chunk's size: the file's, less the chunk's own 8 bytes.
        riff_size = min(UNKNOWN_LENGTH, WAV_HEADER_BYTES - 8 + data_bytes)
    return (
        _RIFF.pack(b"RIFF", riff_size, b"WAVE")
        + _FMT.pack(b"fmt ", 16, 1, channels, rate, rate * block, block, bits)
        + _DATA.pack(b"data", data_size)
    )


def wav_bytes(pcm: bytes, *, rate: int = TTS_SAMPLE_RATE) -> bytes:
    """A complete 16-bit mono PCM WAV file of pcm's int16 samples."""
    return wav_header(len(pcm), rate=rate) + pcm


def float32_bytes(samples: np.ndarray) -> bytes:
    """samples as the transcription request carries them: little-endian
    float32."""
    return np.ascontiguousarray(samples, dtype="<f4").tobytes()


class SpeechEdges:
    """The ends of one speech answer's int16 PCM, smoothed as it streams.

    A continuation (reference_speech_id) starts where its reference's last
    syllable is still dying away: its first frames decode that tail, then a
    pause, then the new speech. Played after silence that tail is an abrupt
    burst, so a continuation that starts loud and falls quiet within
    ONSET_SECONDS is cut at the quiet point, with a short fade in. Any
    answer may stop on a sound still ringing; its last FADE_OUT_SECONDS
    fade out (held back until the answer ends)."""

    ONSET_SECONDS = 0.3
    # 20 ms windows every 10 ms; a start above LOUD_DB is a leaked tail,
    # which ends at the first window below QUIET_DB.
    WINDOW_SECONDS = 0.02
    HOP_SECONDS = 0.01
    LOUD_DB = -40.0
    QUIET_DB = -50.0
    FADE_IN_SECONDS = 0.005
    FADE_OUT_SECONDS = 0.03

    def __init__(self, continuation: bool, rate: int = TTS_SAMPLE_RATE):
        self.rate = rate
        self.onset = continuation
        self.held = np.zeros(0, dtype=np.int16)
        self.carry = b""
        self.tail = round(rate * self.FADE_OUT_SECONDS)

    def push(self, pcm: bytes) -> bytes:
        """What of the answer so far may be sent now."""
        data = self.carry + pcm
        whole = len(data) - len(data) % 2
        self.carry = data[whole:]
        samples = np.frombuffer(data[:whole], dtype="<i2")
        self.held = np.concatenate([self.held, samples])
        if self.onset:
            if len(self.held) < round(self.rate * self.ONSET_SECONDS):
                return b""
            self.held = self._trim_onset(self.held)
            self.onset = False
        ready = max(0, len(self.held) - self.tail)
        out, self.held = self.held[:ready], self.held[ready:]
        return out.astype("<i2").tobytes()

    def finish(self) -> bytes:
        """The rest of the answer, its end faded out."""
        samples = self.held
        if self.onset:
            samples = self._trim_onset(samples)
            self.onset = False
        self.held = np.zeros(0, dtype=np.int16)
        if not len(samples):
            return b""
        out = samples.astype(np.float32)
        count = min(len(out), self.tail)
        out[len(out) - count :] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
        return np.round(out).astype("<i2").tobytes()

    def _trim_onset(self, samples: np.ndarray) -> np.ndarray:
        window = round(self.rate * self.WINDOW_SECONDS)
        hop = round(self.rate * self.HOP_SECONDS)
        limit = min(len(samples), round(self.rate * self.ONSET_SECONDS))
        x = samples[:limit].astype(np.float32) / 32768.0

        def level(start):
            frame = x[start : start + window]
            return 20 * math.log10(math.sqrt(float(np.mean(frame * frame))) + 1e-9)

        if limit < window or level(0) <= self.LOUD_DB:
            return samples
        cut = next(
            (
                start
                for start in range(hop, limit - window + 1, hop)
                if level(start) < self.QUIET_DB
            ),
            None,
        )
        out = (
            samples[cut:].astype(np.float32)
            if cut is not None
            else samples.astype(np.float32)
        )
        count = min(len(out), round(self.rate * self.FADE_IN_SECONDS))
        out[:count] *= np.linspace(0.0, 1.0, count, dtype=np.float32)
        return np.round(out).astype(np.int16)
