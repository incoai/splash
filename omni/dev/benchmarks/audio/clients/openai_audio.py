"""HTTP client for OpenAI-compatible audio endpoints (Splash Omni, sglang-omni,
mlx-audio server, ...), with benchmark timing.

TTS (``POST /v1/audio/speech``): time to first audio byte, total time, audio
seconds decoded from the response bytes (wav header parsed by hand so that
streamed WAVs with placeholder sizes still decode; pcm = int16 mono).
ASR (``POST /v1/audio/transcriptions``): request latency and transcript.

Stdlib only.
"""

from __future__ import annotations

import http.client
import json
import struct
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

WAV_HEADER_MIN = 44


@dataclass
class TTSResult:
    ttfa_s: float | None = None  # time to first audio byte (payload, not header)
    total_s: float | None = None  # request start -> last byte
    audio_s: float = 0.0  # decoded audio duration
    sample_rate: int = 24000
    num_bytes: int = 0
    num_chunks: int = 0
    audio_path: str | None = None
    status: int | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float | None:
        return self.total_s / self.audio_s if (self.total_s and self.audio_s) else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rtf"] = self.rtf
        return d


@dataclass
class ASRResult:
    latency_s: float | None = None
    text: str = ""
    language: str | None = None
    status: int | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_wav(data: bytes) -> tuple[int, int, int, int]:
    """Return (sample_rate, channels, bits, payload_bytes) from a WAV byte string.
    Tolerates streamed headers whose RIFF/data sizes are 0 or 0xFFFFFFFF."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    pos, sr, ch, bits = 12, None, None, None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        if cid == b"fmt ":
            _fmt, ch, sr = struct.unpack("<HHI", data[pos + 8 : pos + 16])
            bits = struct.unpack("<H", data[pos + 22 : pos + 24])[0]
            pos += 8 + size + (size & 1)
        elif cid == b"data":
            payload = len(data) - (pos + 8)
            if sr is None:
                raise ValueError("data chunk before fmt chunk")
            return sr, ch, bits, payload
        else:
            pos += 8 + size + (size & 1)
    raise ValueError("no data chunk")


def audio_seconds(
    data: bytes, response_format: str, sample_rate: int = 24000
) -> tuple[float, int]:
    if response_format == "pcm":
        return len(data) / 2 / sample_rate, sample_rate
    sr, ch, bits, payload = parse_wav(data)
    return payload / (bits // 8) / ch / sr, sr


def write_wav_pcm16(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
    )
    path.write_bytes(header + pcm)


class OpenAIAudioClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 600.0,
        api_key: str | None = None,
    ):
        u = urlsplit(base_url)
        if u.scheme not in ("http", ""):
            raise ValueError("only http:// base urls are supported")
        self.host, self.port = u.hostname or "127.0.0.1", u.port or 80
        self.prefix = u.path.rstrip("/")
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def get_json(self, path: str):
        c = self._conn()
        try:
            c.request("GET", self.prefix + path, headers=self.headers)
            r = c.getresponse()
            body = r.read()
            return r.status, (json.loads(body) if body else None)
        finally:
            c.close()

    def healthy(self) -> bool:
        for path in ("/health", "/ready", "/v1/models"):
            try:
                status, _ = self.get_json(path)
                if status == 200:
                    return True
            except (OSError, ValueError):
                return False
        return False

    # ----- TTS -----
    def speech(
        self,
        model: str,
        text: str,
        voice: str = "default",
        response_format: str = "pcm",
        stream: bool = True,
        sample_rate: int = 24000,
        save_path: Path | str | None = None,
        extra: dict | None = None,
        chunk_bytes: int = 4096,
    ) -> TTSResult:
        body = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "stream": stream,
        }
        if extra:
            body.update(extra)
        payload = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", **self.headers}
        res = TTSResult(sample_rate=sample_rate)
        header_len = 0 if response_format == "pcm" else WAV_HEADER_MIN
        chunks: list[bytes] = []
        total = 0
        c = self._conn()
        t0 = time.perf_counter()
        try:
            c.request(
                "POST", self.prefix + "/v1/audio/speech", body=payload, headers=headers
            )
            r = c.getresponse()
            res.status = r.status
            while True:
                # read1 returns what has arrived (up to chunk_bytes); read
                # would block until chunk_bytes arrive, timing a small first
                # chunk at the second one.
                chunk = r.read1(chunk_bytes)
                if not chunk:
                    break
                now = time.perf_counter()
                total += len(chunk)
                chunks.append(chunk)
                res.num_chunks += 1
                if res.ttfa_s is None and total > header_len:
                    res.ttfa_s = now - t0
            res.total_s = time.perf_counter() - t0
        except (OSError, http.client.HTTPException) as e:
            res.error = f"{type(e).__name__}: {e}"
            res.total_s = time.perf_counter() - t0
            return res
        finally:
            c.close()
        data = b"".join(chunks)
        res.num_bytes = len(data)
        if res.status != 200:
            res.error = f"HTTP {res.status}: {data[:300]!r}"
            return res
        try:
            res.audio_s, res.sample_rate = audio_seconds(
                data, response_format, sample_rate
            )
        except ValueError as e:
            res.error = f"undecodable audio: {e}"
            return res
        if save_path is not None:
            save_path = Path(save_path)
            if response_format == "pcm":
                write_wav_pcm16(save_path, data, sample_rate)
            else:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(data)
            res.audio_path = str(save_path)
        return res

    # ----- ASR -----
    def transcribe(
        self,
        model: str,
        wav_path: Path | str,
        language: str | None = None,
        response_format: str = "json",
        extra: dict | None = None,
    ) -> ASRResult:
        wav_path = Path(wav_path)
        boundary = "----omnibench" + uuid.uuid4().hex
        fields = {"model": model, "response_format": response_format}
        if language:
            fields["language"] = language
        if extra:
            fields.update({k: str(v) for k, v in extra.items()})
        parts = []
        for k, v in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            )
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{wav_path.name}"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode()
        )
        parts.append(wav_path.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        payload = b"".join(parts)
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            **self.headers,
        }
        res = ASRResult()
        c = self._conn()
        t0 = time.perf_counter()
        try:
            c.request(
                "POST",
                self.prefix + "/v1/audio/transcriptions",
                body=payload,
                headers=headers,
            )
            r = c.getresponse()
            res.status = r.status
            body = r.read()
            res.latency_s = time.perf_counter() - t0
        except (OSError, http.client.HTTPException) as e:
            res.error = f"{type(e).__name__}: {e}"
            res.latency_s = time.perf_counter() - t0
            return res
        finally:
            c.close()
        if res.status != 200:
            res.error = f"HTTP {res.status}: {body[:300]!r}"
            return res
        if response_format == "text":
            res.text = body.decode("utf-8", "replace").strip()
        else:
            try:
                obj = json.loads(body)
            except ValueError:
                res.error = f"bad JSON: {body[:200]!r}"
                return res
            res.text = (obj.get("text") or "").strip()
            res.language = obj.get("language")
            if "usage" in obj:
                res.extra["usage"] = obj["usage"]
        return res
