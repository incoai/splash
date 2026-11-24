"""What the audio server tests share: the fake audio engine, a FakeRuntime
running audio plans; the fake tokenizer; model profiles; uploads and
request bodies; and the harness that serves the audio frontend."""

import http.client
import io
import json
import re
import secrets
import struct
import threading
import unittest
import wave
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dev.tests.server_fixtures import AUDIO_READY, FakeRuntime
from server import (
    audio_backend,
    audio_frontend,
    audio_server,
    http_server,
    serve_options,
)
from server import audio_protocol as audio_wire
from server.origins import parse_allowed_origin

ASR_READY = audio_wire.ReadyEvent(8, 8192, audio_wire.ModelKind.ASR)
TTS_LANGUAGES = {
    "chinese": 2055,
    "english": 2050,
    "german": 2053,
    "italian": 2070,
    "portuguese": 2071,
    "spanish": 2054,
    "japanese": 2058,
    "korean": 2064,
    "french": 2061,
    "russian": 2069,
}
ASR_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "German",
    "French",
    "Spanish",
    "Japanese",
    "Korean",
)
TTS_PROFILE = audio_frontend.ModelProfile(
    audio_frontend.TTS_FAMILY, TTS_LANGUAGES, dict(audio_frontend.TTS_DEFAULTS)
)
# The VoiceDesign and CustomVoice checkpoints' profiles (their configs'
# spk_id and spk_is_dialect, two of the nine speakers).
VOICE_DESIGN_PROFILE = audio_frontend.ModelProfile(
    audio_frontend.TTS_FAMILY,
    TTS_LANGUAGES,
    dict(audio_frontend.TTS_DEFAULTS),
    audio_frontend.VOICE_DESIGN,
)
CUSTOM_VOICE_PROFILE = audio_frontend.ModelProfile(
    audio_frontend.TTS_FAMILY,
    TTS_LANGUAGES,
    dict(audio_frontend.TTS_DEFAULTS),
    audio_frontend.CUSTOM_VOICE,
    {"vivian": 3065, "dylan": 2878},
    {"dylan": 2074},
)
ASR_PROFILE = audio_frontend.ModelProfile(audio_frontend.ASR_FAMILY, ASR_LANGUAGES)
# The model id requests name.
MODEL = "test-model"


class FakeAudioTokenizer:
    """One token per word, space run, punctuation mark or special token, so
    decoding reproduces the text; the special tokens keep Qwen's ids."""

    SPECIAL = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|audio_start|>": 151669,
        "<|audio_end|>": 151670,
    }
    _PIECES = re.compile(r"<\|[a-z_]+\|>|<asr_text>|\s+|\w+|[^\w\s]")

    def __init__(self):
        self.ids = dict(self.SPECIAL)
        self.pieces = {token: piece for piece, token in self.SPECIAL.items()}
        self.calls = []

    def _id(self, piece):
        token = self.ids.get(piece)
        if token is None:
            token = 1000 + len(self.ids)
            self.ids[piece] = token
            self.pieces[token] = piece
        return token

    def encode_text(self, text):
        return [self._id(piece) for piece in self._PIECES.findall(text)]

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": self.encode_text(text)}

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return "".join(
            self.pieces[int(token)]
            for token in token_ids
            if not (skip_special_tokens and self.pieces[int(token)] in self.SPECIAL)
        )


def pcm(samples):
    """int16 PCM bytes of the sample values."""
    return struct.pack(f"<{len(samples)}h", *samples)


def pcm_values(data):
    return struct.unpack(f"<{len(data) // 2}h", data)


class AudioPlan:
    """What the fake engine does with one request: emits `chunks` (int16
    PCM bytes) or `tokens` (batches of ids), then Done with `reason`. With
    `stepped`, it waits for `step` before each chunk or batch after the
    first; with `block`, for `release` before any output."""

    def __init__(
        self,
        chunks=(),
        tokens=(),
        reason="stop",
        block=False,
        exception=None,
        delay=0,
        stepped=False,
        prompt_tokens=20,
        frames=None,
        before_start=False,
        hold_done=False,
        codes=None,
    ):
        self.chunks = list(chunks)
        # A speech request asking for its codes gets these before Done, or
        # by default one frame of codes per frame Done counts.
        self.codes = codes
        self.tokens = [tuple(batch) for batch in tokens]
        self.reason = reason
        self.exception = exception
        self.delay = delay
        self.stepped = stepped
        self.prompt_tokens = prompt_tokens
        self.frames = frames
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.start_release = threading.Event()
        if not before_start:
            self.start_release.set()
        self.step = threading.Event()
        self.cancelled = threading.Event()
        self.terminal = threading.Event()
        # With hold_done, Done waits for terminal_release after the output.
        self.terminal_release = threading.Event()
        if not hold_done:
            self.terminal_release.set()


class FakeAudioRuntime(FakeRuntime):
    """AudioRuntime's stand-in: each call runs the next of `plans`, or one
    chunk of silence once they run out."""

    def __init__(self, *plans, manual=False, kind=audio_wire.ModelKind.TTS):
        super().__init__(*plans, manual=manual)
        self.readiness = AUDIO_READY if kind is audio_wire.ModelKind.TTS else ASR_READY

    @staticmethod
    def default_plan(request):
        if request.speech:
            return AudioPlan(chunks=[pcm([0] * audio_wire.SAMPLES_PER_FRAME)])
        # <|im_end|>, which every tokenizer knows and skips.
        return AudioPlan(tokens=[(151645,)])

    def _run(self, call):
        plan = call.plan
        plan.start_release.wait(2)
        if call.done:
            return
        call.emit(audio_wire.StartEvent(call.request_id, 0, 0))
        plan.started.set()
        plan.release.wait(2)
        if plan.exception:
            call.complete(error=plan.exception)
            return
        offset = 0
        completion = 0
        if not plan.cancelled.is_set():
            for index, chunk in enumerate(plan.chunks):
                if plan.cancelled.is_set():
                    break
                if index and plan.stepped:
                    plan.step.wait(2)
                    plan.step.clear()
                call.emit(audio_wire.AudioEvent(call.request_id, offset, chunk))
                offset += len(chunk) // 2
                plan.cancelled.wait(plan.delay)
            for index, batch in enumerate(plan.tokens):
                if plan.cancelled.is_set():
                    break
                if index and plan.stepped:
                    plan.step.wait(2)
                    plan.step.clear()
                call.emit(audio_wire.TokensEvent(call.request_id, completion, batch))
                completion += len(batch)
                plan.cancelled.wait(plan.delay)
        reason = (
            audio_wire.FinishReason.CANCELLED
            if plan.cancelled.is_set()
            else {
                "stop": audio_wire.FinishReason.STOP,
                "length": audio_wire.FinishReason.LENGTH,
            }[plan.reason]
        )
        plan.terminal.set()
        plan.terminal_release.wait(2)
        if call.request.speech:
            completion = (
                plan.frames
                if plan.frames is not None
                else -(-offset // audio_wire.SAMPLES_PER_FRAME)
            )
        if reason is audio_wire.FinishReason.CANCELLED:
            completion = 0
        elif call.request.speech and (
            call.request.frame.flags & audio_wire.SPEECH_FLAG_CODES
        ):
            codes = plan.codes
            if codes is None:
                codes = tuple(
                    index % 2048
                    for index in range(completion * audio_wire.CODES_PER_FRAME)
                )
            call.emit(audio_wire.CodesEvent(call.request_id, tuple(codes)))
        done = audio_wire.DoneEvent(
            call.request_id,
            reason,
            plan.prompt_tokens,
            completion,
            1_000,
            2_000,
            3_000,
            (),
        )
        call.complete(result=done)


class AudioHarness:
    def __init__(
        self,
        runtime,
        tokenizer=None,
        profile=TTS_PROFILE,
        queue_size=4,
        timeout=2,
        model=MODEL,
        request_logger=lambda _record: None,
        api_key=None,
        max_request_bytes=serve_options.DEFAULT_MAX_REQUEST_BYTES,
        host="127.0.0.1",
        allowed_hosts=(),
        allowed_origins=(),
        io_timeout=None,
        **frontend_options,
    ):
        self.tokenizer = tokenizer or FakeAudioTokenizer()
        runtime.pending_limit = queue_size
        self.backend = audio_backend.AudioBackend(runtime, request_logger)
        self.app = audio_frontend.AudioFrontend(
            self.tokenizer,
            self.backend,
            model,
            profile,
            timeout,
            2,
            **frontend_options,
        )
        self.tokenizer.calls.clear()
        self.server = audio_server.AudioServer(
            (host, 0),
            self.app,
            request_capacity=queue_size,
            api_key=api_key,
            max_request_bytes=max_request_bytes,
            allowed_hosts=allowed_hosts,
            allowed_origins=map(parse_allowed_origin, allowed_origins),
        )
        # The server reads HTTP_IO_TIMEOUT per connection. A harness given its
        # own holds it until close(); patching only once setup has succeeded
        # never leaves it patched.
        self._io_timeout = None
        if io_timeout is not None:
            self._io_timeout = mock.patch.object(
                http_server, "HTTP_IO_TIMEOUT", io_timeout
            )
            self._io_timeout.start()
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def request(self, method, path, body=None, headers=None, data=None):
        """(status, headers, payload) of one request; `body` is sent as
        JSON, `data` as given."""
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        if headers is None:
            headers = {"Content-Type": "application/json"} if body is not None else {}
        if data is None and body is not None:
            data = json.dumps(body)
        connection.request(method, path, data, headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, dict(response.getheaders()), payload

    def raw_post(self, payload, length, path="/v1/audio/speech"):
        """(status, payload) of a speech request whose Content-Length states
        `length`, whatever `payload` holds."""
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(length))
        connection.endheaders(payload)
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response.status, body

    def open_stream(self, path, body):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        connection.request(
            "POST", path, json.dumps(body), {"Content-Type": "application/json"}
        )
        return connection, connection.getresponse()

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.backend.close()
        if self._io_timeout is not None:
            self._io_timeout.stop()


class AudioHarnessTestCase(unittest.TestCase):
    """A test case whose harnesses close when its test ends."""

    def harness(self, runtime=None, **kwargs):
        harness = AudioHarness(runtime or FakeAudioRuntime(), **kwargs)
        self.addCleanup(harness.close)
        return harness

    def asr_harness(self, *plans, **kwargs):
        return self.harness(
            FakeAudioRuntime(*plans, kind=audio_wire.ModelKind.ASR),
            profile=ASR_PROFILE,
            **kwargs,
        )


def speech_body(**extra):
    body = {"model": MODEL, "input": "Hello, world!"}
    body.update(extra)
    return body


def wav_upload(samples, rate=16_000, channels=1, width=2):
    """A PCM WAV of float samples in [-1, 1], each channel the same."""
    values = np.asarray(samples, dtype=np.float64)
    if width == 1:
        frames = (np.clip(values * 128 + 128, 0, 255)).astype(np.uint8)
        frames = np.repeat(frames, channels).tobytes()
    elif width == 2:
        frames = np.clip(values * 32767, -32768, 32767).astype("<i2")
        frames = np.repeat(frames, channels).tobytes()
    elif width == 3:
        ints = np.clip(values * (2**23 - 1), -(2**23), 2**23 - 1).astype(np.int32)
        ints = np.repeat(ints, channels)
        frames = b"".join(int(v).to_bytes(3, "little", signed=True) for v in ints)
    else:
        frames = np.clip(values * (2**31 - 1), -(2**31), 2**31 - 1).astype("<i4")
        frames = np.repeat(frames, channels).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return buffer.getvalue()


def sine(seconds, rate=16_000, frequency=440.0, amplitude=0.5):
    times = np.arange(int(seconds * rate)) / rate
    return amplitude * np.sin(2 * np.pi * frequency * times)


def multipart(fields=None, files=None):
    """A multipart/form-data body and its Content-Type: fields by name, and
    files by name as (filename, bytes, content type)."""
    boundary = "----splash" + secrets.token_hex(8)
    parts = []
    for name, value in (fields or {}).items():
        parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}'.encode()
        )
    for name, (filename, data, content_type) in (files or {}).items():
        parts.append(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'
            ).encode()
            + data
        )
    body = b"".join(
        b"--" + boundary.encode() + b"\r\n" + part + b"\r\n" for part in parts
    )
    body += b"--" + boundary.encode() + b"--\r\n"
    return body, f"multipart/form-data; boundary={boundary}"


def transcription_upload(samples=None, rate=16_000, fields=None, filename="a.wav"):
    """A transcription request's body and headers for a WAV of samples."""
    data = wav_upload(sine(0.5) if samples is None else samples, rate)
    body, content_type = multipart(
        {"model": MODEL, **(fields or {})}, {"file": (filename, data, "audio/wav")}
    )
    return body, {"Content-Type": content_type}


def audio_main_args(**overrides):
    """Parsed command-line arguments for audio_server.main() tests."""
    return SimpleNamespace(
        **{
            "model_root": "model",
            "tokenizer": "tokenizer",
            "model": MODEL,
            "served_model_name": [],
            "announce_served_name": False,
            "max_memory": None,
            "idle_release": None,
            "lanes": None,
            "disable_ane": False,
            "segment_seconds": 30.0,
            "draft_model": None,
            "no_draft_model": False,
            "draft_tokens": None,
            "speculative_lanes": None,
            "request_timeout": None,
            "queue_size": 1,
            "host": "127.0.0.1",
            "allowed_host": [],
            "allowed_origin": [],
            "api_key": None,
            "max_request_size": serve_options.DEFAULT_MAX_REQUEST_BYTES,
            "port": 0,
            "binary": "splash",
            **overrides,
        }
    )
