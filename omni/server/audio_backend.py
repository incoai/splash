"""Audio request submission, cancellation, recovery and completion ownership.

NativeBackend (backend.py) owns the engine's recovery: its restarts after
a failure, the crash-loop stop, the status snapshots and the refusals
while it recovers; this backend keeps all of that and replaces what is
about an LLM request, the frames written and the events a job's reader
takes, with a speech or transcription request's.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from . import audio_protocol as audio_wire
from . import audio_runtime
from . import backend as llm_backend
from . import runtime as engine_runtime
from .diagnostics import print_status
from .errors import APIError
from .latency import RequestLatency


@dataclass
class AudioJob:
    """One speech or transcription request, from preparation to its answer.
    Its reader takes events from `events`: ("start", None), ("audio",
    pcm bytes) or ("tokens", token ids), then ("done", AudioResult) or
    ("error", APIError)."""

    request_id: int
    # request_id 0: AudioRuntime.submit stamps it.
    frame: audio_wire.SpeechRequest | audio_wire.TranscribeRequest
    deadline: float
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    latency: RequestLatency | None = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    # SPLASH_AUDIO_TRACE=1: named time.monotonic() instants of the request
    # (audio_server.py prints them; dev/tools/omni_request_trace.py joins
    # them with the engine's OMNI_ASR_TRACE lines).
    marks: dict | None = None

    @property
    def speech(self) -> bool:
        return isinstance(self.frame, audio_wire.SpeechRequest)


@dataclass
class AudioResult:
    """How a job ended: its finish reason and the engine's counts."""

    reason: str
    prompt_tokens: int
    # Frames of a speech request; tokens of a transcription.
    completion_count: int
    # The audio samples, or the tokens, its reader was given.
    streamed: int
    prefill_ms: float
    decode_ms: float
    wall_ms: float
    # From submission to the first audio chunk or token; None without one.
    first_output_ms: float | None
    # A speech request's frames (audio_wire.CODES_PER_FRAME codes each), when
    # it asked for them (audio_wire.SPEECH_FLAG_CODES).
    codes: tuple[int, ...] | None = None


@dataclass
class _JobState:
    job: AudioJob
    call: object | None = None
    terminal_enqueued: bool = False
    detached: bool = False
    shutdown_requested: bool = False
    streamed: int = 0
    submitted_at: float = field(default_factory=time.monotonic)
    first_output_at: float | None = None
    codes: tuple[int, ...] | None = None

    def detach(self):
        self.detached = True
        self.call = None


class AudioBackend(llm_backend.NativeBackend):
    """Write admitted audio requests directly to the engine; a finalizer
    thread delivers each job's terminal and logs it off the reader thread."""

    def submit(self, job: AudioJob):
        state = _JobState(job)
        request = audio_runtime.AudioRequest(job.frame, job.deadline)

        def on_event(call, event):
            with self.lock:
                if state.detached:
                    return
                if state.call is None:
                    state.call = call
                cancel = job.cancelled.is_set() or self.closing
            # A raise here is the call's callback error, which cancels it.
            self._on_event(state, event)
            if cancel:
                call.cancel()

        def on_complete(call):
            if job.marks is not None:
                job.marks["engine_id"] = call.request_id
                job.marks["done_read"] = time.monotonic()
            with self.lock:
                if state.detached or state.terminal_enqueued:
                    return
                if state.call is None:
                    state.call = call
                state.terminal_enqueued = True
                self.terminals.put((state, call))

        with self.lock:
            if self.closing:
                state.detach()
                job.events.put(
                    (
                        "error",
                        APIError(503, "server is shutting down", "server_shutdown"),
                    )
                )
                return
            self.active[job.request_id] = state
        try:
            if job.marks is not None:
                job.marks["submit"] = time.monotonic()
            call = self.runtime.submit(
                request, on_event=on_event, on_complete=on_complete
            )
            if job.marks is not None:
                job.marks["written"] = time.monotonic()
            with self.lock:
                if not state.detached:
                    state.call = call
                cancel = state.detached or self.closing or job.cancelled.is_set()
            if cancel:
                call.cancel()
        except Exception as error:
            with self.lock:
                deliver = not state.detached and not state.terminal_enqueued
                if deliver:
                    self._detach_locked(state)
            if deliver:
                refusal = None
                if isinstance(
                    error,
                    (
                        engine_runtime.EngineUnhealthy,
                        engine_runtime.ProtocolFatal,
                        engine_runtime.RuntimeClosed,
                    ),
                ):
                    # Not admitted: refused as a request arriving now would be.
                    refusal = self.refusal()
                job.events.put(("error", refusal or self._api_error(error)))

    def _on_event(self, state, event):
        job = state.job
        if isinstance(event, audio_wire.StartEvent):
            job.events.put(("start", None))
            return
        if isinstance(event, audio_wire.CodesEvent):
            # Delivered with the result.
            state.codes = event.codes
            return
        if isinstance(event, audio_wire.TokensEvent):
            count, item = len(event.tokens), ("tokens", event.tokens)
        elif isinstance(event, audio_wire.AudioEvent):
            count, item = event.sample_count, ("audio", event.pcm)
        else:
            return
        if state.first_output_at is None:
            state.first_output_at = time.monotonic()
            if job.marks is not None:
                job.marks["first_read"] = state.first_output_at
        if job.latency is not None and count:
            job.latency.tokens()
        state.streamed += count
        job.events.put(item)

    def _finalize(self, state, call):
        job = state.job
        error = None
        result = None
        try:
            done = call.result(0)
            if call.callback_error is not None:
                raise self._api_error(call.callback_error)
            result = AudioResult(
                reason=self._FINISH_NAMES[done.reason],
                prompt_tokens=done.prompt_tokens,
                completion_count=done.completion_tokens,
                streamed=state.streamed,
                prefill_ms=done.prefill_micros / 1000.0,
                decode_ms=done.decode_micros / 1000.0,
                wall_ms=done.wall_micros / 1000.0,
                first_output_ms=(
                    None
                    if state.first_output_at is None
                    else (state.first_output_at - state.submitted_at) * 1000.0
                ),
                codes=state.codes,
            )
        except (engine_runtime.EngineUnhealthy, engine_runtime.ProtocolFatal):
            # An admitted request ends with EngineUnhealthy or ProtocolFatal
            # only when the engine running it fails: it stopped or broke the
            # protocol. The listener has already decided whether that engine
            # restarts; the console names the failure.
            with self.lock:
                fatal_error = self.fatal_error
            if fatal_error is not None:
                error = APIError(500, fatal_error, "engine_failed")
            else:
                error = APIError(
                    503,
                    "the inference engine stopped unexpectedly and is restarting; "
                    "retry the request",
                    "runtime_unavailable",
                )
        except Exception as unexpected:
            error = self._api_error(unexpected)
        finally:
            with self.lock:
                shutdown_requested = state.shutdown_requested
                self._detach_locked(state)
        if shutdown_requested:
            result = None
            error = APIError(503, "server is shutting down", "server_shutdown")
        self._record(job, result=result, error=error)
        if job.marks is not None:
            job.marks["finalized"] = time.monotonic()
        job.events.put(("error", error) if error else ("done", result))

    @staticmethod
    def _api_error(error):
        if isinstance(error, engine_runtime.RequestFailed):
            code = error.code.decode("ascii", "replace")
            message = error.message_bytes.decode("utf-8", "replace")
            # The engine refuses what a request asks for as its own fault:
            # its text, its samples or its sampling.
            if not error.retryable and (
                code.startswith("invalid_") or code == "limit_exceeded"
            ):
                return APIError(400, message, code)
        return llm_backend.NativeBackend._api_error(error)

    def _record(self, job, result=None, error=None):
        record = {
            "outcome": result.reason if result else "error",
            "kind": "speech" if job.speech else "transcription",
        }
        if job.speech:
            record["text_tokens"] = len(job.frame.text_tokens)
        else:
            record["seconds"] = (
                job.frame.sample_count / audio_wire.TRANSCRIBE_SAMPLE_RATE
            )
        if result:
            record["prompt_tokens"] = result.prompt_tokens
            record["completion"] = result.completion_count
            record["streamed"] = result.streamed
            record["wall_ms"] = result.wall_ms
            record["first_output_ms"] = result.first_output_ms
        if error:
            record["error_code"] = error.code
        try:
            self.request_logger(record)
        except Exception:
            pass


def print_audio_request(record):
    """One console line per finished audio request."""
    outcome = record["outcome"]
    if outcome == "error":
        print_status(f"Error · {record.get('error_code', 'runtime_error')}", error=True)
        return
    parts = ["Cancelled" if outcome == "cancelled" else "Done"]
    if record["kind"] == "speech":
        seconds = record.get("streamed", 0) / audio_wire.AUDIO_SAMPLE_RATE
        parts.append(f"text {record.get('text_tokens', 0):,} tokens")
        parts.append(f"audio {seconds:.1f}s")
        wall = record.get("wall_ms") or 0.0
        if wall > 0 and seconds > 0:
            parts.append(f"{seconds * 1000.0 / wall:.1f}x realtime")
    else:
        parts.append(f"audio {record.get('seconds', 0.0):.1f}s")
        parts.append(f"output {record.get('completion', 0):,} tokens")
    first = record.get("first_output_ms")
    if first is not None:
        parts.append(f"first output {first / 1000:.2f}s")
    print_status(" · ".join(parts))
