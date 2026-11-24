"""The subprocess client of the audio engine, `splash serve-omni`.

MultiplexedRuntime (runtime.py) owns the process, its reader thread, the
multiplexing of requests and events by request id, cancellation, the
liveness probes, crash handling and relaunches; this module gives it the
version-11 codec (audio_protocol.py) and the checks a speech or transcription
request holds the engine's events to.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import audio_protocol as audio_wire
from . import runtime as engine_runtime

# The failures callers catch, as runtime.py raises them.
EngineRuntimeError = engine_runtime.EngineRuntimeError
EngineUnhealthy = engine_runtime.EngineUnhealthy
PendingLimitExceeded = engine_runtime.PendingLimitExceeded
ProtocolFatal = engine_runtime.ProtocolFatal
RequestFailed = engine_runtime.RequestFailed
RuntimeClosed = engine_runtime.RuntimeClosed


@dataclass(slots=True, frozen=True)
class AudioRequest:
    """One native request: its wire frame, without the request id
    AudioRuntime.submit fills in, and its time.monotonic() deadline, math.inf
    for none."""

    frame: audio_wire.SpeechRequest | audio_wire.TranscribeRequest
    deadline: float

    @property
    def speech(self) -> bool:
        return isinstance(self.frame, audio_wire.SpeechRequest)


class AudioCall(engine_runtime.RuntimeCall):
    """Future-like handle for one admitted speech or transcription request.

    A speech request's output is its audio chunks, in order, then, when it
    asked for them, its codes (as many frames as Done counts), then Done,
    whose completion count is its frames; a transcription's is its tokens,
    then Done, whose completion count is their number. Neither request's
    prompt count is known to the server: the engine adds the role tokens
    and the codec prefix to a text, and the audio embeddings to a
    transcription, so Done's prompt count is reported, never checked.
    """

    def _expect(self, request) -> None:
        frame = request.frame
        self._speech = isinstance(frame, audio_wire.SpeechRequest)
        if self._speech:
            self._max_frames = frame.max_frames or audio_wire.DEFAULT_MAX_FRAMES
            self._codes_wanted = bool(frame.flags & audio_wire.SPEECH_FLAG_CODES)
        else:
            self._logical_max = frame.max_tokens or audio_wire.DEFAULT_MAX_TOKENS
            self._codes_wanted = False
        self._next_sample_offset = 0
        self._codes_frames = None

    @property
    def speech(self) -> bool:
        return self._speech

    def _record_tokens(self, event) -> str | None:
        if self._speech:
            return "TokensEvent arrived for a speech request"
        return super()._record_tokens(event)

    def _record_audio(self, event: audio_wire.AudioEvent) -> str | None:
        with self._lock:
            if not self._speech:
                return "AudioEvent arrived for a transcription request"
            if self._event.is_set():
                return "AudioEvent arrived after the request became terminal"
            if self._start is None:
                return "AudioEvent arrived before StartEvent"
            if event.sample_offset != self._next_sample_offset:
                relation = (
                    "overlaps or duplicates prior samples"
                    if event.sample_offset < self._next_sample_offset
                    else "leaves a gap in the audio stream"
                )
                return (
                    f"AudioEvent offset {event.sample_offset} {relation}; "
                    f"expected {self._next_sample_offset}"
                )
            self._next_sample_offset += event.sample_count
        self._emit(event)
        return None

    def _record_codes(self, event: audio_wire.CodesEvent) -> str | None:
        with self._lock:
            if not self._codes_wanted:
                return "CodesEvent arrived for a request that did not ask for codes"
            if self._event.is_set():
                return "CodesEvent arrived after the request became terminal"
            if self._start is None:
                return "CodesEvent arrived before StartEvent"
            if self._codes_frames is not None:
                return "duplicate CodesEvent"
            self._codes_frames = event.frame_count
        self._emit(event)
        return None

    def _check_done(self, done):
        with self._lock:
            streamed = (
                self._next_sample_offset if self._speech else self._next_token_offset
            )
            if self._start is None and not (
                self._cancel_requested
                and done.reason is audio_wire.FinishReason.CANCELLED
                and not streamed
            ):
                raise engine_runtime.ProtocolFatal(
                    "DoneEvent arrived before StartEvent"
                )
            if self._speech:
                if done.completion_tokens > self._max_frames:
                    raise engine_runtime.ProtocolFatal(
                        f"DoneEvent frame count {done.completion_tokens} exceeds "
                        f"the request's maximum {self._max_frames}"
                    )
                if (
                    self._codes_frames is not None
                    and self._codes_frames != done.completion_tokens
                ):
                    raise engine_runtime.ProtocolFatal(
                        f"CodesEvent holds {self._codes_frames} frames; DoneEvent "
                        f"counts {done.completion_tokens}"
                    )
            else:
                if done.completion_tokens != streamed:
                    raise engine_runtime.ProtocolFatal(
                        f"DoneEvent completion count {done.completion_tokens} does "
                        f"not match streamed count {streamed}"
                    )
                if (
                    done.reason is audio_wire.FinishReason.LENGTH
                    and streamed != self._logical_max
                ):
                    raise engine_runtime.ProtocolFatal(
                        f"length-finished DoneEvent has {streamed} tokens; expected "
                        f"the maximum {self._logical_max}"
                    )
            return done


class AudioRuntime(engine_runtime.MultiplexedRuntime):
    """One-reader, direct-admission client of the audio engine."""

    _wire = audio_wire
    _call_type = AudioCall
    _ready_changed = (
        "native context window, concurrency or model kind changed; restart the "
        "Splash server"
    )

    def _ready_signature(self, ready) -> tuple:
        return (ready.max_context_tokens, ready.max_concurrent_requests, ready.kind)

    def _dispatch_event(self, generation: int, message) -> None:
        if isinstance(message, audio_wire.AudioEvent):
            call = self._require_call(generation, message.request_id)
            if issue := call._record_audio(message):
                raise engine_runtime.ProtocolFatal(issue)
            return
        if isinstance(message, audio_wire.CodesEvent):
            call = self._require_call(generation, message.request_id)
            if issue := call._record_codes(message):
                raise engine_runtime.ProtocolFatal(issue)
            return
        super()._dispatch_event(generation, message)


def engine_command(
    binary: str,
    model_root: str,
    *,
    max_memory: int | None = None,
    lanes: int | None = None,
    idle_release: str | None = None,
    disable_ane: bool = False,
    draft_model: str | None = None,
    draft_tokens: int | None = None,
    speculative_lanes: int | None = None,
) -> list[str]:
    """The audio engine's command line: `BINARY serve-omni MODEL_ROOT` with
    the options given."""
    command = [binary, "serve-omni", model_root]
    if max_memory is not None:
        command.extend(("--max-memory", str(max_memory)))
    if lanes is not None:
        command.extend(("--lanes", str(lanes)))
    if idle_release is not None:
        command.extend(("--idle-release", idle_release))
    if disable_ane:
        command.append("--disable-ane")
    if draft_model is not None:
        command.extend(("--draft-model", draft_model))
    if draft_tokens is not None:
        command.extend(("--draft-tokens", str(draft_tokens)))
    if speculative_lanes is not None:
        command.extend(("--speculative-lanes", str(speculative_lanes)))
    return command
