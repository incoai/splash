"""The server's side of the Splash Omni native protocol, version 11.

It encodes the audio client frames and decodes the engine's events;
``runtime/engine/Protocol.*`` does the reverse, and docs/omni-design.md,
"Protocol (version 11)", is the contract. The framing, the issue codes and
the frames both protocols keep as they are (Cancel, StatusRequest, Start,
Tokens, Done, Error, StatusJson) are protocol.py's, version 8's codec, whose
decoders this module reuses; only the version in each header differs.
"""

from __future__ import annotations

import array
import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

from . import protocol as wire
from .protocol import (
    _FLOAT32_MIN,
    _HEADER,
    _MAGIC,
    _allocation_issue,
    _bytes,
    _fail,
    _fail_request,
    _fatal,
    _float32,
    _issue,
    _tail,
    _u32,
    _u64,
    _words,
)

# The frames, issues and names both protocols share.
CancelFrame = wire.CancelFrame
StatusRequestFrame = wire.StatusRequestFrame
StartEvent = wire.StartEvent
TokensEvent = wire.TokensEvent
DoneEvent = wire.DoneEvent
ErrorEvent = wire.ErrorEvent
StatusJsonEvent = wire.StatusJsonEvent
FinishReason = wire.FinishReason
FailureClass = wire.FailureClass
IssueCode = wire.IssueCode
ProtocolIssue = wire.ProtocolIssue
ProtocolError = wire.ProtocolError
Frame = wire.Frame
ParseStep = wire.ParseStep
frame_type_name = wire.frame_type_name
failure_class_name = wire.failure_class_name
issue_code_name = wire.issue_code_name

PROTOCOL_VERSION = 11
FRAME_HEADER_BYTES = wire.FRAME_HEADER_BYTES
STATUS_SCHEMA_VERSION = wire.STATUS_SCHEMA_VERSION
MAX_FRAME_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_STATUS_JSON_BYTES = wire.MAX_STATUS_JSON_BYTES
MAX_ERROR_STRING_BYTES = wire.MAX_ERROR_STRING_BYTES
MAX_TOKEN_BATCH = wire.MAX_TOKEN_BATCH
# The plain text's tokens, at most; the engine adds the role tokens and the
# codec prefix. Reference text is bounded the same way.
MAX_TEXT_TOKENS = 4096
# A VoiceDesign or CustomVoice request's instruction: its whole user turn,
# "<|im_start|>user\n{instruction}<|im_end|>\n", tokenized.
MAX_INSTRUCTION_TOKENS = 512
# What a transcription request carries: float32 samples at 16 kHz, mono, at
# most 20 minutes; the server splits longer audio.
TRANSCRIBE_SAMPLE_RATE = 16_000
MAX_TRANSCRIBE_SECONDS = 20 * 60
MAX_SAMPLES = TRANSCRIBE_SAMPLE_RATE * MAX_TRANSCRIBE_SECONDS
# What the engine emits for speech: int16 samples at 24 kHz, mono, 12.5
# codec frames per second.
AUDIO_SAMPLE_RATE = 24_000
FRAMES_PER_SECOND = 12.5
SAMPLES_PER_FRAME = 1920
CODES_PER_FRAME = 16
SPEAKER_EMBEDDING_DIMS = 2048
# SpeechRequest flags: a CodesEvent with the request's frames before its
# Done. The other bits are reserved and must be 0.
SPEECH_FLAG_CODES = 1
# The engine's defaults, which a zero in the request selects.
DEFAULT_MAX_FRAMES = 8192
DEFAULT_CHUNK_FRAMES = 25
DEFAULT_MAX_TOKENS = 4096

_SPEECH = struct.Struct("<QIIfIfffIfQIIIIIIII")
_TRANSCRIBE = struct.Struct("<QIIIII")
_READY = struct.Struct("<IIB")
_AUDIO = struct.Struct("<QII")
_CODES = struct.Struct("<QI")
_START = struct.Struct("<QII")
_TOKENS = struct.Struct("<QII")
_DONE = struct.Struct("<QBIIQQQI")
_ERROR = struct.Struct("<BBQII")
_STATUS_JSON = struct.Struct("<Q")
_CANCEL = struct.Struct("<Q")
_STATUS_REQUEST = struct.Struct("<Q")

assert array.array("h").itemsize == 2
assert _SPEECH.size == 84
assert _TRANSCRIBE.size == 28
assert _READY.size == 9
assert _AUDIO.size == 16
assert _CODES.size == 12

SPEECH_FIXED_BYTES = _SPEECH.size
TRANSCRIBE_FIXED_BYTES = _TRANSCRIBE.size


class FrameType(IntEnum):
    CANCEL = 0x0002
    STATUS_REQUEST = 0x0004
    SPEECH_REQUEST = 0x0011
    TRANSCRIBE_REQUEST = 0x0012

    READY = 0x0100
    START = 0x0101
    TOKENS = 0x0102
    DONE = 0x0104
    ERROR = 0x0105
    STATUS_JSON = 0x0107
    AUDIO = 0x0110
    CODES = 0x0111


# The frame types the engine sends; the others are the server's.
_EVENT_TYPES = frozenset(
    {
        FrameType.READY,
        FrameType.START,
        FrameType.TOKENS,
        FrameType.DONE,
        FrameType.ERROR,
        FrameType.STATUS_JSON,
        FrameType.AUDIO,
        FrameType.CODES,
    }
)
# The events whose layout version 8 defines, decoded by its codec.
_SHARED_EVENT_TYPES = {
    FrameType.START: wire.FrameType.START,
    FrameType.TOKENS: wire.FrameType.TOKENS,
    FrameType.DONE: wire.FrameType.DONE,
    FrameType.ERROR: wire.FrameType.ERROR,
    FrameType.STATUS_JSON: wire.FrameType.STATUS_JSON,
}


class ModelKind(IntEnum):
    """What the engine serves, as its Ready event names it."""

    TTS = 1
    ASR = 2


@dataclass(slots=True, frozen=True)
class SpeechRequest:
    """Speech for a text: its tokens, the plain text tokenized without
    special tokens (the engine adds the role tokens and the codec prefix),
    the codec language id (0 selects automatic), the talker's and the code
    predictor's sampling (a temperature of 0 is greedy; a top_k of 0 keeps
    every code), the frame budget and the chunking of the audio stream (0
    selects the engine's defaults), and the optional reference: its text
    tokens, its codes (CODES_PER_FRAME per frame) and the speaker embedding
    (SPEAKER_EMBEDDING_DIMS values, or none). A reference's text and codes
    make the request a continuation of that speech (docs/omni-design.md,
    "Reference continuation"); the engine refuses a speaker embedding. A
    VoiceDesign or CustomVoice model takes the instruction's user turn
    tokenized whole (instruction_tokens), which leads the prompt, and a
    CustomVoice model its preset speaker's codec id (preset_speaker, 0 for
    none). flags SPEECH_FLAG_CODES asks for the generated frames, a
    CodesEvent before Done."""

    request_id: int
    text_tokens: tuple[int, ...]
    language_id: int = 0
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    predictor_temperature: float = 0.0
    predictor_top_k: int = 0
    predictor_top_p: float = 1.0
    seed: int = 0
    max_frames: int = 0
    chunk_frames: int = 0
    reference_text_tokens: tuple[int, ...] = ()
    reference_codes: tuple[int, ...] = ()
    speaker_embedding: tuple[float, ...] = ()
    # SPEECH_FLAG_CODES or 0; the other bits are reserved.
    flags: int = 0
    instruction_tokens: tuple[int, ...] = ()
    preset_speaker: int = 0


@dataclass(slots=True, frozen=True)
class TranscribeRequest:
    """A transcription of samples: the chat prefix tokens up to and including
    <|audio_start|>, the suffix tokens from <|audio_end|> to the end of the
    assistant prefix (plus any forced `language X<asr_text>`), and the
    samples, little-endian float32 at 16 kHz, mono, between which the engine
    inserts the audio embeddings. max_tokens 0 selects the engine's
    default."""

    request_id: int
    prefix_tokens: tuple[int, ...]
    suffix_tokens: tuple[int, ...]
    samples: bytes
    max_tokens: int = 0
    # Reserved; must be 0.
    flags: int = 0

    @property
    def sample_count(self) -> int:
        return len(self.samples) // 4


@dataclass(slots=True, frozen=True)
class ReadyEvent:
    max_concurrent_requests: int
    max_context_tokens: int
    kind: ModelKind


@dataclass(slots=True, frozen=True)
class AudioEvent:
    """One chunk of a speech request's audio: little-endian int16 samples at
    24 kHz, mono, from sample_offset of the request's stream. Chunks arrive
    in order; Done follows the last."""

    request_id: int
    sample_offset: int
    pcm: bytes

    @property
    def sample_count(self) -> int:
        return len(self.pcm) // 2


@dataclass(slots=True, frozen=True)
class CodesEvent:
    """A speech request's generated frames, CODES_PER_FRAME codes each,
    before its Done, when the request asked for them (SPEECH_FLAG_CODES):
    what a later request takes as its reference."""

    request_id: int
    codes: tuple[int, ...]

    @property
    def frame_count(self) -> int:
        return len(self.codes) // CODES_PER_FRAME


ClientMessage: TypeAlias = (
    SpeechRequest | TranscribeRequest | CancelFrame | StatusRequestFrame
)
EngineEvent: TypeAlias = (
    ReadyEvent
    | StartEvent
    | TokensEvent
    | AudioEvent
    | CodesEvent
    | DoneEvent
    | ErrorEvent
    | StatusJsonEvent
)


def _payload_bounds(frame_type: FrameType) -> tuple[int, int]:
    match frame_type:
        case FrameType.SPEECH_REQUEST:
            bounds = (_SPEECH.size, MAX_FRAME_PAYLOAD_BYTES)
        case FrameType.TRANSCRIBE_REQUEST:
            bounds = (_TRANSCRIBE.size, MAX_FRAME_PAYLOAD_BYTES)
        case FrameType.CANCEL:
            bounds = (_CANCEL.size, _CANCEL.size)
        case FrameType.STATUS_REQUEST:
            bounds = (_STATUS_REQUEST.size, _STATUS_REQUEST.size)
        case FrameType.READY:
            bounds = (_READY.size, _READY.size)
        case FrameType.START:
            bounds = (_START.size, _START.size)
        case FrameType.TOKENS:
            bounds = (_TOKENS.size, _TOKENS.size + MAX_TOKEN_BATCH * 4)
        case FrameType.AUDIO:
            # A chunk is a few frames of audio; the frame limit is the bound.
            bounds = (_AUDIO.size, MAX_FRAME_PAYLOAD_BYTES)
        case FrameType.CODES:
            bounds = (_CODES.size, MAX_FRAME_PAYLOAD_BYTES)
        case FrameType.DONE:
            # No logits follow a Done event: its logit count is 0.
            bounds = (_DONE.size, _DONE.size)
        case FrameType.ERROR:
            bounds = (_ERROR.size, _ERROR.size + MAX_ERROR_STRING_BYTES * 2)
        case FrameType.STATUS_JSON:
            bounds = (_STATUS_JSON.size, _STATUS_JSON.size + MAX_STATUS_JSON_BYTES)
    return bounds[0], min(bounds[1], MAX_FRAME_PAYLOAD_BYTES)


def _payload_length_issue(
    frame_type: FrameType, payload_bytes: int
) -> ProtocolIssue | None:
    minimum, maximum = _payload_bounds(frame_type)
    if payload_bytes > maximum:
        return _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.FRAME_TOO_LARGE,
            f"{frame_type_name(frame_type)} payload length {payload_bytes} "
            f"exceeds its safe limit {maximum}",
        )
    if payload_bytes < minimum:
        return _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.INVALID_PAYLOAD_LENGTH,
            f"{frame_type_name(frame_type)} payload length {payload_bytes} "
            f"is below its required minimum {minimum}",
        )
    return None


def _frame(frame_type: FrameType, payload_bytes: int) -> bytearray:
    """A frame's header, in a buffer sized for the payload written after it."""
    frame = bytearray(FRAME_HEADER_BYTES + payload_bytes)
    _HEADER.pack_into(
        frame, 0, _MAGIC, PROTOCOL_VERSION, int(frame_type), payload_bytes
    )
    return frame


def _request_id(value: object, label: str) -> int:
    try:
        request_id = _u64(value, label)
        if not request_id:
            raise ValueError(f"{label} must be non-zero")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_REQUEST_ID, str(error))
    return request_id


def _reserved_flags(value: object, label: str, request_id: int) -> int:
    try:
        flags = _u32(value, label)
        if flags:
            raise ValueError(f"{label} are reserved and must be 0")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_ENUM_VALUE, str(error), request_id)
    return flags


def _sampling(temperature, top_k, top_p, label: str) -> tuple:
    """A sampling block as the frame carries it: a temperature of 0 or a
    normal float32 (the kernels divide by it, and Metal flushes a subnormal
    one to zero), a top_k count, and a top_p in (0, 1]."""
    values = (
        _float32(temperature, f"{label} temperature"),
        _u32(top_k, f"{label} top_k"),
        _float32(top_p, f"{label} top_p"),
    )
    temperature, _, top_p = values
    if not (temperature == 0.0 or _FLOAT32_MIN <= temperature < math.inf):
        raise ValueError(f"{label} temperature must be 0 or at least FLT_MIN")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError(f"{label} top_p must be in (0, 1]")
    return values


def _floats(values: object, label: str) -> array.array:
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple of float32 values")
    result = array.array("f", (_float32(value, f"{label} element") for value in values))
    if not all(map(math.isfinite, result)):
        raise ValueError(f"{label} must hold finite values")
    return result


def _speech_flags(value: object, request_id: int) -> int:
    try:
        flags = _u32(value, "speech request flags")
        if flags & ~SPEECH_FLAG_CODES:
            raise ValueError(
                "speech request flags other than SPEECH_FLAG_CODES are reserved"
            )
    except ValueError as error:
        _fail_request(IssueCode.INVALID_ENUM_VALUE, str(error), request_id)
    return flags


def _validated_speech(request: SpeechRequest) -> tuple:
    """The request's token and float arrays and its sampling values, each
    checked once; raises a ProtocolError with the first rule it breaks."""
    request_id = _request_id(request.request_id, "request id")
    flags = _speech_flags(request.flags, request_id)
    try:
        language_id = _u32(request.language_id, "language id")
        preset_speaker = _u32(request.preset_speaker, "preset speaker")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_ENUM_VALUE, str(error), request_id)
    try:
        text = _words(request.text_tokens, "text tokens")
        reference_text = _words(request.reference_text_tokens, "reference text tokens")
        instruction = _words(request.instruction_tokens, "instruction tokens")
        if not text or len(text) > MAX_TEXT_TOKENS:
            raise ValueError(
                f"text must hold 1..{MAX_TEXT_TOKENS} tokens, not {len(text)}"
            )
        if len(instruction) > MAX_INSTRUCTION_TOKENS:
            raise ValueError(f"instruction exceeds {MAX_INSTRUCTION_TOKENS} tokens")
        if len(reference_text) > MAX_TEXT_TOKENS:
            raise ValueError(f"reference text exceeds {MAX_TEXT_TOKENS} tokens")
    except ValueError as error:
        _fail_request(IssueCode.LIMIT_EXCEEDED, str(error), request_id)
    try:
        codes = _words(request.reference_codes, "reference codes")
        if len(codes) % CODES_PER_FRAME:
            raise ValueError(
                f"reference codes must hold {CODES_PER_FRAME} codes per frame"
            )
        speaker = _floats(request.speaker_embedding, "speaker embedding")
        if len(speaker) not in (0, SPEAKER_EMBEDDING_DIMS):
            raise ValueError(
                f"speaker embedding must hold {SPEAKER_EMBEDDING_DIMS} values or none"
            )
    except ValueError as error:
        _fail_request(IssueCode.INVALID_COUNT, str(error), request_id)
    try:
        talker = _sampling(request.temperature, request.top_k, request.top_p, "talker")
        predictor = _sampling(
            request.predictor_temperature,
            request.predictor_top_k,
            request.predictor_top_p,
            "predictor",
        )
        penalty = _float32(request.repetition_penalty, "repetition penalty")
        if not math.isfinite(penalty) or penalty <= 0.0:
            raise ValueError("repetition penalty must be positive")
    except (AttributeError, ValueError) as error:
        _fail_request(IssueCode.INVALID_SAMPLING, str(error), request_id)
    try:
        seed = _u64(request.seed, "seed")
        max_frames = _u32(request.max_frames, "max frames")
        chunk_frames = _u32(request.chunk_frames, "chunk frames")
    except ValueError as error:
        _fail_request(IssueCode.INTEGER_OVERFLOW, str(error), request_id)
    return (
        request_id,
        flags,
        language_id,
        talker,
        penalty,
        predictor,
        seed,
        max_frames,
        chunk_frames,
        text,
        instruction,
        reference_text,
        codes,
        speaker,
        preset_speaker,
    )


def _speech_request_frame(request: SpeechRequest) -> bytearray:
    """The whole frame, header included, built in one buffer."""
    (
        request_id,
        flags,
        language_id,
        (temperature, top_k, top_p),
        penalty,
        (predictor_temperature, predictor_top_k, predictor_top_p),
        seed,
        max_frames,
        chunk_frames,
        text,
        instruction,
        reference_text,
        codes,
        speaker,
        preset_speaker,
    ) = _validated_speech(request)
    text_offset = FRAME_HEADER_BYTES + _SPEECH.size
    instruction_offset = text_offset + 4 * len(text)
    reference_offset = instruction_offset + 4 * len(instruction)
    codes_offset = reference_offset + 4 * len(reference_text)
    speaker_offset = codes_offset + 4 * len(codes)
    payload_bytes = speaker_offset + 4 * len(speaker) - FRAME_HEADER_BYTES
    if issue := _payload_length_issue(FrameType.SPEECH_REQUEST, payload_bytes):
        _fail_request(issue.code, issue.message, request_id)
    frame = _frame(FrameType.SPEECH_REQUEST, payload_bytes)
    _SPEECH.pack_into(
        frame,
        FRAME_HEADER_BYTES,
        request_id,
        flags,
        language_id,
        temperature,
        top_k,
        top_p,
        penalty,
        predictor_temperature,
        predictor_top_k,
        predictor_top_p,
        seed,
        max_frames,
        chunk_frames,
        len(text),
        len(reference_text),
        len(codes) // CODES_PER_FRAME,
        1 if speaker else 0,
        len(instruction),
        preset_speaker,
    )
    with memoryview(frame) as view:
        view[text_offset:instruction_offset] = memoryview(text).cast("B")
        view[instruction_offset:reference_offset] = memoryview(instruction).cast("B")
        view[reference_offset:codes_offset] = memoryview(reference_text).cast("B")
        view[codes_offset:speaker_offset] = memoryview(codes).cast("B")
        view[speaker_offset:] = memoryview(speaker).cast("B")
    return frame


def _validated_transcribe(request: TranscribeRequest) -> tuple:
    request_id = _request_id(request.request_id, "request id")
    _reserved_flags(request.flags, "transcribe request flags", request_id)
    try:
        prefix = _words(request.prefix_tokens, "prefix tokens")
        suffix = _words(request.suffix_tokens, "suffix tokens")
        samples = _bytes(request.samples, "samples")
        if len(samples) % 4:
            raise ValueError("samples must be whole little-endian float32 values")
        if not samples or len(samples) // 4 > MAX_SAMPLES:
            raise ValueError(
                f"samples must hold 1..{MAX_SAMPLES} values, not {len(samples) // 4}"
            )
        if len(prefix) + len(suffix) > MAX_TEXT_TOKENS:
            raise ValueError(f"prompt tokens exceed {MAX_TEXT_TOKENS}")
    except ValueError as error:
        _fail_request(IssueCode.LIMIT_EXCEEDED, str(error), request_id)
    try:
        max_tokens = _u32(request.max_tokens, "max tokens")
    except ValueError as error:
        _fail_request(IssueCode.INTEGER_OVERFLOW, str(error), request_id)
    return request_id, max_tokens, prefix, suffix, samples


def _transcribe_request_frame(request: TranscribeRequest) -> bytearray:
    """The whole frame, header included, built in one buffer: the samples,
    tens of MiB for long audio, are copied once."""
    request_id, max_tokens, prefix, suffix, samples = _validated_transcribe(request)
    prefix_offset = FRAME_HEADER_BYTES + _TRANSCRIBE.size
    suffix_offset = prefix_offset + 4 * len(prefix)
    samples_offset = suffix_offset + 4 * len(suffix)
    payload_bytes = samples_offset + len(samples) - FRAME_HEADER_BYTES
    if issue := _payload_length_issue(FrameType.TRANSCRIBE_REQUEST, payload_bytes):
        _fail_request(issue.code, issue.message, request_id)
    frame = _frame(FrameType.TRANSCRIBE_REQUEST, payload_bytes)
    _TRANSCRIBE.pack_into(
        frame,
        FRAME_HEADER_BYTES,
        request_id,
        0,
        max_tokens,
        len(prefix),
        len(suffix),
        len(samples) // 4,
    )
    with memoryview(frame) as view:
        view[prefix_offset:suffix_offset] = memoryview(prefix).cast("B")
        view[suffix_offset:samples_offset] = memoryview(suffix).cast("B")
        view[samples_offset:] = samples
    return frame


def _cancel_frame(cancel: CancelFrame) -> bytearray:
    request_id = _request_id(cancel.request_id, "cancel request id")
    frame = _frame(FrameType.CANCEL, _CANCEL.size)
    _CANCEL.pack_into(frame, FRAME_HEADER_BYTES, request_id)
    return frame


def _status_request_frame(request: StatusRequestFrame) -> bytearray:
    try:
        correlation_id = _u64(request.correlation_id, "status correlation id")
    except ValueError as error:
        _fail_request(IssueCode.INTEGER_OVERFLOW, str(error))
    frame = _frame(FrameType.STATUS_REQUEST, _STATUS_REQUEST.size)
    _STATUS_REQUEST.pack_into(frame, FRAME_HEADER_BYTES, correlation_id)
    return frame


def serialize_message(message: ClientMessage) -> bytearray:
    """Validate and serialize one client message as a whole frame, built in
    one buffer: a transcription's samples are copied once."""
    try:
        if isinstance(message, SpeechRequest):
            return _speech_request_frame(message)
        if isinstance(message, TranscribeRequest):
            return _transcribe_request_frame(message)
        if isinstance(message, CancelFrame):
            return _cancel_frame(message)
        if isinstance(message, StatusRequestFrame):
            return _status_request_frame(message)
    except MemoryError as error:
        raise ProtocolError(
            _allocation_issue("allocation failed while serializing a client frame")
        ) from error
    _fail(
        FailureClass.PROTOCOL_FATAL,
        IssueCode.UNKNOWN_FRAME_TYPE,
        "message is not an audio protocol client message",
    )


def _checked_frame(frame: Frame) -> tuple[FrameType, bytes]:
    try:
        frame_type = FrameType(frame.type)
    except (TypeError, ValueError) as error:
        raise ProtocolError(
            _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.UNKNOWN_FRAME_TYPE,
                "frame type is not defined by the audio protocol",
            )
        ) from error
    try:
        payload = _bytes(frame.payload, "frame payload")
    except ValueError as error:
        _fatal(IssueCode.INVALID_PAYLOAD_LENGTH, str(error))
    if issue := _payload_length_issue(frame_type, len(payload)):
        raise ProtocolError(issue)
    return frame_type, payload


def serialize_frame(frame: Frame) -> bytes:
    """Serialize a validated low-level frame with a strict runtime header."""

    frame_type, payload = _checked_frame(frame)
    try:
        return (
            _HEADER.pack(_MAGIC, PROTOCOL_VERSION, int(frame_type), len(payload))
            + payload
        )
    except MemoryError as error:
        raise ProtocolError(
            _allocation_issue("allocation failed while serializing protocol frame")
        ) from error


def _decode_event(frame: Frame) -> EngineEvent:
    frame_type, payload = _checked_frame(frame)
    if frame_type is FrameType.READY:
        concurrent, context, kind = _READY.unpack(payload)
        try:
            kind = ModelKind(kind)
        except ValueError:
            _fatal(IssueCode.INVALID_ENUM_VALUE, "ready model kind is invalid")
        if not concurrent or not context:
            _fatal(IssueCode.INVALID_COUNT, "ready event capacities must be non-zero")
        return ReadyEvent(concurrent, context, kind)
    if frame_type is FrameType.AUDIO:
        request_id, offset, count = _AUDIO.unpack_from(payload)
        if len(payload) - _AUDIO.size != count * 2:
            _fatal(
                IssueCode.INVALID_PAYLOAD_LENGTH,
                "audio sample count does not match the binary event payload",
                request_id,
            )
        if not request_id:
            _fatal(
                IssueCode.INVALID_REQUEST_ID, "audio event request id must be non-zero"
            )
        if offset + count > 0xFFFFFFFF:
            _fatal(
                IssueCode.INTEGER_OVERFLOW,
                "audio event sample range overflows uint32",
                request_id,
            )
        return AudioEvent(request_id, offset, payload[_AUDIO.size :])
    if frame_type is FrameType.CODES:
        request_id, frames = _CODES.unpack_from(payload)
        if not request_id:
            _fatal(
                IssueCode.INVALID_REQUEST_ID, "codes event request id must be non-zero"
            )
        codes = _tail(
            payload, _CODES.size, frames * CODES_PER_FRAME, "I", "codes", request_id
        )
        return CodesEvent(request_id, codes)
    if frame_type in _SHARED_EVENT_TYPES:
        # Version 8's layout and rules; a Done event carries no logits here.
        event = wire.decode_frame(Frame(_SHARED_EVENT_TYPES[frame_type], payload))
        if isinstance(event, DoneEvent) and event.option_logits:
            _fatal(
                IssueCode.INVALID_COUNT,
                "done event carries option logits",
                event.request_id,
            )
        return event
    _fatal(IssueCode.UNKNOWN_FRAME_TYPE, "frame type is not an audio protocol event")


def decode_frame(frame: Frame) -> EngineEvent:
    """Decode and strictly validate one engine event."""

    try:
        return _decode_event(frame)
    except MemoryError as error:
        raise ProtocolError(
            _allocation_issue("allocation failed while decoding protocol message")
        ) from error


def pcm_samples(pcm: bytes) -> tuple[int, ...]:
    """The int16 samples an Audio event's bytes hold."""
    samples = array.array("h")
    samples.frombytes(pcm)
    return tuple(samples)


def float_samples(samples: bytes) -> tuple[float, ...]:
    """The float32 samples a transcription request's bytes hold."""
    return _tail(samples, 0, len(samples) // 4, "f", "sample", 0)


class FrameParser(wire.FrameParser):
    """Incremental, one-frame-at-a-time parser of engine events, version 11's
    frame types. A client frame type is an unknown frame type."""

    def __init__(self):
        super().__init__()
        self._current_type = FrameType.READY

    def _parse_header(self) -> ProtocolIssue | None:
        magic, version, raw_type, payload_bytes = _HEADER.unpack(self._header)
        if magic != _MAGIC:
            return _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.BAD_MAGIC,
                "frame magic is not SPLH",
            )
        if version != PROTOCOL_VERSION:
            return _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.UNSUPPORTED_VERSION,
                "unsupported native protocol version",
            )
        if raw_type not in _EVENT_TYPES:
            return _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.UNKNOWN_FRAME_TYPE,
                "frame type is not an audio protocol event",
            )
        self._current_type = FrameType(raw_type)
        if issue := _payload_length_issue(self._current_type, payload_bytes):
            return issue
        self._expected_payload_bytes = payload_bytes
        self._payload.clear()
        self._reading_payload = True
        return None
