"""The base of the Splash Omni native protocol: what version 9, the audio
protocol in ``audio_protocol.py``, keeps of version 8.

It defines the framing, a 16-byte header of the magic, the protocol version,
the frame type and the payload length, then the payload; the failure classes
and issue codes; the frames whose layouts both versions share (Cancel and
StatusRequest from the server, Ready, Start, Tokens, Done, Error and
StatusJson from the engine) with their encoders and decoders; and the
incremental frame parser. ``runtime/engine/Protocol.*`` is the engine's
side, and docs/omni-design.md, "Protocol (version 9)", the contract. The
audio protocol adds its own request and audio frames, reads Ready's last
byte as the model kind, and stamps version 9 in each header.
"""

from __future__ import annotations

import array
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

PROTOCOL_VERSION = 8
# A frame is its header, the magic, the protocol version, the frame type and
# the payload length, then the payload.
FRAME_HEADER_BYTES = 16
STATUS_SCHEMA_VERSION = 6
# A transcription's samples travel inside the request frame: tens of MiB for
# long audio.
MAX_FRAME_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_STATUS_JSON_BYTES = 32 * 1024 * 1024
MAX_ERROR_STRING_BYTES = 1024 * 1024
MAX_TOKEN_BATCH = 4096

_MAGIC = b"SPLH"
_HEADER = struct.Struct("<4sHHQ")
_CANCEL = struct.Struct("<Q")
_STATUS_REQUEST = struct.Struct("<Q")
_READY = struct.Struct("<IIB")
_START = struct.Struct("<QII")
_TOKENS = struct.Struct("<QII")
_DONE = struct.Struct("<QBIIQQQI")
_ERROR = struct.Struct("<BBQII")
_STATUS_JSON = struct.Struct("<Q")

assert (
    array.array("I").itemsize == 4
    and array.array("f").itemsize == 4
    and sys.byteorder == "little"
)
assert _HEADER.size == FRAME_HEADER_BYTES
assert _READY.size == 9
assert _START.size == 16
assert _DONE.size == 45
assert _ERROR.size == 18
assert _STATUS_JSON.size == 8


class FrameType(IntEnum):
    CANCEL = 0x0002
    STATUS_REQUEST = 0x0004

    READY = 0x0100
    START = 0x0101
    TOKENS = 0x0102
    DONE = 0x0104
    ERROR = 0x0105
    STATUS_JSON = 0x0107


# The frame types the engine sends; the others are the server's.
_EVENT_TYPES = frozenset(
    {
        FrameType.READY,
        FrameType.START,
        FrameType.TOKENS,
        FrameType.DONE,
        FrameType.ERROR,
        FrameType.STATUS_JSON,
    }
)


class FailureClass(IntEnum):
    REQUEST_ERROR = 1
    ENGINE_UNHEALTHY = 2
    PROTOCOL_FATAL = 3


# Issue codes travel by name; the values are runtime/engine/Protocol.hpp's.
class IssueCode(IntEnum):
    NONE = 0
    BAD_MAGIC = 1
    UNSUPPORTED_VERSION = 2
    UNKNOWN_FRAME_TYPE = 3
    FRAME_TOO_LARGE = 4
    INVALID_PAYLOAD_LENGTH = 5
    TRUNCATED_FRAME = 6
    INVALID_REQUEST_ID = 7
    INVALID_ENUM_VALUE = 8
    INVALID_DEADLINE = 9
    INVALID_SAMPLING = 10
    INVALID_COUNT = 11
    INVALID_CONSTRAINT = 12
    INVALID_ERROR_CLASSIFICATION = 13
    LIMIT_EXCEEDED = 14
    INTEGER_OVERFLOW = 15
    ALLOCATION_FAILURE = 16


def frame_type_name(frame_type: FrameType) -> str:
    return frame_type.name.lower()


def failure_class_name(failure_class: FailureClass) -> str:
    return failure_class.name.lower()


def issue_code_name(code: IssueCode) -> str:
    return code.name.lower()


@dataclass(slots=True, frozen=True)
class ProtocolIssue:
    failure_class: FailureClass
    code: IssueCode
    request_id: int
    message: str

    def describe(self) -> str:
        result = (
            f"{failure_class_name(self.failure_class)}:{issue_code_name(self.code)}"
        )
        if self.request_id:
            result += f" request={self.request_id}"
        if self.message:
            result += f": {self.message}"
        return result


class ProtocolError(Exception):
    def __init__(self, issue: ProtocolIssue):
        self.issue = issue
        super().__init__(issue.describe())


@dataclass(slots=True, frozen=True)
class CancelFrame:
    request_id: int


@dataclass(slots=True, frozen=True)
class StatusRequestFrame:
    correlation_id: int


@dataclass(slots=True, frozen=True)
class ReadyEvent:
    """Version 8's Ready: its last byte says whether requests may carry
    images. Version 9 reads the same bytes as the model kind
    (audio_protocol.ReadyEvent)."""

    max_concurrent_requests: int
    max_context_tokens: int
    vision: bool


@dataclass(slots=True, frozen=True)
class StartEvent:
    request_id: int
    lane: int
    # The cached prefix the request starts from; zero for a cold start.
    matched_prompt_tokens: int


@dataclass(slots=True, frozen=True)
class TokensEvent:
    request_id: int
    sequence_offset: int
    tokens: tuple[int, ...]


class FinishReason(IntEnum):
    STOP = 0
    LENGTH = 1
    CANCELLED = 2


@dataclass(slots=True, frozen=True)
class DoneEvent:
    request_id: int
    reason: FinishReason
    prompt_tokens: int
    # Tokens of a transcription; audio frames of a speech request.
    completion_tokens: int
    prefill_micros: int
    decode_micros: int
    wall_micros: int
    # The frame ends with a logit count, 0: no logits follow a Done event.
    option_logits: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class ErrorEvent:
    failure_class: FailureClass
    request_id: int
    retryable: bool
    code: bytes
    message: bytes


# The JSON is opaque to the transport: the document carries its own
# schema_version.
@dataclass(slots=True, frozen=True)
class StatusJsonEvent:
    correlation_id: int
    json: bytes


ClientMessage: TypeAlias = CancelFrame | StatusRequestFrame
EngineEvent: TypeAlias = (
    ReadyEvent | StartEvent | TokensEvent | DoneEvent | ErrorEvent | StatusJsonEvent
)


@dataclass(slots=True, frozen=True)
class Frame:
    type: FrameType
    payload: bytes


@dataclass(slots=True, frozen=True)
class ParseStep:
    consumed_bytes: int = 0
    frame: Frame | None = None
    issue: ProtocolIssue | None = None


def _issue(
    failure_class: FailureClass,
    code: IssueCode,
    message: str,
    request_id: int = 0,
) -> ProtocolIssue:
    return ProtocolIssue(failure_class, code, request_id, message)


def _fail(
    failure_class: FailureClass,
    code: IssueCode,
    message: str,
    request_id: int = 0,
) -> None:
    raise ProtocolError(_issue(failure_class, code, message, request_id))


def _fatal(code: IssueCode, message: str, request_id: int = 0) -> None:
    """An engine event that breaks the protocol: the stream is not trusted."""
    _fail(FailureClass.PROTOCOL_FATAL, code, message, request_id)


def _fail_request(code: IssueCode, message: str, request_id: int = 0) -> None:
    """A client frame's failure fails the request it names. One that names
    none cannot be answered as a request's error and fails the stream, as
    the engine classifies it (Protocol.cpp requestIssue)."""
    _fail(
        FailureClass.REQUEST_ERROR if request_id else FailureClass.PROTOCOL_FATAL,
        code,
        message,
        request_id,
    )


def _allocation_issue(message: str) -> ProtocolIssue:
    return _issue(FailureClass.ENGINE_UNHEALTHY, IssueCode.ALLOCATION_FAILURE, message)


def _integer(value: object, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _u32(value: object, label: str) -> int:
    return _integer(value, 0, 0xFFFFFFFF, label)


def _u64(value: object, label: str) -> int:
    return _integer(value, 0, 0xFFFFFFFFFFFFFFFF, label)


def _float32(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a float32 value")
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{label} must be a float32 value") from error


# A sampling temperature is 0 or a normal float32: the kernels divide by it,
# and Metal flushes a subnormal one to zero.
_FLOAT32_MIN = float.fromhex("0x1p-126")


def _bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise ValueError(f"{label} must be bytes")
    return value


def _words(values: object, label: str) -> array.array:
    """A tuple of uint32 words as an array, laid out as the frame carries them."""
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple of uint32 values")
    # Token runs reach thousands of words, so accept the common valid case in
    # C: the array refuses an int outside uint32. Anything else falls through
    # to the per-word check for its exact error.
    if set(map(type, values)) == {int}:
        try:
            return array.array("I", values)
        except OverflowError:
            pass
    return array.array("I", (_u32(value, f"{label} element") for value in values))


def _payload_bounds(frame_type: FrameType) -> tuple[int, int]:
    match frame_type:
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


def _cancel_frame(cancel: CancelFrame) -> bytearray:
    try:
        if not _u64(cancel.request_id, "cancel request id"):
            raise ValueError("cancel request id must be non-zero")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_REQUEST_ID, str(error))
    frame = _frame(FrameType.CANCEL, _CANCEL.size)
    _CANCEL.pack_into(frame, FRAME_HEADER_BYTES, cancel.request_id)
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
    """Validate and serialize one client message as a whole frame."""
    try:
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
        "message is not a native protocol client message",
    )


def _checked_frame(frame: Frame) -> tuple[FrameType, bytes]:
    try:
        frame_type = FrameType(frame.type)
    except (TypeError, ValueError) as error:
        raise ProtocolError(
            _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.UNKNOWN_FRAME_TYPE,
                "frame type is not defined by native protocol",
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


def _tail(
    payload: bytes, offset: int, count: int, code: str, label: str, request_id: int
) -> tuple:
    """The ``count`` words (``I``) or floats (``f``) that end an event payload."""
    if len(payload) - offset != count * 4:
        _fatal(
            IssueCode.INVALID_PAYLOAD_LENGTH,
            f"{label} count does not match the binary event payload",
            request_id,
        )
    values = array.array(code)
    values.frombytes(payload[offset:])
    return tuple(values)


def _decode_event(frame: Frame) -> EngineEvent:
    frame_type, payload = _checked_frame(frame)
    if frame_type is FrameType.READY:
        concurrent, context, vision = _READY.unpack(payload)
        if vision > 1:
            _fatal(IssueCode.INVALID_ENUM_VALUE, "ready vision must be a boolean")
        if not concurrent or not context:
            _fatal(IssueCode.INVALID_COUNT, "ready event capacities must be non-zero")
        return ReadyEvent(concurrent, context, bool(vision))
    if frame_type is FrameType.START:
        request_id, lane, matched = _START.unpack(payload)
        if not request_id:
            _fatal(
                IssueCode.INVALID_REQUEST_ID, "start event request id must be non-zero"
            )
        return StartEvent(request_id, lane, matched)
    if frame_type is FrameType.TOKENS:
        request_id, offset, count = _TOKENS.unpack_from(payload)
        tokens = _tail(payload, _TOKENS.size, count, "I", "token", request_id)
        if not request_id:
            _fatal(
                IssueCode.INVALID_REQUEST_ID, "tokens event request id must be non-zero"
            )
        if not tokens:
            _fatal(
                IssueCode.LIMIT_EXCEEDED, "tokens event carries no token", request_id
            )
        if offset + len(tokens) > 0xFFFFFFFF:
            _fatal(
                IssueCode.INTEGER_OVERFLOW,
                "tokens event sequence range overflows uint32",
                request_id,
            )
        return TokensEvent(request_id, offset, tokens)
    if frame_type is FrameType.DONE:
        request_id, reason, *counts, logit_count = _DONE.unpack_from(payload)
        logits = _tail(
            payload, _DONE.size, logit_count, "f", "option logit", request_id
        )
        try:
            reason = FinishReason(reason)
        except ValueError:
            _fatal(
                IssueCode.INVALID_ENUM_VALUE,
                "done finish reason is invalid",
                request_id,
            )
        if not request_id:
            _fatal(
                IssueCode.INVALID_REQUEST_ID, "done event request id must be non-zero"
            )
        return DoneEvent(request_id, reason, *counts, logits)
    if frame_type is FrameType.ERROR:
        failure, retryable, request_id, code_bytes, message_bytes = _ERROR.unpack_from(
            payload
        )
        if retryable > 1 or code_bytes + message_bytes != len(payload) - _ERROR.size:
            _fatal(
                IssueCode.INVALID_PAYLOAD_LENGTH,
                "error string lengths do not match the frame payload",
            )
        try:
            classification = FailureClass(failure)
        except ValueError:
            _fatal(
                IssueCode.INVALID_ERROR_CLASSIFICATION,
                "error event classification is invalid",
                request_id,
            )
        if (classification is FailureClass.REQUEST_ERROR) != bool(request_id):
            _fatal(
                IssueCode.INVALID_ERROR_CLASSIFICATION,
                "only request errors may carry a non-zero request id",
                request_id,
            )
        if (
            not code_bytes
            or code_bytes > MAX_ERROR_STRING_BYTES
            or message_bytes > MAX_ERROR_STRING_BYTES
        ):
            _fatal(
                IssueCode.LIMIT_EXCEEDED,
                "error code or message exceeds its safe limit",
                request_id,
            )
        message_start = _ERROR.size + code_bytes
        return ErrorEvent(
            classification,
            request_id,
            bool(retryable),
            payload[_ERROR.size : message_start],
            payload[message_start:],
        )
    if frame_type is FrameType.STATUS_JSON:
        (correlation_id,) = _STATUS_JSON.unpack_from(payload)
        json = payload[_STATUS_JSON.size :]
        if not json:
            _fatal(IssueCode.LIMIT_EXCEEDED, "status JSON is empty")
        return StatusJsonEvent(correlation_id, json)
    _fatal(IssueCode.UNKNOWN_FRAME_TYPE, "frame type is not a native protocol event")


def decode_frame(frame: Frame) -> EngineEvent:
    """Decode and strictly validate one engine event."""

    try:
        return _decode_event(frame)
    except MemoryError as error:
        raise ProtocolError(
            _allocation_issue("allocation failed while decoding protocol message")
        ) from error


class FrameParser:
    """Incremental, one-frame-at-a-time parser of engine events. A client
    frame type is an unknown frame type."""

    def __init__(self):
        self._header = bytearray()
        self._reading_payload = False
        self._current_type = FrameType.READY
        self._expected_payload_bytes = 0
        self._payload = bytearray()
        self._terminal_issue: ProtocolIssue | None = None

    def _fail(self, consumed: int, issue: ProtocolIssue) -> ParseStep:
        self._terminal_issue = issue
        return ParseStep(consumed, issue=issue)

    def _reset_current_frame(self) -> None:
        self._header.clear()
        self._reading_payload = False
        self._expected_payload_bytes = 0
        self._payload.clear()

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
                "frame type is not a native protocol event",
            )
        self._current_type = FrameType(raw_type)
        if issue := _payload_length_issue(self._current_type, payload_bytes):
            return issue
        self._expected_payload_bytes = payload_bytes
        self._payload.clear()
        self._reading_payload = True
        return None

    def consume(self, data: bytes | bytearray | memoryview) -> ParseStep:
        """Consume up to one frame and report exactly how many bytes were used."""

        if self._terminal_issue:
            raise RuntimeError("frame parser used after it failed")
        try:
            view = memoryview(data).cast("B")
        except (TypeError, ValueError) as error:
            return self._fail(
                0,
                _issue(
                    FailureClass.PROTOCOL_FATAL,
                    IssueCode.INVALID_PAYLOAD_LENGTH,
                    f"parser input must be a contiguous byte buffer: {error}",
                ),
            )

        consumed = 0
        while consumed < len(view):
            if not self._reading_payload:
                count = min(
                    FRAME_HEADER_BYTES - len(self._header), len(view) - consumed
                )
                self._header.extend(view[consumed : consumed + count])
                consumed += count
                if len(self._header) < FRAME_HEADER_BYTES:
                    return ParseStep(consumed)
                if issue := self._parse_header():
                    return self._fail(consumed, issue)
                if not self._expected_payload_bytes:
                    frame = Frame(self._current_type, b"")
                    self._reset_current_frame()
                    return ParseStep(consumed, frame)

            needed = self._expected_payload_bytes - len(self._payload)
            count = min(needed, len(view) - consumed)
            try:
                self._payload.extend(view[consumed : consumed + count])
            except MemoryError:
                return self._fail(
                    consumed,
                    _allocation_issue(
                        "allocation failed while receiving frame payload"
                    ),
                )
            consumed += count
            if len(self._payload) == self._expected_payload_bytes:
                try:
                    payload = bytes(self._payload)
                except MemoryError:
                    return self._fail(
                        consumed,
                        _allocation_issue(
                            "allocation failed while publishing frame payload"
                        ),
                    )
                frame = Frame(self._current_type, payload)
                self._reset_current_frame()
                return ParseStep(consumed, frame)
        return ParseStep(consumed)

    def finish(self) -> ProtocolIssue | None:
        """Finish EOF processing, making any partial frame terminally fatal."""

        if self._terminal_issue:
            raise RuntimeError("frame parser used after it failed")
        if not self._header and not self._reading_payload:
            return None
        if not self._reading_payload:
            message = (
                f"stream ended after {len(self._header)} of "
                f"{FRAME_HEADER_BYTES} frame-header bytes"
            )
        else:
            message = (
                f"stream ended after {len(self._payload)} of "
                f"{self._expected_payload_bytes} "
                f"{frame_type_name(self._current_type)} payload bytes"
            )
        self._terminal_issue = _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.TRUNCATED_FRAME,
            message,
        )
        return self._terminal_issue
