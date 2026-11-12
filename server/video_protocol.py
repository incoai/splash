"""The server's side of the video engine's stream.

It encodes client frames and decodes the video engine's events;
``runtime/video/protocol/VideoProtocol.*`` does the reverse.
``dev/tests/engine/video_protocol_golden.txt`` pins the bytes both sides
agree on.

The stream's frames have the engine protocol's header (server/protocol.py)
with their own magic and version, so a frame sent to the other engine fails
at its header. Cancel, Error and StartFailed carry the engine protocol's
payloads, through its codec.
"""

from __future__ import annotations

import array
import struct
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import IntEnum, IntFlag
from typing import TypeAlias

from . import protocol
from .protocol import (
    FRAME_HEADER_BYTES,
    MAX_ERROR_STRING_BYTES,
    MAX_FRAME_PAYLOAD_BYTES,
    MAX_PROMPT_TOKENS,
    CancelFrame,
    ErrorEvent,
    FailureClass,
    IssueCode,
    ParseStep,
    ProtocolError,
    ProtocolIssue,
    StartFailedEvent,
)

VIDEO_PROTOCOL_VERSION = 1
_MAGIC = b"SPLV"
# The engine protocol's magic, which a start that fails before the video
# engine runs sends its StartFailed under (runtime/main.mm refuseStart).
ENGINE_MAGIC = b"SPLH"
_HEADER = struct.Struct("<4sHHQ")
# The header's magic and version.
_HEADER_START = struct.Struct("<4sH")
# A video request carries its Ref2VA references' normalized frames (three 15 s
# reference videos on a 768p canvas are 3.4 GB of RGB); the video engine
# takes VideoRequest frames up to this.
MAX_VIDEO_REQUEST_PAYLOAD_BYTES = 4 * 1024 * 1024 * 1024
# The references a video request carries at most (Pika H3 Flex's 12).
MAX_VIDEO_REFERENCES = 12

_VIDEO_REQUEST = struct.Struct("<QBBIIIIQBII")
_VIDEO_REFERENCE = struct.Struct("<BIIIQ")
# Concurrent requests, prompt tokens.
_VIDEO_READY = struct.Struct("<II")
_VIDEO_PROGRESS = struct.Struct("<QBIIQQ")
_VIDEO_OUTPUT = struct.Struct("<QBIQ")
# The engine protocol's fixed parts of Cancel, Error and StartFailed
# (server/protocol.py), which bound their frames here.
_CANCEL_BYTES = 8
_ERROR_FIXED_BYTES = 22
_START_FAILED_FIXED_BYTES = 26

assert _HEADER.size == FRAME_HEADER_BYTES
assert _VIDEO_REQUEST.size == 43
assert _VIDEO_REFERENCE.size == 21
assert _VIDEO_READY.size == 8
assert _VIDEO_PROGRESS.size == 33
assert _VIDEO_OUTPUT.size == 21


# A type the engine protocol also has keeps its id and its payload.
class VideoFrameType(IntEnum):
    CANCEL = 0x0002
    VIDEO_REQUEST = 0x0007

    ERROR = 0x0105
    START_FAILED = 0x0109
    VIDEO_PROGRESS = 0x010B
    VIDEO_OUTPUT = 0x010C
    VIDEO_READY = 0x010D


assert VideoFrameType.CANCEL == protocol.FrameType.CANCEL
assert VideoFrameType.ERROR == protocol.FrameType.ERROR
assert VideoFrameType.START_FAILED == protocol.FrameType.START_FAILED

# The frame types the video engine sends; the others are the server's.
_EVENT_TYPES = frozenset(
    {
        VideoFrameType.ERROR,
        VideoFrameType.START_FAILED,
        VideoFrameType.VIDEO_PROGRESS,
        VideoFrameType.VIDEO_OUTPUT,
        VideoFrameType.VIDEO_READY,
    }
)


# A video request names the DiT it runs on: a directory name under the
# weights, 1 to 64 of a-z, 0-9, ".", "_" and "-", without ".."
# (runtime/video/protocol/VideoProtocol.hpp validDitName).
MAX_DIT_NAME_BYTES = 64
_DIT_NAME_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")


def valid_dit_name(name: object) -> bool:
    return (
        isinstance(name, str)
        and 0 < len(name) <= MAX_DIT_NAME_BYTES
        and set(name) <= _DIT_NAME_CHARACTERS
        and ".." not in name
    )


class VideoOutputFlag(IntFlag):
    PREVIEW = 1 << 0
    AUDIO = 1 << 1


_VIDEO_OUTPUT_BITS = int(VideoOutputFlag.PREVIEW | VideoOutputFlag.AUDIO)


class VideoKeyframeFlag(IntFlag):
    FIRST = 1 << 0
    LAST = 1 << 1


class VideoReferenceKind(IntEnum):
    IMAGE = 0
    VIDEO = 1
    AUDIO = 2


@dataclass(slots=True, frozen=True)
class VideoReference:
    """A Ref2VA reference of a video request, as the server normalized it
    (references.normalize): an image at its tier size, a video's frames at
    24 fps, truncated to the clip, on its tier canvas, with its soundtrack,
    or an audio reference, a soundtrack alone (frames, width and height 0,
    no pixels)."""

    kind: VideoReferenceKind
    # An image's frames are 1, an audio reference's 0.
    frames: int
    width: int
    height: int
    # uint8 RGB [frames][height][width][3], any bytes-like object.
    pixels: bytes = dataclass_field(repr=False)
    # Little-endian fp32 [2][samples] at 32 kHz: a video's soundtrack (empty:
    # none), an audio reference's samples.
    soundtrack: bytes = dataclass_field(default=b"", repr=False)


@dataclass(slots=True, frozen=True)
class VideoRequestFrame:
    """One clip of a video model, of a prompt of the text encoder's token
    ids. Answered by VideoProgressEvents, then the clip's VideoOutputEvent,
    or an ErrorEvent."""

    request_id: int
    # The DiT's name (valid_dit_name): its weights' dit-<name>/.
    dit: str
    outputs: VideoOutputFlag
    frames: int
    width: int
    height: int
    # Denoising steps: a VDN-H3 DiT's; a FastH3 DiT's 0 or its rungs'.
    steps: int
    seed: int
    prompt_tokens: tuple[int, ...]
    # Keyframes, each empty or uint8 RGB [height][width][3] already on the
    # canvas (images.keyframe): the first frame anchors the clip's first
    # latent frame, the last frame its last.
    first_frame: bytes = b""
    last_frame: bytes = b""
    # Ref2VA references in presentation order; none with keyframes.
    references: tuple[VideoReference, ...] = ()


@dataclass(slots=True, frozen=True)
class VideoReadyEvent:
    """Sent once the video engine takes requests; a start that fails sends a
    StartFailedEvent instead."""

    max_concurrent_requests: int
    # A presentation's tokens at most, its vision blocks' included.
    max_prompt_tokens: int


class VideoPhase(IntEnum):
    ENCODE = 1
    PREPARE = 2
    DENOISE = 3
    PREVIEW = 4
    DECODE = 5
    AUDIO = 6
    MUX = 7


@dataclass(slots=True, frozen=True)
class VideoProgressEvent:
    """A phase has done `step` of its `steps`: step 0 starts it, `steps`
    completes it; its time so far and the engine's peak footprint."""

    request_id: int
    phase: VideoPhase
    step: int
    steps: int
    elapsed_micros: int
    peak_footprint_bytes: int


class VideoOutputKind(IntEnum):
    VIDEO = 0
    PREVIEW = 1


@dataclass(slots=True, frozen=True)
class VideoOutputEvent:
    """An mp4 of the request: the preview, or the clip, which ends it."""

    request_id: int
    output: VideoOutputKind
    frames: int
    # From the request's arrival.
    wall_micros: int
    mp4: bytes


ClientMessage: TypeAlias = VideoRequestFrame | CancelFrame
VideoEvent: TypeAlias = (
    VideoReadyEvent
    | VideoProgressEvent
    | VideoOutputEvent
    | ErrorEvent
    | StartFailedEvent
)


@dataclass(slots=True, frozen=True)
class Frame:
    type: VideoFrameType
    payload: bytes


def _issue(
    failure_class: FailureClass, code: IssueCode, message: str, request_id: int = 0
) -> ProtocolIssue:
    return ProtocolIssue(failure_class, code, request_id, message)


def _fail(
    failure_class: FailureClass, code: IssueCode, message: str, request_id: int = 0
) -> None:
    raise ProtocolError(_issue(failure_class, code, message, request_id))


def _fatal(code: IssueCode, message: str, request_id: int = 0) -> None:
    """An event that breaks the protocol: the stream is not trusted."""
    _fail(FailureClass.PROTOCOL_FATAL, code, message, request_id)


def _fail_request(code: IssueCode, message: str, request_id: int = 0) -> None:
    """A client frame's failure fails the request it names; one that names
    none fails the stream, as the engine classifies it."""
    _fail(
        FailureClass.REQUEST_ERROR if request_id else FailureClass.PROTOCOL_FATAL,
        code,
        message,
        request_id,
    )


def _allocation_issue(message: str) -> ProtocolIssue:
    return _issue(FailureClass.ENGINE_UNHEALTHY, IssueCode.ALLOCATION_FAILURE, message)


def _integer(value: object, maximum: int, label: str) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [0, {maximum}]")
    return value


def _u32(value: object, label: str) -> int:
    return _integer(value, 0xFFFFFFFF, label)


def _u64(value: object, label: str) -> int:
    return _integer(value, 0xFFFFFFFFFFFFFFFF, label)


def _enum_value(value: object, enum_type: type[IntEnum], label: str) -> IntEnum:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not defined by video protocol")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not defined by video protocol") from error


def _words(values: object, label: str) -> array.array:
    """A tuple of uint32 words as an array, laid out as the frame carries them."""
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple of uint32 values")
    if set(map(type, values)) == {int}:
        try:
            return array.array("I", values)
        except OverflowError:
            pass
    return array.array("I", (_u32(value, f"{label} element") for value in values))


def _payload_bounds(frame_type: VideoFrameType) -> tuple[int, int]:
    match frame_type:
        case VideoFrameType.VIDEO_REQUEST:
            # Its references' frames dominate; the video stream's own bound.
            return _VIDEO_REQUEST.size, MAX_VIDEO_REQUEST_PAYLOAD_BYTES
        case VideoFrameType.CANCEL:
            bounds = (_CANCEL_BYTES, _CANCEL_BYTES)
        case VideoFrameType.VIDEO_READY:
            bounds = (_VIDEO_READY.size, _VIDEO_READY.size)
        case VideoFrameType.VIDEO_PROGRESS:
            bounds = (_VIDEO_PROGRESS.size, _VIDEO_PROGRESS.size)
        case VideoFrameType.VIDEO_OUTPUT:
            bounds = (_VIDEO_OUTPUT.size, MAX_FRAME_PAYLOAD_BYTES)
        case VideoFrameType.ERROR:
            bounds = (
                _ERROR_FIXED_BYTES,
                _ERROR_FIXED_BYTES + MAX_ERROR_STRING_BYTES * 2,
            )
        case VideoFrameType.START_FAILED:
            bounds = (
                _START_FAILED_FIXED_BYTES,
                _START_FAILED_FIXED_BYTES + MAX_ERROR_STRING_BYTES,
            )
    return bounds[0], min(bounds[1], MAX_FRAME_PAYLOAD_BYTES)


def _payload_length_issue(
    frame_type: VideoFrameType, payload_bytes: int
) -> ProtocolIssue | None:
    minimum, maximum = _payload_bounds(frame_type)
    name = frame_type.name.lower()
    if payload_bytes > maximum:
        return _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.FRAME_TOO_LARGE,
            f"{name} payload length {payload_bytes} exceeds its safe limit {maximum}",
        )
    if payload_bytes < minimum:
        return _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.INVALID_PAYLOAD_LENGTH,
            f"{name} payload length {payload_bytes} is below its required minimum "
            f"{minimum}",
        )
    return None


def _frame(frame_type: VideoFrameType, payload_bytes: int) -> bytearray:
    """A frame's header, in a buffer sized for the payload written after it."""
    frame = bytearray(FRAME_HEADER_BYTES + payload_bytes)
    _HEADER.pack_into(
        frame, 0, _MAGIC, VIDEO_PROTOCOL_VERSION, int(frame_type), payload_bytes
    )
    return frame


def _video_headed(frame: bytearray) -> bytearray:
    """An engine protocol frame, its type and payload kept, under the video
    stream's magic and version."""
    _HEADER_START.pack_into(frame, 0, _MAGIC, VIDEO_PROTOCOL_VERSION)
    return frame


def _video_request_frame(request: VideoRequestFrame) -> bytearray:
    try:
        request_id = _u64(request.request_id, "video request id")
        if not request_id:
            raise ValueError("video request id must be non-zero")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_REQUEST_ID, str(error))
    try:
        if not valid_dit_name(request.dit):
            raise ValueError(
                "video DiT name is 1 to 64 of a-z, 0-9, '.', '_' and '-', without '..'"
            )
        dit = request.dit.encode()
        outputs = request.outputs
        if not isinstance(outputs, VideoOutputFlag) or outputs & ~_VIDEO_OUTPUT_BITS:
            raise ValueError("video outputs are not defined by video protocol")
        frames = _u32(request.frames, "video frames")
        width = _u32(request.width, "video width")
        height = _u32(request.height, "video height")
        steps = _u32(request.steps, "video steps")
        seed = _u64(request.seed, "video seed")
        tokens = _words(request.prompt_tokens, "prompt tokens")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_ENUM_VALUE, str(error), request_id)
    if not frames or not width or not height:
        _fail_request(
            IssueCode.INVALID_COUNT,
            "a video request needs frames, a width and a height",
            request_id,
        )
    if not tokens or len(tokens) > MAX_PROMPT_TOKENS:
        _fail_request(
            IssueCode.LIMIT_EXCEEDED,
            "a video prompt holds 1 token or more, within the prompt limit",
            request_id,
        )
    keyframes = (request.first_frame, request.last_frame)
    if any(len(pixels) not in (0, width * height * 3) for pixels in keyframes):
        _fail_request(
            IssueCode.INVALID_COUNT,
            "a keyframe holds height x width x 3 bytes",
            request_id,
        )
    flags = VideoKeyframeFlag(0)
    for flag, pixels in zip(VideoKeyframeFlag, keyframes):
        if pixels:
            flags |= flag
    references = _video_references(request.references, request_id)
    payload_bytes = (
        _VIDEO_REQUEST.size
        + len(dit)
        + 4 * len(tokens)
        + _VIDEO_REFERENCE.size * len(references)
        + sum(map(len, keyframes))
        + sum(len(pixels) + len(sound) for _, pixels, sound in references)
    )
    if issue := _payload_length_issue(VideoFrameType.VIDEO_REQUEST, payload_bytes):
        _fail_request(issue.code, issue.message, request_id)
    frame = _frame(VideoFrameType.VIDEO_REQUEST, payload_bytes)
    _VIDEO_REQUEST.pack_into(
        frame,
        FRAME_HEADER_BYTES,
        request_id,
        len(dit),
        int(outputs),
        frames,
        width,
        height,
        steps,
        seed,
        int(flags),
        len(tokens),
        len(references),
    )
    at = FRAME_HEADER_BYTES + _VIDEO_REQUEST.size
    frame[at : at + len(dit)] = dit
    at += len(dit)
    frame[at : at + 4 * len(tokens)] = memoryview(tokens).cast("B")
    at += 4 * len(tokens)
    for header, _, _ in references:
        _VIDEO_REFERENCE.pack_into(frame, at, *header)
        at += _VIDEO_REFERENCE.size
    # The variable parts: the keyframes, then each reference's pixels and
    # soundtrack, each copied once.
    for part in (*keyframes, *(b for _, p, s in references for b in (p, s))):
        frame[at : at + len(part)] = part
        at += len(part)
    return frame


def _video_references(references, request_id):
    """Each reference's header fields and its pixels and soundtrack as byte
    views, checked as the engine checks them."""
    if len(references) > MAX_VIDEO_REFERENCES:
        _fail_request(
            IssueCode.LIMIT_EXCEEDED,
            f"a video request holds at most {MAX_VIDEO_REFERENCES} references",
            request_id,
        )
    checked = []
    for reference in references:
        try:
            kind = _enum_value(
                reference.kind, VideoReferenceKind, "video reference kind"
            )
            frames = _u32(reference.frames, "reference frames")
            width = _u32(reference.width, "reference width")
            height = _u32(reference.height, "reference height")
            pixels = memoryview(reference.pixels).cast("B")
            sound = memoryview(reference.soundtrack).cast("B")
        except (TypeError, ValueError) as error:
            _fail_request(IssueCode.INVALID_ENUM_VALUE, str(error), request_id)
        image = kind == VideoReferenceKind.IMAGE
        if kind == VideoReferenceKind.AUDIO:
            if frames or width or height or not len(sound):
                _fail_request(
                    IssueCode.INVALID_COUNT,
                    "an audio reference is a soundtrack without frames",
                    request_id,
                )
        elif (
            not frames
            or not width
            or not height
            or (image and (frames != 1 or len(sound)))
        ):
            _fail_request(
                IssueCode.INVALID_COUNT,
                "a video reference needs frames, a width and a height; an "
                "image is one frame without a soundtrack",
                request_id,
            )
        if len(pixels) != frames * height * width * 3 or len(sound) % 8:
            _fail_request(
                IssueCode.INVALID_COUNT,
                "a video reference holds frames x height x width x 3 bytes and "
                "a stereo soundtrack",
                request_id,
            )
        checked.append(
            ((int(kind), frames, width, height, len(sound) // 8), pixels, sound)
        )
    return checked


def serialize_message(message: ClientMessage) -> bytearray:
    """Validate and serialize one client message as a whole frame, built in
    one buffer: a request's pixels are copied once. A Cancel is the engine
    protocol's (protocol.serialize_message)."""
    try:
        if isinstance(message, VideoRequestFrame):
            return _video_request_frame(message)
        if isinstance(message, CancelFrame):
            return _video_headed(protocol.serialize_message(message))
    except MemoryError as error:
        raise ProtocolError(
            _allocation_issue("allocation failed while serializing a client frame")
        ) from error
    _fail(
        FailureClass.PROTOCOL_FATAL,
        IssueCode.UNKNOWN_FRAME_TYPE,
        "message is not a video protocol client message",
    )


def _checked_frame(frame: Frame) -> tuple[VideoFrameType, bytes]:
    try:
        frame_type = VideoFrameType(frame.type)
    except (TypeError, ValueError) as error:
        raise ProtocolError(
            _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.UNKNOWN_FRAME_TYPE,
                "frame type is not defined by video protocol",
            )
        ) from error
    if type(frame.payload) is not bytes:
        _fatal(IssueCode.INVALID_PAYLOAD_LENGTH, "frame payload must be bytes")
    if issue := _payload_length_issue(frame_type, len(frame.payload)):
        raise ProtocolError(issue)
    return frame_type, frame.payload


def _decode_event(frame: Frame) -> VideoEvent:
    frame_type, payload = _checked_frame(frame)
    if frame_type in (VideoFrameType.ERROR, VideoFrameType.START_FAILED):
        # The engine protocol's payloads, which its codec checks.
        return protocol.decode_frame(
            protocol.Frame(protocol.FrameType(frame_type), payload)
        )
    if frame_type is VideoFrameType.VIDEO_READY:
        concurrent, tokens = _VIDEO_READY.unpack(payload)
        if not concurrent or not tokens:
            _fatal(IssueCode.INVALID_COUNT, "video ready capacities must be non-zero")
        return VideoReadyEvent(concurrent, tokens)
    if frame_type is VideoFrameType.VIDEO_PROGRESS:
        request_id, phase, step, steps, elapsed, peak = _VIDEO_PROGRESS.unpack(payload)
        if not request_id or phase not in VideoPhase._value2member_map_:
            _fatal(
                IssueCode.INVALID_ENUM_VALUE,
                "video progress request id or phase is invalid",
                request_id,
            )
        if not steps or step > steps:
            _fatal(
                IssueCode.INVALID_COUNT,
                "video progress steps must be 1 or more, its step at most them",
                request_id,
            )
        return VideoProgressEvent(
            request_id, VideoPhase(phase), step, steps, elapsed, peak
        )
    if frame_type is VideoFrameType.VIDEO_OUTPUT:
        request_id, output, frames, wall = _VIDEO_OUTPUT.unpack_from(payload)
        if not request_id or output not in VideoOutputKind._value2member_map_:
            _fatal(
                IssueCode.INVALID_ENUM_VALUE,
                "video output request id or kind is invalid",
                request_id,
            )
        mp4 = payload[_VIDEO_OUTPUT.size :]
        if not frames or not mp4:
            _fatal(
                IssueCode.LIMIT_EXCEEDED,
                "video output must hold frames and an mp4",
                request_id,
            )
        return VideoOutputEvent(request_id, VideoOutputKind(output), frames, wall, mp4)
    _fatal(IssueCode.UNKNOWN_FRAME_TYPE, "frame type is not a video protocol event")


def decode_frame(frame: Frame) -> VideoEvent:
    """Decode and strictly validate one video engine event."""

    try:
        return _decode_event(frame)
    except MemoryError as error:
        raise ProtocolError(
            _allocation_issue("allocation failed while decoding protocol message")
        ) from error


def starts_engine_protocol(data) -> bool:
    """Whether a stream that starts with data starts with an engine protocol
    frame: a start that fails before the video engine runs says why in one."""
    return bytes(data[: len(ENGINE_MAGIC)]) == ENGINE_MAGIC


class FrameParser:
    """Incremental, one-frame-at-a-time parser of video engine events, as
    protocol.FrameParser is: a client frame type is an unknown frame type,
    and an engine protocol frame has bad magic."""

    def __init__(self):
        self._header = bytearray()
        self._reading_payload = False
        self._current_type = VideoFrameType.VIDEO_READY
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
                "frame magic is not SPLV",
            )
        if version != VIDEO_PROTOCOL_VERSION:
            return _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.UNSUPPORTED_VERSION,
                "unsupported video protocol version",
            )
        if raw_type not in _EVENT_TYPES:
            return _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.UNKNOWN_FRAME_TYPE,
                "frame type is not a video protocol event",
            )
        self._current_type = VideoFrameType(raw_type)
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
                f"{self._current_type.name.lower()} payload bytes"
            )
        self._terminal_issue = _issue(
            FailureClass.PROTOCOL_FATAL, IssueCode.TRUNCATED_FRAME, message
        )
        return self._terminal_issue
