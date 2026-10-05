"""The server's side of the Splash native protocol.

It encodes client frames and decodes engine events; ``runtime/engine/
Protocol.*`` does the reverse. ``dev/tests/engine/protocol_golden.txt`` pins
the bytes both sides agree on.
"""

from __future__ import annotations

import array
import math
import struct
import sys
from dataclasses import dataclass, fields
from enum import IntEnum, IntFlag
from typing import TypeAlias

PROTOCOL_VERSION = 8
# A frame is its header, the magic, the protocol version, the frame type and
# the payload length, then the payload.
FRAME_HEADER_BYTES = 16
STATUS_SCHEMA_VERSION = 6
# Score-only requests carry 2..255 distinct option token ids and produce no
# generated tokens; a successful score DoneEvent returns one raw
# final-position logit per requested token, in request order.
MIN_SCORE_TOKENS = 2
MAX_SCORE_TOKENS = 255
# Patches per image, the most the vision encoder takes; the server's pixel
# cap may allow fewer.
MAX_IMAGE_PATCHES = 16384
# Image pixels travel inside the request frame; a multi-image agent turn can
# carry well over 64 MiB of resized RGB bytes.
MAX_FRAME_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_STATUS_JSON_BYTES = 32 * 1024 * 1024
MAX_ERROR_STRING_BYTES = 1024 * 1024
MAX_PROMPT_TOKENS = 1 << 20
MAX_LOGICAL_OUTPUT_TOKENS = 1 << 20
MAX_TOKEN_BATCH = 4096
MAX_SIMULATION_TOKENS = 32
MAX_IMAGE_SPANS = 64
MAX_MASK_WORDS = 1 << 20

_MAGIC = b"SPLH"
_HEADER = struct.Struct("<4sHHQ")
# Replay can update the integer deadlines without decoding sampling floats.
_REQUEST_HEAD = struct.Struct("<QBBQQ")
_REQUEST = struct.Struct(_REQUEST_HEAD.format + "IIIffIffffQIII")
_IMAGE_SPAN = struct.Struct("<IIIIQQ")
_CANCEL = struct.Struct("<Q")
_MASK_RESPONSE = struct.Struct("<QQI")
_STATUS_REQUEST = struct.Struct("<Q")
_READY = struct.Struct("<IIB")
_START = struct.Struct("<QII")
_PROMPT_PROGRESS = struct.Struct("<QIQ")
_TOKENS = struct.Struct("<QII")
_MASK_REQUEST = struct.Struct("<QQII")
_DONE = struct.Struct("<QBIIQQQI")
_ERROR = struct.Struct("<BBQII")
_STATUS_JSON = struct.Struct("<Q")

assert (
    array.array("I").itemsize == 4
    and array.array("f").itemsize == 4
    and sys.byteorder == "little"
)
assert _HEADER.size == FRAME_HEADER_BYTES
assert _REQUEST.size == 86
assert _IMAGE_SPAN.size == 32
assert _READY.size == 9
assert _START.size == 16
assert _DONE.size == 45
assert _ERROR.size == 18
assert _STATUS_JSON.size == 8

REQUEST_FIXED_BYTES = _REQUEST.size
IMAGE_SPAN_BYTES = _IMAGE_SPAN.size


class FrameType(IntEnum):
    REQUEST = 0x0001
    CANCEL = 0x0002
    MASK_RESPONSE = 0x0003
    STATUS_REQUEST = 0x0004

    READY = 0x0100
    START = 0x0101
    TOKENS = 0x0102
    MASK_REQUEST = 0x0103
    DONE = 0x0104
    ERROR = 0x0105
    STATUS_JSON = 0x0107
    PROMPT_PROGRESS = 0x0108


# The frame types the engine sends; the others are the server's.
_EVENT_TYPES = frozenset(
    {
        FrameType.READY,
        FrameType.START,
        FrameType.TOKENS,
        FrameType.MASK_REQUEST,
        FrameType.DONE,
        FrameType.ERROR,
        FrameType.STATUS_JSON,
        FrameType.PROMPT_PROGRESS,
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


class RequestPriority(IntEnum):
    FOREGROUND = 0
    NORMAL = 1
    BACKGROUND = 2


class ConstraintMode(IntEnum):
    NONE = 0
    TOKEN_MASK = 1


class RequestFlag(IntFlag):
    # Never select the model's stop tokens, so generation runs to its output
    # limit. Only unconstrained generation can carry it.
    IGNORE_END_OF_SEQUENCE = 1 << 0
    # Report the prompt's progress while it prefills (PromptProgressEvent).
    RETURN_PROGRESS = 1 << 1


# A request with any other bit set is a request error.
_REQUEST_FLAG_BITS = int(
    RequestFlag.IGNORE_END_OF_SEQUENCE | RequestFlag.RETURN_PROGRESS
)


@dataclass(slots=True, frozen=True)
class SamplingParameters:
    """The defaults are greedy selection with nothing changing the logits,
    which score requests require. A top_k of 0, or one past the vocabulary,
    keeps every token; the default penalties change nothing, and a min_p of
    0 drops no token."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_p: float = 0.0


# The sampling options a request names, in the frame's order.
SAMPLING_FIELDS: tuple[str, ...] = tuple(
    field.name for field in fields(SamplingParameters)
)


@dataclass(slots=True, frozen=True)
class ImageSpan:
    """One image in the prompt: its placeholder token run, the patch grid of
    the resized pixels, and a 128-bit digest of that content. Placeholder
    token ids are identical for every image, so cache identity keys on spans.
    """

    offset: int
    tokens: int
    grid_height: int
    grid_width: int
    digest_lo: int
    digest_hi: int

    @property
    def pixel_bytes(self) -> int:
        return self.grid_height * 16 * self.grid_width * 16 * 3


@dataclass(slots=True, frozen=True)
class RequestFrame:
    request_id: int
    priority: RequestPriority
    absolute_deadline_unix_micros: int
    remaining_deadline_micros: int
    logical_max_output_tokens: int
    prompt_tokens: tuple[int, ...]
    sampling: SamplingParameters
    seed: int
    constraint: ConstraintMode
    # Sorted, non-overlapping image spans and their resized uint8 RGB pixels
    # concatenated in span order; both empty for text-only requests.
    image_spans: tuple[ImageSpan, ...]
    image_pixels: bytes
    # Option token ids for score-only requests; empty means ordinary
    # generation. Score tokens serialize after the image pixel bytes.
    score_tokens: tuple[int, ...]
    # Trailing prompt tokens of the chat template's generation prompt; zero
    # when unknown. It must leave at least one prompt token.
    generation_prompt_tokens: int
    flags: RequestFlag


@dataclass(slots=True, frozen=True)
class CancelFrame:
    request_id: int


@dataclass(slots=True, frozen=True)
class MaskResponseFrame:
    request_id: int
    mask_request_id: int
    # Little-endian uint32 words, words_per_mask per mask row.
    mask_words: bytes


@dataclass(slots=True, frozen=True)
class StatusRequestFrame:
    correlation_id: int


@dataclass(slots=True, frozen=True)
class ReadyEvent:
    max_concurrent_requests: int
    max_context_tokens: int
    # Requests may carry image spans; false for a model serving without vision.
    vision: bool


@dataclass(slots=True, frozen=True)
class StartEvent:
    request_id: int
    lane: int
    # The cached prefix the request starts from; zero for a cold start.
    matched_prompt_tokens: int


@dataclass(slots=True, frozen=True)
class PromptProgressEvent:
    request_id: int
    processed_tokens: int
    elapsed_micros: int


@dataclass(slots=True, frozen=True)
class TokensEvent:
    request_id: int
    sequence_offset: int
    tokens: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class MaskRequestEvent:
    request_id: int
    mask_request_id: int
    words_per_mask: int
    simulation_tokens: tuple[int, ...]

    @property
    def mask_rows(self) -> int:
        """Rows before and after each simulated context token.

        The initial request carries no tokens and therefore has one row. A
        verify request carries ``(pending_anchor, *draft_proposals)``; its
        target-logit rows consume masks beginning at row one.
        """
        return len(self.simulation_tokens) + 1


class FinishReason(IntEnum):
    STOP = 0
    LENGTH = 1
    CANCELLED = 2


@dataclass(slots=True, frozen=True)
class DoneEvent:
    request_id: int
    reason: FinishReason
    prompt_tokens: int
    completion_tokens: int
    prefill_micros: int
    decode_micros: int
    wall_micros: int
    # Raw final-position logits for a score-only request, in requested token
    # order; empty for generation and for cancelled or failed scoring.
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


ClientMessage: TypeAlias = (
    RequestFrame | CancelFrame | MaskResponseFrame | StatusRequestFrame
)
EngineEvent: TypeAlias = (
    ReadyEvent
    | StartEvent
    | PromptProgressEvent
    | TokensEvent
    | MaskRequestEvent
    | DoneEvent
    | ErrorEvent
    | StatusJsonEvent
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


def _sampling_values(sampling: SamplingParameters) -> tuple:
    """The sampling block as the frame carries it, in its wire order."""
    return (
        _float32(sampling.temperature, "temperature"),
        _float32(sampling.top_p, "top_p"),
        _u32(sampling.top_k, "top_k"),
        _float32(sampling.presence_penalty, "presence_penalty"),
        _float32(sampling.frequency_penalty, "frequency_penalty"),
        _float32(sampling.repetition_penalty, "repetition_penalty"),
        _float32(sampling.min_p, "min_p"),
    )


_NEUTRAL_SAMPLING = _sampling_values(SamplingParameters())
# A sampling temperature is 0 or a normal float32: the kernels divide by it,
# and Metal flushes a subnormal one to zero.
_FLOAT32_MIN = float.fromhex("0x1p-126")


def _enum_value(value: object, enum_type: type[IntEnum], label: str) -> IntEnum:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not defined by native protocol")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not defined by native protocol") from error


def _bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise ValueError(f"{label} must be bytes")
    return value


def _words(values: object, label: str) -> array.array:
    """A tuple of uint32 words, packed as the frame carries them."""
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple of uint32 values")
    # Prompts run to a million words, so accept the common valid case in C:
    # the array refuses an int outside uint32. Anything else falls through to
    # the per-word check for its exact error.
    if set(map(type, values)) == {int}:
        try:
            return array.array("I", values)
        except OverflowError:
            pass
    return array.array("I", (_u32(value, f"{label} element") for value in values))


def _payload_bounds(frame_type: FrameType) -> tuple[int, int]:
    match frame_type:
        case FrameType.REQUEST:
            # Image pixels dominate prompt tokens; the frame limit is the bound.
            bounds = (_REQUEST.size, MAX_FRAME_PAYLOAD_BYTES)
        case FrameType.CANCEL:
            bounds = (_CANCEL.size, _CANCEL.size)
        case FrameType.MASK_RESPONSE:
            bounds = (_MASK_RESPONSE.size, _MASK_RESPONSE.size + MAX_MASK_WORDS * 4)
        case FrameType.STATUS_REQUEST:
            bounds = (_STATUS_REQUEST.size, _STATUS_REQUEST.size)
        case FrameType.READY:
            bounds = (_READY.size, _READY.size)
        case FrameType.PROMPT_PROGRESS:
            bounds = (_PROMPT_PROGRESS.size, _PROMPT_PROGRESS.size)
        case FrameType.START:
            bounds = (_START.size, _START.size)
        case FrameType.TOKENS:
            bounds = (_TOKENS.size, _TOKENS.size + MAX_TOKEN_BATCH * 4)
        case FrameType.MASK_REQUEST:
            bounds = (
                _MASK_REQUEST.size,
                _MASK_REQUEST.size + MAX_SIMULATION_TOKENS * 4,
            )
        case FrameType.DONE:
            bounds = (_DONE.size, _DONE.size + MAX_SCORE_TOKENS * 4)
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


def _image_spans_check(request: RequestFrame, prompt_tokens: int) -> None:
    previous_end = 0
    pixel_bytes = 0
    for span in request.image_spans:
        offset = _u32(span.offset, "image span offset")
        tokens = _u32(span.tokens, "image span tokens")
        grid_height = _u32(span.grid_height, "image grid height")
        grid_width = _u32(span.grid_width, "image grid width")
        _u64(span.digest_lo, "image digest")
        _u64(span.digest_hi, "image digest")
        if (
            grid_height < 2
            or grid_width < 2
            or grid_height % 2
            or grid_width % 2
            or grid_height * grid_width > MAX_IMAGE_PATCHES
        ):
            raise ValueError("image grid must be even-sided and within the patch limit")
        if tokens != (grid_height // 2) * (grid_width // 2):
            raise ValueError("image span tokens must equal the merged grid size")
        if offset < previous_end or offset + tokens > prompt_tokens:
            raise ValueError(
                "image spans must be sorted, non-overlapping runs inside the prompt"
            )
        previous_end = offset + tokens
        pixel_bytes += span.pixel_bytes
    if len(_bytes(request.image_pixels, "image pixels")) != pixel_bytes:
        raise ValueError("image pixels do not match the image grids")


def _validated_request(
    request: RequestFrame,
) -> tuple[array.array, array.array, tuple]:
    """The request's prompt and score words as uint32 arrays and its float32
    sampling block, each checked once; raises a ProtocolError with the first
    rule the request breaks."""
    try:
        request_id = _u64(request.request_id, "request id")
        if not request_id:
            raise ValueError("request id must be non-zero")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_REQUEST_ID, str(error))
    try:
        _enum_value(request.priority, RequestPriority, "request priority")
        constraint = _enum_value(request.constraint, ConstraintMode, "constraint mode")
        flags = _u32(
            int(request.flags)
            if isinstance(request.flags, RequestFlag)
            else request.flags,
            "request flags",
        )
        if flags & ~_REQUEST_FLAG_BITS:
            raise ValueError("request flags are not defined by native protocol")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_ENUM_VALUE, str(error), request_id)
    try:
        absolute = _u64(request.absolute_deadline_unix_micros, "absolute deadline")
        remaining = _u64(request.remaining_deadline_micros, "remaining deadline")
        if not absolute or not remaining:
            raise ValueError("absolute and remaining deadlines must be non-zero")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_DEADLINE, str(error), request_id)
    try:
        output_tokens = _u32(request.logical_max_output_tokens, "logical max output")
        prompt = _words(request.prompt_tokens, "prompt tokens")
        scores = _words(request.score_tokens, "score tokens")
        if scores:
            if output_tokens:
                _fail_request(
                    IssueCode.INVALID_COUNT,
                    "score requests must not generate output tokens",
                    request_id,
                )
        elif not output_tokens or output_tokens > MAX_LOGICAL_OUTPUT_TOKENS:
            raise ValueError("logical max output token count exceeds its limit")
        if not prompt or len(prompt) > MAX_PROMPT_TOKENS:
            raise ValueError("prompt token count exceeds its limit")
        if len(request.image_spans) > MAX_IMAGE_SPANS:
            raise ValueError("image span count exceeds its limit")
    except ValueError as error:
        _fail_request(IssueCode.LIMIT_EXCEEDED, str(error), request_id)
    try:
        _image_spans_check(request, len(prompt))
        generation = _u32(request.generation_prompt_tokens, "generation prompt tokens")
        if generation >= len(prompt):
            raise ValueError("generation prompt must leave a prompt token")
        if scores and (request.image_spans or request.image_pixels):
            raise ValueError("score requests are text-only")
        if scores and (
            len(scores) < MIN_SCORE_TOKENS
            or len(scores) > MAX_SCORE_TOKENS
            or len(set(scores)) != len(scores)
        ):
            raise ValueError(
                f"score requests need {MIN_SCORE_TOKENS}..{MAX_SCORE_TOKENS} "
                "distinct option tokens"
            )
    except ValueError as error:
        _fail_request(IssueCode.INVALID_COUNT, str(error), request_id)
    try:
        sampling = _sampling_values(request.sampling)
        temperature, top_p, _, presence, frequency, repetition, min_p = sampling
        if (
            not (temperature == 0.0 or _FLOAT32_MIN <= temperature < math.inf)
            or not math.isfinite(top_p)
            or not 0.0 < top_p <= 1.0
            or not 0.0 <= min_p <= 1.0
        ):
            raise ValueError(
                "sampling requires temperature 0 or at least FLT_MIN, top_p in "
                "(0,1] and min_p in [0,1]"
            )
        if (
            not abs(presence) <= 2.0
            or not abs(frequency) <= 2.0
            or not math.isfinite(repetition)
            or repetition <= 0.0
        ):
            raise ValueError(
                "sampling requires presence and frequency penalties in "
                "[-2,2] and a positive repetition penalty"
            )
        if scores and sampling != _NEUTRAL_SAMPLING:
            raise ValueError("score requests require default greedy sampling")
    except (AttributeError, ValueError) as error:
        _fail_request(IssueCode.INVALID_SAMPLING, str(error), request_id)
    if scores and constraint is not ConstraintMode.NONE:
        _fail_request(
            IssueCode.INVALID_CONSTRAINT,
            "score requests do not accept output constraints",
            request_id,
        )
    # A grammar decides where constrained output ends.
    if flags & RequestFlag.IGNORE_END_OF_SEQUENCE and (
        scores or constraint is not ConstraintMode.NONE
    ):
        _fail_request(
            IssueCode.INVALID_CONSTRAINT,
            "only unconstrained generation can ignore end-of-sequence",
            request_id,
        )
    try:
        _u64(request.seed, "seed")
    except ValueError as error:
        _fail_request(IssueCode.INTEGER_OVERFLOW, str(error), request_id)
    return prompt, scores, sampling


def _mask_payload(values):
    if not isinstance(values, bytes):
        raise ValueError("mask words must be packed bytes")
    if len(values) % 4:
        raise ValueError("mask bytes must contain complete uint32 words")
    return values


def _frame(frame_type: FrameType, payload_bytes: int) -> bytearray:
    """A frame's header, in a buffer sized for the payload written after it."""
    frame = bytearray(FRAME_HEADER_BYTES + payload_bytes)
    _HEADER.pack_into(
        frame, 0, _MAGIC, PROTOCOL_VERSION, int(frame_type), payload_bytes
    )
    return frame


def _request_frame(request: RequestFrame) -> bytearray:
    """The whole request frame, header included, in one buffer that each part
    is copied into once: image pixels run to hundreds of MiB."""
    prompt, scores, sampling = _validated_request(request)
    prompt_offset = FRAME_HEADER_BYTES + _REQUEST.size
    spans_offset = prompt_offset + 4 * len(prompt)
    pixels_offset = spans_offset + _IMAGE_SPAN.size * len(request.image_spans)
    scores_offset = pixels_offset + len(request.image_pixels)
    payload_bytes = scores_offset + 4 * len(scores) - FRAME_HEADER_BYTES
    if issue := _payload_length_issue(FrameType.REQUEST, payload_bytes):
        _fail_request(issue.code, issue.message, request.request_id)
    frame = _frame(FrameType.REQUEST, payload_bytes)
    _REQUEST.pack_into(
        frame,
        FRAME_HEADER_BYTES,
        request.request_id,
        int(request.priority),
        int(request.constraint),
        request.absolute_deadline_unix_micros,
        request.remaining_deadline_micros,
        request.logical_max_output_tokens,
        len(request.prompt_tokens),
        len(request.image_spans),
        *sampling,
        request.seed,
        len(request.score_tokens),
        request.generation_prompt_tokens,
        request.flags,
    )
    for index, span in enumerate(request.image_spans):
        _IMAGE_SPAN.pack_into(
            frame,
            spans_offset + index * _IMAGE_SPAN.size,
            span.offset,
            span.tokens,
            span.grid_height,
            span.grid_width,
            span.digest_lo,
            span.digest_hi,
        )
    # A bytearray copies a slice assignment's source first; its view does not.
    with memoryview(frame) as view:
        view[prompt_offset:spans_offset] = memoryview(prompt).cast("B")
        view[pixels_offset:scores_offset] = request.image_pixels
        view[scores_offset:] = memoryview(scores).cast("B")
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


def _mask_response_frame(response: MaskResponseFrame) -> bytearray:
    request_id = 0
    try:
        request_id = _u64(response.request_id, "mask response request id")
        mask_request_id = _u64(response.mask_request_id, "mask request id")
        if not request_id or not mask_request_id:
            raise ValueError("mask response request ids must be non-zero")
    except ValueError as error:
        _fail_request(IssueCode.INVALID_REQUEST_ID, str(error), request_id)
    try:
        mask = _mask_payload(response.mask_words)
        if not mask or len(mask) // 4 > MAX_MASK_WORDS:
            raise ValueError("mask response word count exceeds its limit")
    except ValueError as error:
        _fail_request(IssueCode.LIMIT_EXCEEDED, str(error), request_id)
    frame = _frame(FrameType.MASK_RESPONSE, _MASK_RESPONSE.size + len(mask))
    _MASK_RESPONSE.pack_into(
        frame, FRAME_HEADER_BYTES, request_id, mask_request_id, len(mask) // 4
    )
    frame[FRAME_HEADER_BYTES + _MASK_RESPONSE.size :] = mask
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
    one buffer: a request's image pixels are copied once."""
    try:
        if isinstance(message, RequestFrame):
            return _request_frame(message)
        if isinstance(message, CancelFrame):
            return _cancel_frame(message)
        if isinstance(message, MaskResponseFrame):
            return _mask_response_frame(message)
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


def refresh_request_deadline(frame: bytes, now_unix_micros: int) -> bytes:
    """Re-stamp a REQUEST frame's absolute deadline as ``now`` plus its budget.

    Every other byte, and every other frame type, is left as it is. Only the
    integer head is unpacked: decoding and repacking a signaling NaN would
    change its bits. Invalid and incomplete frames may be the reason for the
    trace, so no validator runs and short fixed prefixes pass through.
    """
    if len(frame) < _HEADER.size + _REQUEST.size:
        return frame
    _, _, frame_type, _ = _HEADER.unpack_from(frame)
    if frame_type != FrameType.REQUEST:
        return frame
    request_id, priority, constraint, _, remaining = _REQUEST_HEAD.unpack_from(
        frame, _HEADER.size
    )
    head = _REQUEST_HEAD.pack(
        request_id,
        priority,
        constraint,
        min(now_unix_micros + remaining, 0xFFFFFFFFFFFFFFFF),
        remaining,
    )
    return frame[: _HEADER.size] + head + frame[_HEADER.size + _REQUEST_HEAD.size :]


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
    if frame_type is FrameType.PROMPT_PROGRESS:
        request_id, processed, elapsed = _PROMPT_PROGRESS.unpack(payload)
        if not request_id or processed > MAX_PROMPT_TOKENS:
            _fatal(
                IssueCode.INVALID_COUNT,
                "prompt progress id or token count is invalid",
                request_id,
            )
        return PromptProgressEvent(request_id, processed, elapsed)
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
    if frame_type is FrameType.MASK_REQUEST:
        request_id, mask_request_id, words_per_mask, count = _MASK_REQUEST.unpack_from(
            payload
        )
        tokens = _tail(
            payload, _MASK_REQUEST.size, count, "I", "simulation token", request_id
        )
        if not request_id or not mask_request_id:
            _fatal(
                IssueCode.INVALID_REQUEST_ID,
                "mask request ids must be non-zero",
                request_id,
            )
        if not words_per_mask:
            _fatal(
                IssueCode.INVALID_COUNT,
                "mask request dimensions are invalid",
                request_id,
            )
        if words_per_mask * (len(tokens) + 1) > MAX_MASK_WORDS:
            _fatal(
                IssueCode.LIMIT_EXCEEDED,
                "mask request output would exceed the mask limit",
                request_id,
            )
        return MaskRequestEvent(request_id, mask_request_id, words_per_mask, tokens)
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
        event = DoneEvent(request_id, reason, *counts, logits)
        if logits and (
            len(logits) < MIN_SCORE_TOKENS
            or not all(map(math.isfinite, logits))
            or reason is not FinishReason.STOP
            or event.completion_tokens
            or event.decode_micros
        ):
            _fatal(
                IssueCode.INVALID_COUNT,
                "scored done events need 2..255 finite logits, stop, and no "
                "completion or decode activity",
                request_id,
            )
        return event
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
