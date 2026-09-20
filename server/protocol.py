"""Strict Python codec for the Splash native protocol.

Wire framing and typed payload validation mirror ``runtime/engine/Protocol.*``.
"""

from __future__ import annotations

import array
import math
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import TypeAlias

PROTOCOL_VERSION = 6
FRAME_HEADER_BYTES = 24
STATUS_SCHEMA_VERSION = 5
# Largest top-k the native sampler keeps as candidates.
MAX_TOP_K = 32
# Score-only requests carry 2..255 distinct option token ids and produce no
# generated tokens; a successful score DoneEvent returns one raw
# final-position logit per requested token, in request order.
MIN_SCORE_TOKENS = 2
MAX_SCORE_TOKENS = 255
ABSOLUTE_MAX_FRAME_PAYLOAD_BYTES = 256 * 1024 * 1024

_MAGIC = b"SPLH"
_HEADER = struct.Struct("<4sHHHHQI")
# Replay can update the integer deadlines without decoding sampling floats.
_REQUEST_HEAD = struct.Struct("<QBBBQQ")
_REQUEST = struct.Struct(_REQUEST_HEAD.format + "IIIffIQBI")
_IMAGE_SPAN = struct.Struct("<IIIIQQ")
_CANCEL = struct.Struct("<Q")
_MASK_RESPONSE = struct.Struct("<QQI")
_STATUS_REQUEST = struct.Struct("<Q")
_READY = struct.Struct("<QIIQ")
_START = struct.Struct("<QBiII")
_PROMPT_PROGRESS = struct.Struct("<QIQ")
_TOKENS = struct.Struct("<QII")
_MASK_REQUEST = struct.Struct("<QQII")
_DONE = struct.Struct("<QBIIQQQ")
_ERROR = struct.Struct("<BBQII")
_CAPACITY_EXHAUSTED = struct.Struct("<QIIQ")
_STATUS_JSON = struct.Struct("<QI")

assert (
    array.array("I").itemsize == 4
    and array.array("f").itemsize == 4
    and sys.byteorder == "little"
)
assert _HEADER.size == FRAME_HEADER_BYTES
assert _REQUEST.size == 64
assert _IMAGE_SPAN.size == 32
assert _START.size == 21
assert _DONE.size == 41
assert _ERROR.size == 18

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
    CAPACITY_EXHAUSTED = 0x0106
    STATUS_JSON = 0x0107
    PROMPT_PROGRESS = 0x0108


class FailureClass(IntEnum):
    REQUEST_ERROR = 1
    ENGINE_UNHEALTHY = 2
    PROTOCOL_FATAL = 3


class IssueCode(IntEnum):
    NONE = 0
    BAD_MAGIC = 1
    UNSUPPORTED_VERSION = 2
    INVALID_HEADER_SIZE = 3
    UNKNOWN_FRAME_TYPE = 4
    NON_ZERO_HEADER_FLAGS = 5
    NON_ZERO_RESERVED_FIELD = 6
    FRAME_TOO_LARGE = 7
    INVALID_PAYLOAD_LENGTH = 8
    TRUNCATED_FRAME = 9
    PARSER_ALREADY_FAILED = 10
    INVALID_REQUEST_ID = 11
    INVALID_ENUM_VALUE = 12
    INVALID_DEADLINE = 13
    INVALID_SAMPLING = 14
    INVALID_COUNT = 15
    INVALID_COHORT_CONSTRAINT = 16
    INVALID_ERROR_CLASSIFICATION = 17
    INVALID_STATUS_SCHEMA = 18
    LIMIT_EXCEEDED = 19
    INTEGER_OVERFLOW = 20
    ALLOCATION_FAILURE = 21


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
    request_id: int = 0
    message: str = ""

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
class ProtocolLimits:
    max_frame_payload_bytes: int = ABSOLUTE_MAX_FRAME_PAYLOAD_BYTES
    max_status_json_bytes: int = 32 * 1024 * 1024
    max_error_string_bytes: int = 1 * 1024 * 1024
    max_prompt_tokens: int = 1 << 20
    max_logical_output_tokens: int = 1 << 20
    max_token_batch: int = 4096
    max_simulation_tokens: int = 32
    max_image_spans: int = 64
    max_image_patches: int = 16384
    max_mask_words: int = 1 << 20


class RequestPriority(IntEnum):
    FOREGROUND = 0
    NORMAL = 1
    BACKGROUND = 2


class Cohort(IntEnum):
    GREEDY = 0
    SAMPLING = 1
    CONSTRAINED = 2


class ConstraintMode(IntEnum):
    NONE = 0
    TOKEN_MASK = 1


@dataclass(slots=True, frozen=True)
class SamplingParameters:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0


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
    cohort: Cohort
    constraint: ConstraintMode
    # Sorted, non-overlapping image spans and their resized uint8 RGB pixels
    # concatenated in span order; both empty for text-only requests.
    image_spans: tuple[ImageSpan, ...] = ()
    image_pixels: bytes = b""
    return_progress: bool = False
    # Option token ids for score-only requests; empty means ordinary
    # generation. Score tokens serialize after the image pixel bytes.
    score_tokens: tuple[int, ...] = ()


@dataclass(slots=True, frozen=True)
class CancelFrame:
    request_id: int


@dataclass(slots=True, frozen=True)
class MaskResponseFrame:
    request_id: int
    mask_request_id: int
    mask_words: tuple[int, ...] | bytes


@dataclass(slots=True, frozen=True)
class StatusRequestFrame:
    correlation_id: int


class ReadyFeature(IntFlag):
    CANCELLATION = 1 << 0
    TOKEN_MASKS = 1 << 1
    STATUS_JSON = 1 << 2
    MULTIPLEXING = 1 << 3


@dataclass(slots=True, frozen=True)
class ReadyEvent:
    engine_instance_id: int
    max_concurrent_requests: int
    max_context_tokens: int
    feature_bits: int | ReadyFeature


class CacheDisposition(IntEnum):
    MISS = 0
    PREFIX_HIT = 1


@dataclass(slots=True, frozen=True)
class StartEvent:
    request_id: int
    cache_disposition: CacheDisposition
    slot_index: int
    matched_prompt_tokens: int
    capacity_tokens: int


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
    option_logits: tuple[float, ...] = ()


@dataclass(slots=True, frozen=True)
class ErrorEvent:
    failure_class: FailureClass
    request_id: int
    retryable: bool
    code: bytes
    message: bytes


@dataclass(slots=True, frozen=True)
class CapacityExhaustedEvent:
    request_id: int
    required_kv_pages: int
    available_kv_pages: int
    retry_after_micros: int


@dataclass(slots=True, frozen=True)
class StatusJsonEvent:
    correlation_id: int
    schema_version: int
    json: bytes


Message: TypeAlias = (
    RequestFrame
    | CancelFrame
    | MaskResponseFrame
    | StatusRequestFrame
    | ReadyEvent
    | StartEvent
    | PromptProgressEvent
    | TokensEvent
    | MaskRequestEvent
    | DoneEvent
    | ErrorEvent
    | CapacityExhaustedEvent
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


def _i32(value: object, label: str) -> int:
    return _integer(value, -(1 << 31), (1 << 31) - 1, label)


def _float32(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a float32 value")
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{label} must be a float32 value") from error


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


def _words(values: object, label: str) -> tuple[int, ...]:
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple of uint32 values")
    return tuple(_u32(value, f"{label} element") for value in values)


def _limits_issue(limits: ProtocolLimits) -> ProtocolIssue | None:
    try:
        max_frame = _u64(limits.max_frame_payload_bytes, "max frame payload")
        max_status = _u64(limits.max_status_json_bytes, "max status JSON")
        max_error = _u64(limits.max_error_string_bytes, "max error string")
        counts = (
            _u32(limits.max_prompt_tokens, "max prompt tokens"),
            _u32(limits.max_logical_output_tokens, "max output tokens"),
            _u32(limits.max_token_batch, "max token batch"),
            _u32(limits.max_simulation_tokens, "max simulation tokens"),
            _u32(limits.max_mask_words, "max mask words"),
            _u32(limits.max_image_spans, "max image spans"),
            _u32(limits.max_image_patches, "max image patches"),
        )
    except (AttributeError, ValueError) as error:
        return _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.LIMIT_EXCEEDED,
            f"invalid protocol limits: {error}",
        )
    if not _REQUEST.size <= max_frame <= ABSOLUTE_MAX_FRAME_PAYLOAD_BYTES:
        return _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.LIMIT_EXCEEDED,
            f"maxFramePayloadBytes must be in [{_REQUEST.size}, 256 MiB]",
        )
    if max_status > max_frame - _STATUS_JSON.size:
        return _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.LIMIT_EXCEEDED,
            "status JSON limit does not fit the configured frame limit",
        )
    if max_error > max_frame:
        return _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.LIMIT_EXCEEDED,
            "error string limit exceeds the configured frame limit",
        )
    if not all(counts):
        return _issue(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.LIMIT_EXCEEDED,
            "all configured token and mask limits must be non-zero",
        )
    return None


def _check_limits(limits: ProtocolLimits) -> None:
    if issue := _limits_issue(limits):
        raise ProtocolError(issue)


def _payload_bounds(frame_type: FrameType, limits: ProtocolLimits) -> tuple[int, int]:
    match frame_type:
        case FrameType.REQUEST:
            # Image pixels dominate prompt tokens; the frame limit is the bound.
            bounds = (_REQUEST.size, limits.max_frame_payload_bytes)
        case FrameType.CANCEL:
            bounds = (_CANCEL.size, _CANCEL.size)
        case FrameType.MASK_RESPONSE:
            bounds = (
                _MASK_RESPONSE.size,
                _MASK_RESPONSE.size + limits.max_mask_words * 4,
            )
        case FrameType.STATUS_REQUEST:
            bounds = (_STATUS_REQUEST.size, _STATUS_REQUEST.size)
        case FrameType.READY:
            bounds = (_READY.size, _READY.size)
        case FrameType.PROMPT_PROGRESS:
            bounds = (_PROMPT_PROGRESS.size, _PROMPT_PROGRESS.size)
        case FrameType.START:
            bounds = (_START.size, _START.size)
        case FrameType.TOKENS:
            bounds = (_TOKENS.size, _TOKENS.size + limits.max_token_batch * 4)
        case FrameType.MASK_REQUEST:
            bounds = (
                _MASK_REQUEST.size,
                _MASK_REQUEST.size + limits.max_simulation_tokens * 4,
            )
        case FrameType.DONE:
            bounds = (
                _DONE.size + 4,
                _DONE.size + 4 + MAX_SCORE_TOKENS * 4,
            )
        case FrameType.ERROR:
            bounds = (
                _ERROR.size,
                _ERROR.size + limits.max_error_string_bytes * 2,
            )
        case FrameType.CAPACITY_EXHAUSTED:
            bounds = (_CAPACITY_EXHAUSTED.size, _CAPACITY_EXHAUSTED.size)
        case FrameType.STATUS_JSON:
            bounds = (
                _STATUS_JSON.size,
                _STATUS_JSON.size + limits.max_status_json_bytes,
            )
    return bounds[0], min(bounds[1], limits.max_frame_payload_bytes)


def _payload_length_issue(
    frame_type: FrameType,
    payload_bytes: int,
    limits: ProtocolLimits,
) -> ProtocolIssue | None:
    minimum, maximum = _payload_bounds(frame_type, limits)
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


def _pack_words(values: tuple[int, ...]) -> bytes:
    if not values:
        return b""
    return array.array("I", values).tobytes()


def _unpack_words(payload: bytes, offset: int, count: int) -> tuple[int, ...]:
    if len(payload) - offset != count * 4:
        raise ValueError("word count does not match binary payload")
    return _unpack_prefix_words(payload, offset, count)


def _unpack_prefix_words(payload: bytes, offset: int, count: int) -> tuple[int, ...]:
    byte_count = count * 4
    if len(payload) - offset < byte_count:
        raise ValueError("word count does not match binary payload")
    if not count:
        return ()
    words = array.array("I")
    words.frombytes(payload[offset : offset + byte_count])
    return tuple(words)


def _pack_floats(values: tuple[float, ...]) -> bytes:
    if not values:
        return b""
    return array.array("f", values).tobytes()


def _unpack_floats(payload: bytes, offset: int, count: int) -> tuple[float, ...]:
    byte_count = count * 4
    if len(payload) - offset != byte_count:
        raise ValueError("float count does not match binary payload")
    if not count:
        return ()
    floats = array.array("f")
    floats.frombytes(payload[offset : offset + byte_count])
    return tuple(floats)


def _score_logits(values: object, label: str) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple of finite float32 values")
    logits = tuple(_float32(value, f"{label} element") for value in values)
    if any(not math.isfinite(value) for value in logits):
        raise ValueError(f"{label} must be finite")
    if logits and not MIN_SCORE_TOKENS <= len(logits) <= MAX_SCORE_TOKENS:
        raise ValueError(
            f"{label} must be empty or {MIN_SCORE_TOKENS}..{MAX_SCORE_TOKENS} values"
        )
    return logits


def _image_spans_check(request: RequestFrame, prompt_tokens: int, limits) -> None:
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
            or grid_height * grid_width > limits.max_image_patches
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


def _request_issue(
    request: RequestFrame, limits: ProtocolLimits
) -> ProtocolIssue | None:
    request_id = request.request_id if type(request.request_id) is int else 0
    try:
        request_id = _u64(request.request_id, "request id")
        if not request_id:
            raise ValueError("request id must be non-zero")
    except ValueError as error:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_REQUEST_ID,
            str(error),
            request_id,
        )
    try:
        if not isinstance(request.return_progress, bool):
            raise ValueError("return_progress must be a boolean")
        _enum_value(request.priority, RequestPriority, "request priority")
        cohort = _enum_value(request.cohort, Cohort, "cohort")
        constraint = _enum_value(request.constraint, ConstraintMode, "constraint mode")
    except ValueError as error:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_ENUM_VALUE,
            str(error),
            request_id,
        )
    try:
        absolute = _u64(request.absolute_deadline_unix_micros, "absolute deadline")
        remaining = _u64(request.remaining_deadline_micros, "remaining deadline")
        if not absolute or not remaining:
            raise ValueError("absolute and remaining deadlines must be non-zero")
    except ValueError as error:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_DEADLINE,
            str(error),
            request_id,
        )
    try:
        output_tokens = _u32(request.logical_max_output_tokens, "logical max output")
        prompt = _words(request.prompt_tokens, "prompt tokens")
        scores = _words(request.score_tokens, "score tokens")
        if scores:
            if output_tokens:
                return _issue(
                    FailureClass.REQUEST_ERROR,
                    IssueCode.INVALID_COUNT,
                    "score requests must not generate output tokens",
                    request_id,
                )
        elif not output_tokens or output_tokens > limits.max_logical_output_tokens:
            raise ValueError("logical max output token count exceeds its limit")
        if not prompt or len(prompt) > limits.max_prompt_tokens:
            raise ValueError("prompt token count exceeds its limit")
        if len(request.image_spans) > limits.max_image_spans:
            raise ValueError("image span count exceeds its limit")
    except ValueError as error:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.LIMIT_EXCEEDED,
            str(error),
            request_id,
        )
    try:
        _image_spans_check(request, len(prompt), limits)
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
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_COUNT,
            str(error),
            request_id,
        )
    try:
        temperature = _float32(request.sampling.temperature, "temperature")
        top_p = _float32(request.sampling.top_p, "top_p")
        top_k = _u32(request.sampling.top_k, "top_k")
        if (
            not math.isfinite(temperature)
            or temperature < 0.0
            or not math.isfinite(top_p)
            or not 0.0 < top_p <= 1.0
            or top_k > MAX_TOP_K
            or (temperature > 0.0 and not top_k)
        ):
            raise ValueError(
                "sampling requires temperature>=0, top_p in (0,1], and "
                "top_k in [1,32] when sampling is enabled"
            )
        if scores and (temperature != 0.0 or top_p != 1.0 or top_k):
            raise ValueError("score requests require default greedy sampling")
    except (AttributeError, ValueError) as error:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_SAMPLING,
            str(error),
            request_id,
        )
    if scores and constraint is not ConstraintMode.NONE:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_COHORT_CONSTRAINT,
            "score requests do not accept output constraints",
            request_id,
        )
    expected = Cohort.CONSTRAINED
    if constraint is ConstraintMode.NONE:
        expected = Cohort.SAMPLING if temperature > 0.0 else Cohort.GREEDY
    if cohort is not expected:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_COHORT_CONSTRAINT,
            "cohort does not match sampling and constraint semantics",
            request_id,
        )
    try:
        _u64(request.seed, "seed")
    except ValueError as error:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INTEGER_OVERFLOW,
            str(error),
            request_id,
        )
    return None


def _cancel_issue(cancel: CancelFrame) -> ProtocolIssue | None:
    try:
        request_id = _u64(cancel.request_id, "cancel request id")
    except ValueError as error:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_REQUEST_ID,
            str(error),
        )
    if request_id:
        return None
    return _issue(
        FailureClass.REQUEST_ERROR,
        IssueCode.INVALID_REQUEST_ID,
        "cancel request id must be non-zero",
    )


def _mask_payload(values):
    if isinstance(values, bytes):
        if len(values) % 4:
            raise ValueError("mask bytes must contain complete uint32 words")
        return values
    return _pack_words(_words(values, "mask words"))


def _mask_response_issue(
    response: MaskResponseFrame,
    limits: ProtocolLimits,
) -> ProtocolIssue | None:
    request_id = response.request_id if type(response.request_id) is int else 0
    try:
        request_id = _u64(response.request_id, "mask response request id")
        mask_request_id = _u64(response.mask_request_id, "mask request id")
        if not request_id or not mask_request_id:
            raise ValueError("mask response request ids must be non-zero")
    except ValueError as error:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_REQUEST_ID,
            str(error),
            request_id,
        )
    try:
        payload = _mask_payload(response.mask_words)
    except ValueError as error:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.LIMIT_EXCEEDED,
            str(error),
            request_id,
        )
    if not payload or len(payload) // 4 > limits.max_mask_words:
        return _issue(
            FailureClass.REQUEST_ERROR,
            IssueCode.LIMIT_EXCEEDED,
            "mask response word count exceeds its limit",
            request_id,
        )
    return None


def _ready_issue(event: ReadyEvent, failure: FailureClass) -> ProtocolIssue | None:
    try:
        instance = _u64(event.engine_instance_id, "engine instance id")
        concurrent = _u32(event.max_concurrent_requests, "max concurrent requests")
        context = _u32(event.max_context_tokens, "max context tokens")
        feature_bits = (
            int(event.feature_bits)
            if isinstance(event.feature_bits, ReadyFeature)
            else event.feature_bits
        )
        _u64(feature_bits, "feature bits")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_COUNT, str(error))
    if instance and concurrent and context:
        return None
    return _issue(
        failure,
        IssueCode.INVALID_COUNT,
        "ready event capacities and instance id must be non-zero",
    )


def _start_issue(event: StartEvent, failure: FailureClass) -> ProtocolIssue | None:
    request_id = event.request_id if type(event.request_id) is int else 0
    try:
        request_id = _u64(event.request_id, "start request id")
        if not request_id:
            raise ValueError("start event request id must be non-zero")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_REQUEST_ID, str(error), request_id)
    try:
        _enum_value(event.cache_disposition, CacheDisposition, "cache disposition")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_ENUM_VALUE, str(error), request_id)
    try:
        slot = _i32(event.slot_index, "slot index")
        matched = _u32(event.matched_prompt_tokens, "matched prompt tokens")
        capacity = _u32(event.capacity_tokens, "capacity tokens")
        if slot < -1 or not capacity or matched > capacity:
            raise ValueError("start event capacity or slot is invalid")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_COUNT, str(error), request_id)
    return None


def _progress_issue(
    event: PromptProgressEvent, limits: ProtocolLimits, failure: FailureClass
) -> ProtocolIssue | None:
    request_id = event.request_id if type(event.request_id) is int else 0
    try:
        if not _u64(event.request_id, "progress request id"):
            raise ValueError("prompt progress id must be non-zero")
        if _u32(event.processed_tokens, "processed tokens") > limits.max_prompt_tokens:
            raise ValueError("prompt progress token count exceeds its limit")
        _u64(event.elapsed_micros, "progress elapsed microseconds")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_COUNT, str(error), request_id)
    return None


def _tokens_issue(
    event: TokensEvent,
    limits: ProtocolLimits,
    failure: FailureClass,
) -> ProtocolIssue | None:
    request_id = event.request_id if type(event.request_id) is int else 0
    try:
        request_id = _u64(event.request_id, "tokens request id")
        if not request_id:
            raise ValueError("tokens event request id must be non-zero")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_REQUEST_ID, str(error), request_id)
    try:
        offset = _u32(event.sequence_offset, "sequence offset")
        tokens = _words(event.tokens, "tokens")
        if not tokens or len(tokens) > limits.max_token_batch:
            raise ValueError("tokens event batch size exceeds its limit")
    except ValueError as error:
        return _issue(failure, IssueCode.LIMIT_EXCEEDED, str(error), request_id)
    if offset + len(tokens) > 0xFFFFFFFF:
        return _issue(
            failure,
            IssueCode.INTEGER_OVERFLOW,
            "tokens event sequence range overflows uint32",
            request_id,
        )
    return None


def _mask_request_issue(
    event: MaskRequestEvent,
    limits: ProtocolLimits,
    failure: FailureClass,
) -> ProtocolIssue | None:
    request_id = event.request_id if type(event.request_id) is int else 0
    try:
        request_id = _u64(event.request_id, "mask request request id")
        mask_request_id = _u64(event.mask_request_id, "mask request id")
        if not request_id or not mask_request_id:
            raise ValueError("mask request ids must be non-zero")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_REQUEST_ID, str(error), request_id)
    try:
        words_per_mask = _u32(event.words_per_mask, "words per mask")
        tokens = _words(event.simulation_tokens, "simulation tokens")
        if not words_per_mask or len(tokens) > limits.max_simulation_tokens:
            raise ValueError("mask request dimensions are invalid")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_COUNT, str(error), request_id)
    if words_per_mask * (len(tokens) + 1) > limits.max_mask_words:
        return _issue(
            failure,
            IssueCode.LIMIT_EXCEEDED,
            "mask request output would exceed the mask limit",
            request_id,
        )
    return None


def _done_issue(event: DoneEvent, failure: FailureClass) -> ProtocolIssue | None:
    request_id = event.request_id if type(event.request_id) is int else 0
    try:
        request_id = _u64(event.request_id, "done request id")
        if not request_id:
            raise ValueError("done event request id must be non-zero")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_REQUEST_ID, str(error), request_id)
    try:
        reason = _enum_value(event.reason, FinishReason, "finish reason")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_ENUM_VALUE, str(error), request_id)
    try:
        _u32(event.prompt_tokens, "done prompt tokens")
        _u32(event.completion_tokens, "done completion tokens")
        _u64(event.prefill_micros, "prefill microseconds")
        _u64(event.decode_micros, "decode microseconds")
        _u64(event.wall_micros, "wall microseconds")
        logits = _score_logits(event.option_logits, "option logits")
        if logits:
            if reason is not FinishReason.STOP:
                raise ValueError("scored done events must finish with stop")
            if event.completion_tokens or event.decode_micros:
                raise ValueError(
                    "scored done events carry no completion or decode activity"
                )
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_COUNT, str(error), request_id)
    return None


def _error_issue(
    event: ErrorEvent,
    limits: ProtocolLimits,
    failure: FailureClass,
) -> ProtocolIssue | None:
    request_id = event.request_id if type(event.request_id) is int else 0
    try:
        classification = _enum_value(
            event.failure_class, FailureClass, "error classification"
        )
        request_id = _u64(event.request_id, "error request id")
        if type(event.retryable) is not bool:
            raise ValueError("error retryable field must be boolean")
        if (classification is FailureClass.REQUEST_ERROR and not request_id) or (
            classification is not FailureClass.REQUEST_ERROR and request_id
        ):
            raise ValueError("only request errors may carry a non-zero request id")
    except ValueError as error:
        return _issue(
            failure,
            IssueCode.INVALID_ERROR_CLASSIFICATION,
            str(error),
            request_id,
        )
    try:
        code = _bytes(event.code, "error code")
        message = _bytes(event.message, "error message")
    except ValueError as error:
        return _issue(failure, IssueCode.LIMIT_EXCEEDED, str(error), request_id)
    if (
        not code
        or len(code) > limits.max_error_string_bytes
        or len(message) > limits.max_error_string_bytes
    ):
        return _issue(
            failure,
            IssueCode.LIMIT_EXCEEDED,
            "error code or message exceeds its safe limit",
            request_id,
        )
    return None


def _capacity_issue(
    event: CapacityExhaustedEvent,
    failure: FailureClass,
) -> ProtocolIssue | None:
    request_id = event.request_id if type(event.request_id) is int else 0
    try:
        request_id = _u64(event.request_id, "capacity request id")
        if not request_id:
            raise ValueError("capacity event request id must be non-zero")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_REQUEST_ID, str(error), request_id)
    try:
        required = _u32(event.required_kv_pages, "required KV pages")
        _u32(event.available_kv_pages, "available KV pages")
        _u64(event.retry_after_micros, "retry delay")
        if not required:
            raise ValueError("capacity event required page count must be non-zero")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_COUNT, str(error), request_id)
    return None


def _status_issue(
    event: StatusJsonEvent,
    limits: ProtocolLimits,
    failure: FailureClass,
) -> ProtocolIssue | None:
    try:
        _u64(event.correlation_id, "status correlation id")
        schema = _u32(event.schema_version, "status schema version")
    except ValueError as error:
        return _issue(failure, IssueCode.INVALID_STATUS_SCHEMA, str(error))
    if schema != STATUS_SCHEMA_VERSION:
        return _issue(
            failure,
            IssueCode.INVALID_STATUS_SCHEMA,
            "status JSON schema version must match the native protocol",
        )
    try:
        payload = _bytes(event.json, "status JSON")
    except ValueError as error:
        return _issue(failure, IssueCode.LIMIT_EXCEEDED, str(error))
    if not payload or len(payload) > limits.max_status_json_bytes:
        return _issue(
            failure,
            IssueCode.LIMIT_EXCEEDED,
            "status JSON byte count exceeds its safe limit",
        )
    return None


def _raise_issue(issue: ProtocolIssue | None) -> None:
    if issue:
        raise ProtocolError(issue)


def _encode_message(
    message: Message, limits: ProtocolLimits = ProtocolLimits()
) -> Frame:
    _check_limits(limits)
    frame_type: FrameType
    payload: bytes
    if isinstance(message, RequestFrame):
        _raise_issue(_request_issue(message, limits))
        prompt = _words(message.prompt_tokens, "prompt tokens")
        scores = _words(message.score_tokens, "score tokens")
        top_p = _float32(message.sampling.top_p, "top_p")
        temperature = _float32(message.sampling.temperature, "temperature")
        payload = (
            _REQUEST.pack(
                message.request_id,
                int(message.priority),
                int(message.cohort),
                int(message.constraint),
                message.absolute_deadline_unix_micros,
                message.remaining_deadline_micros,
                message.logical_max_output_tokens,
                len(prompt),
                len(message.image_spans),
                temperature,
                top_p,
                message.sampling.top_k,
                message.seed,
                message.return_progress,
                len(scores),
            )
            + _pack_words(prompt)
            + b"".join(
                _IMAGE_SPAN.pack(
                    span.offset,
                    span.tokens,
                    span.grid_height,
                    span.grid_width,
                    span.digest_lo,
                    span.digest_hi,
                )
                for span in message.image_spans
            )
            + bytes(message.image_pixels)
            + _pack_words(scores)
        )
        frame_type = FrameType.REQUEST
    elif isinstance(message, CancelFrame):
        _raise_issue(_cancel_issue(message))
        payload = _CANCEL.pack(message.request_id)
        frame_type = FrameType.CANCEL
    elif isinstance(message, MaskResponseFrame):
        _raise_issue(_mask_response_issue(message, limits))
        mask = _mask_payload(message.mask_words)
        payload = (
            _MASK_RESPONSE.pack(
                message.request_id, message.mask_request_id, len(mask) // 4
            )
            + mask
        )
        frame_type = FrameType.MASK_RESPONSE
    elif isinstance(message, StatusRequestFrame):
        try:
            correlation_id = _u64(message.correlation_id, "status correlation id")
        except ValueError as error:
            _fail(
                FailureClass.REQUEST_ERROR,
                IssueCode.INTEGER_OVERFLOW,
                str(error),
            )
        payload = _STATUS_REQUEST.pack(correlation_id)
        frame_type = FrameType.STATUS_REQUEST
    elif isinstance(message, ReadyEvent):
        _raise_issue(_ready_issue(message, FailureClass.ENGINE_UNHEALTHY))
        payload = _READY.pack(
            message.engine_instance_id,
            message.max_concurrent_requests,
            message.max_context_tokens,
            message.feature_bits,
        )
        frame_type = FrameType.READY
    elif isinstance(message, StartEvent):
        _raise_issue(_start_issue(message, FailureClass.ENGINE_UNHEALTHY))
        payload = _START.pack(
            message.request_id,
            int(message.cache_disposition),
            message.slot_index,
            message.matched_prompt_tokens,
            message.capacity_tokens,
        )
        frame_type = FrameType.START
    elif isinstance(message, PromptProgressEvent):
        _raise_issue(_progress_issue(message, limits, FailureClass.ENGINE_UNHEALTHY))
        payload = _PROMPT_PROGRESS.pack(
            message.request_id, message.processed_tokens, message.elapsed_micros
        )
        frame_type = FrameType.PROMPT_PROGRESS
    elif isinstance(message, TokensEvent):
        _raise_issue(_tokens_issue(message, limits, FailureClass.ENGINE_UNHEALTHY))
        tokens = _words(message.tokens, "tokens")
        payload = _TOKENS.pack(
            message.request_id, message.sequence_offset, len(tokens)
        ) + _pack_words(tokens)
        frame_type = FrameType.TOKENS
    elif isinstance(message, MaskRequestEvent):
        _raise_issue(
            _mask_request_issue(message, limits, FailureClass.ENGINE_UNHEALTHY)
        )
        tokens = _words(message.simulation_tokens, "simulation tokens")
        payload = _MASK_REQUEST.pack(
            message.request_id,
            message.mask_request_id,
            message.words_per_mask,
            len(tokens),
        ) + _pack_words(tokens)
        frame_type = FrameType.MASK_REQUEST
    elif isinstance(message, DoneEvent):
        _raise_issue(_done_issue(message, FailureClass.ENGINE_UNHEALTHY))
        payload = (
            _DONE.pack(
                message.request_id,
                int(message.reason),
                message.prompt_tokens,
                message.completion_tokens,
                message.prefill_micros,
                message.decode_micros,
                message.wall_micros,
            )
            + struct.pack("<I", len(message.option_logits))
            + _pack_floats(message.option_logits)
        )
        frame_type = FrameType.DONE
    elif isinstance(message, ErrorEvent):
        _raise_issue(_error_issue(message, limits, FailureClass.ENGINE_UNHEALTHY))
        payload = (
            _ERROR.pack(
                int(message.failure_class),
                int(message.retryable),
                message.request_id,
                len(message.code),
                len(message.message),
            )
            + message.code
            + message.message
        )
        frame_type = FrameType.ERROR
    elif isinstance(message, CapacityExhaustedEvent):
        _raise_issue(_capacity_issue(message, FailureClass.ENGINE_UNHEALTHY))
        payload = _CAPACITY_EXHAUSTED.pack(
            message.request_id,
            message.required_kv_pages,
            message.available_kv_pages,
            message.retry_after_micros,
        )
        frame_type = FrameType.CAPACITY_EXHAUSTED
    elif isinstance(message, StatusJsonEvent):
        _raise_issue(_status_issue(message, limits, FailureClass.ENGINE_UNHEALTHY))
        payload = (
            _STATUS_JSON.pack(message.correlation_id, message.schema_version)
            + message.json
        )
        frame_type = FrameType.STATUS_JSON
    else:
        _fail(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.UNKNOWN_FRAME_TYPE,
            "message type is not defined by native protocol",
        )

    if issue := _payload_length_issue(frame_type, len(payload), limits):
        client = frame_type in {
            FrameType.REQUEST,
            FrameType.CANCEL,
            FrameType.MASK_RESPONSE,
            FrameType.STATUS_REQUEST,
        }
        request_id = getattr(message, "request_id", 0)
        raise ProtocolError(
            ProtocolIssue(
                FailureClass.REQUEST_ERROR if client else FailureClass.ENGINE_UNHEALTHY,
                issue.code,
                request_id,
                issue.message,
            )
        )
    return Frame(frame_type, payload)


def encode_message(
    message: Message, limits: ProtocolLimits = ProtocolLimits()
) -> Frame:
    """Validate and encode one typed runtime message into a payload frame."""

    try:
        return _encode_message(message, limits)
    except MemoryError as error:
        raise ProtocolError(
            _allocation_issue("allocation failed while encoding protocol message")
        ) from error


def _checked_frame(frame: Frame, limits: ProtocolLimits) -> tuple[FrameType, bytes]:
    _check_limits(limits)
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
        _fail(
            FailureClass.PROTOCOL_FATAL,
            IssueCode.INVALID_PAYLOAD_LENGTH,
            str(error),
        )
    _raise_issue(_payload_length_issue(frame_type, len(payload), limits))
    return frame_type, payload


def serialize_frame(frame: Frame, limits: ProtocolLimits = ProtocolLimits()) -> bytes:
    """Serialize a validated low-level frame with a strict runtime header."""

    frame_type, payload = _checked_frame(frame, limits)
    try:
        return (
            _HEADER.pack(
                _MAGIC,
                PROTOCOL_VERSION,
                FRAME_HEADER_BYTES,
                int(frame_type),
                0,
                len(payload),
                0,
            )
            + payload
        )
    except MemoryError as error:
        raise ProtocolError(
            _allocation_issue("allocation failed while serializing protocol frame")
        ) from error


def serialize_message(
    message: Message,
    limits: ProtocolLimits = ProtocolLimits(),
) -> bytes:
    return serialize_frame(encode_message(message, limits), limits)


def refresh_request_deadline(frame: bytes, now_unix_micros: int) -> bytes:
    """Re-stamp a REQUEST frame's absolute deadline as ``now`` plus its budget.

    Every other byte, and every other frame type, is left as it is. Only the
    integer head is unpacked: decoding and repacking a signaling NaN would
    change its bits. Invalid and incomplete frames may be the reason for the
    trace, so no validator runs and short fixed prefixes pass through.
    """
    if len(frame) < _HEADER.size + _REQUEST.size:
        return frame
    _, _, _, frame_type, _, _, _ = _HEADER.unpack_from(frame)
    if frame_type != FrameType.REQUEST:
        return frame
    request_id, priority, cohort, constraint, _, remaining = _REQUEST_HEAD.unpack_from(
        frame, _HEADER.size
    )
    head = _REQUEST_HEAD.pack(
        request_id,
        priority,
        cohort,
        constraint,
        min(now_unix_micros + remaining, 0xFFFFFFFFFFFFFFFF),
        remaining,
    )
    return frame[: _HEADER.size] + head + frame[_HEADER.size + _REQUEST_HEAD.size :]


def _decode_enum(
    value: int,
    enum_type: type[IntEnum],
    label: str,
    failure: FailureClass,
    request_id: int = 0,
) -> IntEnum:
    try:
        return _enum_value(value, enum_type, label)
    except ValueError as error:
        _fail(failure, IssueCode.INVALID_ENUM_VALUE, str(error), request_id)


def _decode_request(payload: bytes, limits: ProtocolLimits) -> RequestFrame:
    (
        request_id,
        priority,
        cohort,
        constraint,
        absolute_deadline,
        remaining_deadline,
        max_output,
        prompt_count,
        image_span_count,
        temperature,
        top_p,
        top_k,
        seed,
        return_progress,
        score_count,
    ) = _REQUEST.unpack_from(payload)
    if return_progress > 1:
        _fail(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_ENUM_VALUE,
            "return_progress must be a boolean",
            request_id,
        )
    if score_count > MAX_SCORE_TOKENS:
        _fail(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_COUNT,
            "score option count exceeds its limit",
            request_id,
        )
    if prompt_count > limits.max_prompt_tokens:
        _fail(
            FailureClass.REQUEST_ERROR,
            IssueCode.LIMIT_EXCEEDED,
            "prompt token count exceeds its limit",
            request_id,
        )
    if image_span_count > limits.max_image_spans:
        _fail(
            FailureClass.REQUEST_ERROR,
            IssueCode.LIMIT_EXCEEDED,
            "image span count exceeds its limit",
            request_id,
        )
    try:
        prompt = _unpack_prefix_words(payload, _REQUEST.size, prompt_count)
        cursor = _REQUEST.size + 4 * prompt_count
        image_spans = []
        for _ in range(image_span_count):
            image_spans.append(ImageSpan(*_IMAGE_SPAN.unpack_from(payload, cursor)))
            cursor += _IMAGE_SPAN.size
        score_bytes = 4 * score_count
        score_offset = len(payload) - score_bytes
        if score_offset < cursor:
            raise ValueError("score tokens")
        image_pixels = payload[cursor:score_offset]
        if len(image_pixels) != sum(span.pixel_bytes for span in image_spans):
            raise ValueError("image pixels")
        score_tokens = _unpack_prefix_words(payload, score_offset, score_count)
    except (ValueError, struct.error):
        _fail(
            FailureClass.REQUEST_ERROR,
            IssueCode.INVALID_PAYLOAD_LENGTH,
            "prompt count does not match the binary token payload",
            request_id,
        )
    request = RequestFrame(
        request_id,
        _decode_enum(
            priority,
            RequestPriority,
            "request priority",
            FailureClass.REQUEST_ERROR,
            request_id,
        ),
        absolute_deadline,
        remaining_deadline,
        max_output,
        prompt,
        SamplingParameters(temperature, top_p, top_k),
        seed,
        _decode_enum(
            cohort,
            Cohort,
            "cohort",
            FailureClass.REQUEST_ERROR,
            request_id,
        ),
        _decode_enum(
            constraint,
            ConstraintMode,
            "constraint mode",
            FailureClass.REQUEST_ERROR,
            request_id,
        ),
        tuple(image_spans),
        bytes(image_pixels),
        bool(return_progress),
        score_tokens,
    )
    _raise_issue(_request_issue(request, limits))
    return request


def _decode_frame(frame: Frame, limits: ProtocolLimits = ProtocolLimits()) -> Message:
    frame_type, payload = _checked_frame(frame, limits)
    if frame_type is FrameType.REQUEST:
        return _decode_request(payload, limits)
    if frame_type is FrameType.CANCEL:
        message = CancelFrame(_CANCEL.unpack(payload)[0])
        _raise_issue(_cancel_issue(message))
        return message
    if frame_type is FrameType.MASK_RESPONSE:
        request_id, mask_request_id, count = _MASK_RESPONSE.unpack_from(payload)
        if count > limits.max_mask_words:
            _fail(
                FailureClass.REQUEST_ERROR,
                IssueCode.LIMIT_EXCEEDED,
                "mask response word count exceeds its limit",
                request_id,
            )
        try:
            words = _unpack_words(payload, _MASK_RESPONSE.size, count)
        except ValueError:
            _fail(
                FailureClass.REQUEST_ERROR,
                IssueCode.INVALID_PAYLOAD_LENGTH,
                "mask word count does not match the binary payload",
                request_id,
            )
        message = MaskResponseFrame(request_id, mask_request_id, words)
        _raise_issue(_mask_response_issue(message, limits))
        return message
    if frame_type is FrameType.STATUS_REQUEST:
        return StatusRequestFrame(_STATUS_REQUEST.unpack(payload)[0])
    if frame_type is FrameType.READY:
        message = ReadyEvent(*_READY.unpack(payload))
        _raise_issue(_ready_issue(message, FailureClass.PROTOCOL_FATAL))
        return message
    if frame_type is FrameType.START:
        request_id, disposition, slot, matched, capacity = _START.unpack(payload)
        message = StartEvent(
            request_id,
            _decode_enum(
                disposition,
                CacheDisposition,
                "cache disposition",
                FailureClass.PROTOCOL_FATAL,
                request_id,
            ),
            slot,
            matched,
            capacity,
        )
        _raise_issue(_start_issue(message, FailureClass.PROTOCOL_FATAL))
        return message
    if frame_type is FrameType.PROMPT_PROGRESS:
        message = PromptProgressEvent(*_PROMPT_PROGRESS.unpack(payload))
        _raise_issue(_progress_issue(message, limits, FailureClass.PROTOCOL_FATAL))
        return message
    if frame_type is FrameType.TOKENS:
        request_id, offset, count = _TOKENS.unpack_from(payload)
        if count > limits.max_token_batch:
            _fail(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.LIMIT_EXCEEDED,
                "tokens event batch size exceeds its limit",
                request_id,
            )
        try:
            tokens = _unpack_words(payload, _TOKENS.size, count)
        except ValueError:
            _fail(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.INVALID_PAYLOAD_LENGTH,
                "token count does not match the binary event payload",
                request_id,
            )
        message = TokensEvent(request_id, offset, tokens)
        _raise_issue(_tokens_issue(message, limits, FailureClass.PROTOCOL_FATAL))
        return message
    if frame_type is FrameType.MASK_REQUEST:
        request_id, mask_request_id, words_per_mask, count = _MASK_REQUEST.unpack_from(
            payload
        )
        if count > limits.max_simulation_tokens:
            _fail(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.LIMIT_EXCEEDED,
                "mask request simulation token count exceeds its limit",
                request_id,
            )
        try:
            tokens = _unpack_words(payload, _MASK_REQUEST.size, count)
        except ValueError:
            _fail(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.INVALID_PAYLOAD_LENGTH,
                "simulation token count does not match the binary event payload",
                request_id,
            )
        message = MaskRequestEvent(request_id, mask_request_id, words_per_mask, tokens)
        _raise_issue(_mask_request_issue(message, limits, FailureClass.PROTOCOL_FATAL))
        return message
    if frame_type is FrameType.DONE:
        values = _DONE.unpack_from(payload)
        request_id = values[0]
        score_count = struct.unpack_from("<I", payload, _DONE.size)[0]
        try:
            logits = _unpack_floats(payload, _DONE.size + 4, score_count)
        except ValueError:
            _fail(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.INVALID_PAYLOAD_LENGTH,
                "option logit count does not match the binary event payload",
                request_id,
            )
        message = DoneEvent(
            request_id,
            _decode_enum(
                values[1],
                FinishReason,
                "finish reason",
                FailureClass.PROTOCOL_FATAL,
                request_id,
            ),
            *values[2:],
            logits,
        )
        _raise_issue(_done_issue(message, FailureClass.PROTOCOL_FATAL))
        return message
    if frame_type is FrameType.ERROR:
        failure, retryable, request_id, code_bytes, message_bytes = _ERROR.unpack_from(
            payload
        )
        if retryable > 1 or code_bytes + message_bytes != len(payload) - _ERROR.size:
            _fail(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.INVALID_PAYLOAD_LENGTH,
                "error string lengths do not match the frame payload",
            )
        code_start = _ERROR.size
        message_start = code_start + code_bytes
        try:
            classification = FailureClass(failure)
        except ValueError as error:
            raise ProtocolError(
                _issue(
                    FailureClass.PROTOCOL_FATAL,
                    IssueCode.INVALID_ERROR_CLASSIFICATION,
                    "error event classification is invalid",
                    request_id,
                )
            ) from error
        message = ErrorEvent(
            classification,
            request_id,
            bool(retryable),
            payload[code_start:message_start],
            payload[message_start:],
        )
        _raise_issue(_error_issue(message, limits, FailureClass.PROTOCOL_FATAL))
        return message
    if frame_type is FrameType.CAPACITY_EXHAUSTED:
        message = CapacityExhaustedEvent(*_CAPACITY_EXHAUSTED.unpack(payload))
        _raise_issue(_capacity_issue(message, FailureClass.PROTOCOL_FATAL))
        return message
    if frame_type is FrameType.STATUS_JSON:
        correlation_id, schema = _STATUS_JSON.unpack_from(payload)
        message = StatusJsonEvent(correlation_id, schema, payload[_STATUS_JSON.size :])
        _raise_issue(_status_issue(message, limits, FailureClass.PROTOCOL_FATAL))
        return message
    _fail(
        FailureClass.PROTOCOL_FATAL,
        IssueCode.UNKNOWN_FRAME_TYPE,
        "frame type is not defined by native protocol",
    )


def decode_frame(frame: Frame, limits: ProtocolLimits = ProtocolLimits()) -> Message:
    """Decode and strictly validate one typed payload frame."""

    try:
        return _decode_frame(frame, limits)
    except MemoryError as error:
        raise ProtocolError(
            _allocation_issue("allocation failed while decoding protocol message")
        ) from error


class FrameParser:
    """Incremental, one-frame-at-a-time runtime stream parser."""

    def __init__(self, limits: ProtocolLimits = ProtocolLimits()):
        self._limits = limits
        self._header = bytearray()
        self._reading_payload = False
        self._current_type = FrameType.REQUEST
        self._expected_payload_bytes = 0
        self._payload = bytearray()
        self._terminal_issue = _limits_issue(limits)

    @property
    def failed(self) -> bool:
        return self._terminal_issue is not None

    def _fail(self, consumed: int, issue: ProtocolIssue) -> ParseStep:
        self._terminal_issue = issue
        return ParseStep(consumed, issue=issue)

    def _reset_current_frame(self) -> None:
        self._header.clear()
        self._reading_payload = False
        self._expected_payload_bytes = 0
        self._payload.clear()

    def _parse_header(self) -> ProtocolIssue | None:
        magic, version, header_bytes, raw_type, flags, payload_bytes, reserved = (
            _HEADER.unpack(self._header)
        )
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
        if header_bytes != FRAME_HEADER_BYTES:
            return _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.INVALID_HEADER_SIZE,
                "native protocol frame header must be exactly 24 bytes",
            )
        try:
            self._current_type = FrameType(raw_type)
        except ValueError:
            return _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.UNKNOWN_FRAME_TYPE,
                "frame type is not defined by native protocol",
            )
        if flags:
            return _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.NON_ZERO_HEADER_FLAGS,
                "native protocol frame flags must be zero",
            )
        if reserved:
            return _issue(
                FailureClass.PROTOCOL_FATAL,
                IssueCode.NON_ZERO_RESERVED_FIELD,
                "native protocol reserved header field must be zero",
            )
        if issue := _payload_length_issue(
            self._current_type, payload_bytes, self._limits
        ):
            return issue
        self._expected_payload_bytes = payload_bytes
        self._payload.clear()
        self._reading_payload = True
        return None

    def consume(self, data: bytes | bytearray | memoryview) -> ParseStep:
        """Consume up to one frame and report exactly how many bytes were used."""

        if self._terminal_issue:
            return ParseStep(
                issue=_issue(
                    FailureClass.PROTOCOL_FATAL,
                    IssueCode.PARSER_ALREADY_FAILED,
                    self._terminal_issue.describe(),
                )
            )
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
            return self._terminal_issue
        if not self._header and not self._reading_payload:
            return None
        if not self._reading_payload:
            message = f"stream ended after {len(self._header)} of 24 frame-header bytes"
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
