"""The engine's side of the native wire, for fake engines and golden tests;
production is server/protocol.py."""

import json
import struct
from dataclasses import replace

from server import protocol as wire

_HEADER = struct.Struct("<4sHHQ")
_REQUEST = struct.Struct("<QBBQQIIIffIffffQIII")
_IMAGE_SPAN = struct.Struct("<IIIIQQ")
_MASK_RESPONSE = struct.Struct("<QQI")
_ID = struct.Struct("<Q")


def request_frame(**overrides) -> wire.RequestFrame:
    """A valid text request with neutral values, each override replacing one
    field."""
    request = wire.RequestFrame(
        request_id=1,
        priority=wire.RequestPriority.NORMAL,
        absolute_deadline_unix_micros=1,
        remaining_deadline_micros=1,
        logical_max_output_tokens=1,
        prompt_tokens=(1,),
        sampling=wire.SamplingParameters(),
        seed=0,
        constraint=wire.ConstraintMode.NONE,
        image_spans=(),
        image_pixels=b"",
        score_tokens=(),
        generation_prompt_tokens=0,
        flags=wire.RequestFlag(0),
    )
    return replace(request, **overrides)


def status_event(correlation_id: int = 1, **fields) -> wire.StatusJsonEvent:
    """A schema-current status document of a ready, healthy engine; each field
    replaces a top-level key, and None removes it."""
    document = {
        "schema_version": wire.STATUS_SCHEMA_VERSION,
        "ready": True,
        "memory_pressure": "normal",
        "thermal_state": "nominal",
        "metal": {"healthy": True},
    }
    for key, value in fields.items():
        if value is None:
            document.pop(key, None)
        else:
            document[key] = value
    return wire.StatusJsonEvent(
        correlation_id, json.dumps(document, separators=(",", ":")).encode()
    )


def _event_payload(event: wire.EngineEvent) -> tuple[wire.FrameType, bytes]:
    match event:
        case wire.ReadyEvent():
            return wire.FrameType.READY, struct.pack(
                "<IIB",
                event.max_concurrent_requests,
                event.max_context_tokens,
                event.vision,
            )
        case wire.StartEvent():
            return wire.FrameType.START, struct.pack(
                "<QII", event.request_id, event.lane, event.matched_prompt_tokens
            )
        case wire.PromptProgressEvent():
            return wire.FrameType.PROMPT_PROGRESS, struct.pack(
                "<QIQ", event.request_id, event.processed_tokens, event.elapsed_micros
            )
        case wire.TokensEvent():
            return wire.FrameType.TOKENS, struct.pack(
                f"<QII{len(event.tokens)}I",
                event.request_id,
                event.sequence_offset,
                len(event.tokens),
                *event.tokens,
            )
        case wire.MaskRequestEvent():
            return wire.FrameType.MASK_REQUEST, struct.pack(
                f"<QQII{len(event.simulation_tokens)}I",
                event.request_id,
                event.mask_request_id,
                event.words_per_mask,
                len(event.simulation_tokens),
                *event.simulation_tokens,
            )
        case wire.DoneEvent():
            payload = struct.pack(
                f"<QBIIQQQI{len(event.option_logits)}f",
                event.request_id,
                event.reason,
                event.prompt_tokens,
                event.completion_tokens,
                event.prefill_micros,
                event.decode_micros,
                event.wall_micros,
                len(event.option_logits),
                *event.option_logits,
            )
            if event.option_logits:
                # The score readout: normalizer, top count, ids, logits.
                count = len(event.top_token_ids)
                payload += struct.pack(
                    f"<dI{count}I{count}f",
                    event.log_normalizer,
                    count,
                    *event.top_token_ids,
                    *event.top_logits,
                )
            return wire.FrameType.DONE, payload
        case wire.ErrorEvent():
            return wire.FrameType.ERROR, struct.pack(
                f"<BBQII{len(event.code)}s{len(event.message)}s",
                event.failure_class,
                event.retryable,
                event.request_id,
                len(event.code),
                len(event.message),
                event.code,
                event.message,
            )
        case wire.StatusJsonEvent():
            return wire.FrameType.STATUS_JSON, _ID.pack(
                event.correlation_id
            ) + event.json
    raise TypeError(f"{type(event).__name__} is not an engine event")


def serialize_event(event: wire.EngineEvent) -> bytes:
    """Header and payload exactly as the engine writes them, without
    validation, so fakes can send the server invalid events."""
    frame_type, payload = _event_payload(event)
    return (
        _HEADER.pack(b"SPLH", wire.PROTOCOL_VERSION, frame_type, len(payload)) + payload
    )


def _words(payload: bytes, offset: int, count: int) -> tuple[int, ...]:
    return struct.unpack_from(f"<{count}I", payload, offset)


def _decode_request(payload: bytes) -> wire.RequestFrame:
    (
        request_id,
        priority,
        constraint,
        absolute_deadline,
        remaining_deadline,
        max_output,
        prompt_count,
        span_count,
        *sampling,
        seed,
        score_count,
        generation_prompt_tokens,
        flags,
    ) = _REQUEST.unpack_from(payload)
    offset = _REQUEST.size
    prompt = _words(payload, offset, prompt_count)
    offset += 4 * prompt_count
    spans = tuple(
        wire.ImageSpan(
            *_IMAGE_SPAN.unpack_from(payload, offset + index * _IMAGE_SPAN.size)
        )
        for index in range(span_count)
    )
    offset += _IMAGE_SPAN.size * span_count
    pixel_bytes = sum(span.pixel_bytes for span in spans)
    pixels = bytes(payload[offset : offset + pixel_bytes])
    scores = _words(payload, offset + pixel_bytes, score_count)
    return wire.RequestFrame(
        request_id,
        wire.RequestPriority(priority),
        absolute_deadline,
        remaining_deadline,
        max_output,
        prompt,
        wire.SamplingParameters(*sampling),
        seed,
        wire.ConstraintMode(constraint),
        spans,
        pixels,
        scores,
        generation_prompt_tokens,
        wire.RequestFlag(flags),
    )


def decode_client_frame(frame_type: int, payload: bytes) -> wire.ClientMessage:
    """One client frame's message, read as the engine reads it but without
    checking the rules the server's encoder already enforced."""
    match frame_type:
        case wire.FrameType.REQUEST:
            return _decode_request(payload)
        case wire.FrameType.CANCEL:
            return wire.CancelFrame(*_ID.unpack(payload))
        case wire.FrameType.MASK_RESPONSE:
            request_id, mask_request_id, count = _MASK_RESPONSE.unpack_from(payload)
            mask = payload[_MASK_RESPONSE.size : _MASK_RESPONSE.size + 4 * count]
            return wire.MaskResponseFrame(request_id, mask_request_id, bytes(mask))
        case wire.FrameType.STATUS_REQUEST:
            return wire.StatusRequestFrame(*_ID.unpack(payload))
    raise ValueError(f"frame type {frame_type:#06x} is not a client frame")


class ClientFrameReader:
    """Splits the server's byte stream into client frames."""

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data) -> list[tuple[wire.ClientMessage, bytes]]:
        """Each complete frame in what has arrived so far, as its message and
        its raw bytes."""
        self._buffer.extend(data)
        frames = []
        while len(self._buffer) >= wire.FRAME_HEADER_BYTES:
            magic, version, frame_type, payload_bytes = _HEADER.unpack_from(
                self._buffer
            )
            if (magic, version) != (b"SPLH", wire.PROTOCOL_VERSION):
                raise ValueError("client frame header is invalid")
            end = wire.FRAME_HEADER_BYTES + payload_bytes
            if len(self._buffer) < end:
                break
            raw = bytes(self._buffer[:end])
            del self._buffer[:end]
            frames.append(
                (decode_client_frame(frame_type, raw[wire.FRAME_HEADER_BYTES :]), raw)
            )
        return frames
