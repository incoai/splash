"""The engine's side of the native wire, for fake engines and protocol tests;
production is server/protocol.py (the base both versions share, framed as
version 8) and server/audio_protocol.py (version 11, the audio engine's)."""

import json
import struct

from server import audio_protocol as audio_wire
from server import protocol as wire

_HEADER = struct.Struct("<4sHHQ")
_ID = struct.Struct("<Q")
_SPEECH = struct.Struct("<QIIfIfffIfQIIIIIIII")
_TRANSCRIBE = struct.Struct("<QIIIII")
_AUDIO = struct.Struct("<QII")
_CODES = struct.Struct("<QI")


def status_event(correlation_id: int = 1, **fields) -> wire.StatusJsonEvent:
    """A schema-current status document of a ready, healthy engine; each field
    replaces a top-level key, and None removes it."""
    document = {
        "schema_version": wire.STATUS_SCHEMA_VERSION,
        "ready": True,
        "memory_pressure": "normal",
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


def _event_payload(event) -> tuple[int, bytes]:
    match event:
        case wire.ReadyEvent():
            return wire.FrameType.READY, struct.pack(
                "<IIB",
                event.max_concurrent_requests,
                event.max_context_tokens,
                event.vision,
            )
        case audio_wire.ReadyEvent():
            return audio_wire.FrameType.READY, struct.pack(
                "<IIB",
                event.max_concurrent_requests,
                event.max_context_tokens,
                event.kind,
            )
        case audio_wire.AudioEvent():
            return (
                audio_wire.FrameType.AUDIO,
                _AUDIO.pack(event.request_id, event.sample_offset, event.sample_count)
                + event.pcm,
            )
        case audio_wire.CodesEvent():
            return (
                audio_wire.FrameType.CODES,
                _CODES.pack(event.request_id, event.frame_count)
                + struct.pack(f"<{len(event.codes)}I", *event.codes),
            )
        case wire.StartEvent():
            return wire.FrameType.START, struct.pack(
                "<QII", event.request_id, event.lane, event.matched_prompt_tokens
            )
        case wire.TokensEvent():
            return wire.FrameType.TOKENS, struct.pack(
                f"<QII{len(event.tokens)}I",
                event.request_id,
                event.sequence_offset,
                len(event.tokens),
                *event.tokens,
            )
        case wire.DoneEvent():
            return wire.FrameType.DONE, struct.pack(
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


def serialize_event(event, version: int = wire.PROTOCOL_VERSION) -> bytes:
    """Header and payload exactly as the engine writes them, without
    validation, so fakes can send the server invalid events."""
    frame_type, payload = _event_payload(event)
    return _HEADER.pack(b"SPLH", version, frame_type, len(payload)) + payload


def serialize_audio_event(event) -> bytes:
    """serialize_event as the audio engine, version 11, writes it."""
    return serialize_event(event, audio_wire.PROTOCOL_VERSION)


def _decode_speech(payload: bytes) -> audio_wire.SpeechRequest:
    (
        request_id,
        flags,
        language_id,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        predictor_temperature,
        predictor_top_k,
        predictor_top_p,
        seed,
        max_frames,
        chunk_frames,
        text_count,
        reference_text_count,
        reference_frame_count,
        speaker_present,
        instruction_count,
        preset_speaker,
    ) = _SPEECH.unpack_from(payload)
    offset = _SPEECH.size
    text = _words(payload, offset, text_count)
    offset += 4 * text_count
    instruction = _words(payload, offset, instruction_count)
    offset += 4 * instruction_count
    reference_text = _words(payload, offset, reference_text_count)
    offset += 4 * reference_text_count
    code_count = reference_frame_count * audio_wire.CODES_PER_FRAME
    codes = _words(payload, offset, code_count)
    offset += 4 * code_count
    speaker = ()
    if speaker_present:
        speaker = struct.unpack_from(
            f"<{audio_wire.SPEAKER_EMBEDDING_DIMS}f", payload, offset
        )
        offset += 4 * audio_wire.SPEAKER_EMBEDDING_DIMS
    assert offset == len(payload), "speech request payload has trailing bytes"
    return audio_wire.SpeechRequest(
        request_id,
        text,
        language_id,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        predictor_temperature,
        predictor_top_k,
        predictor_top_p,
        seed,
        max_frames,
        chunk_frames,
        reference_text,
        codes,
        speaker,
        flags,
        instruction,
        preset_speaker,
    )


def _decode_transcribe(payload: bytes) -> audio_wire.TranscribeRequest:
    request_id, flags, max_tokens, prefix_count, suffix_count, sample_count = (
        _TRANSCRIBE.unpack_from(payload)
    )
    offset = _TRANSCRIBE.size
    prefix = _words(payload, offset, prefix_count)
    offset += 4 * prefix_count
    suffix = _words(payload, offset, suffix_count)
    offset += 4 * suffix_count
    samples = bytes(payload[offset : offset + 4 * sample_count])
    assert offset + 4 * sample_count == len(payload), "transcribe payload length"
    return audio_wire.TranscribeRequest(
        request_id, prefix, suffix, samples, max_tokens, flags
    )


def decode_audio_client_frame(frame_type: int, payload: bytes):
    """One client frame's message, read as the audio engine reads it."""
    match frame_type:
        case audio_wire.FrameType.SPEECH_REQUEST:
            return _decode_speech(payload)
        case audio_wire.FrameType.TRANSCRIBE_REQUEST:
            return _decode_transcribe(payload)
        case audio_wire.FrameType.CANCEL:
            return wire.CancelFrame(*_ID.unpack(payload))
        case audio_wire.FrameType.STATUS_REQUEST:
            return wire.StatusRequestFrame(*_ID.unpack(payload))
    raise ValueError(f"frame type {frame_type:#06x} is not an audio client frame")


def _words(payload: bytes, offset: int, count: int) -> tuple[int, ...]:
    return struct.unpack_from(f"<{count}I", payload, offset)


def decode_client_frame(frame_type: int, payload: bytes) -> wire.ClientMessage:
    """One client frame's message of the base protocol, read as the engine
    reads it but without checking the rules the server's encoder already
    enforced."""
    match frame_type:
        case wire.FrameType.CANCEL:
            return wire.CancelFrame(*_ID.unpack(payload))
        case wire.FrameType.STATUS_REQUEST:
            return wire.StatusRequestFrame(*_ID.unpack(payload))
    raise ValueError(f"frame type {frame_type:#06x} is not a client frame")


class ClientFrameReader:
    """Splits the server's byte stream into client frames: the base
    protocol's, version 8, unless built for the audio engine's."""

    def __init__(self, *, audio: bool = False):
        self._buffer = bytearray()
        self._version = audio_wire.PROTOCOL_VERSION if audio else wire.PROTOCOL_VERSION
        self._decode = decode_audio_client_frame if audio else decode_client_frame

    def feed(self, data) -> list[tuple[object, bytes]]:
        """Each complete frame in what has arrived so far, as its message and
        its raw bytes."""
        self._buffer.extend(data)
        frames = []
        while len(self._buffer) >= wire.FRAME_HEADER_BYTES:
            magic, version, frame_type, payload_bytes = _HEADER.unpack_from(
                self._buffer
            )
            if (magic, version) != (b"SPLH", self._version):
                raise ValueError("client frame header is invalid")
            end = wire.FRAME_HEADER_BYTES + payload_bytes
            if len(self._buffer) < end:
                break
            raw = bytes(self._buffer[:end])
            del self._buffer[:end]
            frames.append(
                (self._decode(frame_type, raw[wire.FRAME_HEADER_BYTES :]), raw)
            )
        return frames
