"""The video engine's side of its stream, for fake video engines and golden
tests; production is server/video_protocol.py. Error and StartFailed are the
engine protocol's frames (native_peer) under the video stream's header."""

import struct
from dataclasses import astuple

from dev.tests import native_peer
from server import protocol
from server import video_protocol as wire

_HEADER = struct.Struct("<4sHHQ")
_START = struct.Struct("<4sH")
_VIDEO_REQUEST = struct.Struct("<QBBIIIIQBII")
_VIDEO_REFERENCE = struct.Struct("<BIIIQ")
_ID = struct.Struct("<Q")


def _frame(frame_type: wire.VideoFrameType, payload: bytes) -> bytes:
    return (
        _HEADER.pack(b"SPLV", wire.VIDEO_PROTOCOL_VERSION, frame_type, len(payload))
        + payload
    )


def serialize_event(event: wire.VideoEvent) -> bytes:
    """Header and payload exactly as the video engine writes them, without
    validation, so fakes can send the server invalid events."""
    match event:
        case wire.VideoReadyEvent():
            return _frame(
                wire.VideoFrameType.VIDEO_READY, struct.pack("<II", *astuple(event))
            )
        case wire.VideoProgressEvent():
            return _frame(
                wire.VideoFrameType.VIDEO_PROGRESS,
                struct.pack("<QBIIQQ", *astuple(event)),
            )
        case wire.VideoOutputEvent():
            *header, mp4 = astuple(event)
            return _frame(
                wire.VideoFrameType.VIDEO_OUTPUT, struct.pack("<QBIQ", *header) + mp4
            )
        case protocol.ErrorEvent() | protocol.StartFailedEvent():
            frame = native_peer.serialize_event(event)
            return (
                _START.pack(b"SPLV", wire.VIDEO_PROTOCOL_VERSION) + frame[_START.size :]
            )
    raise TypeError(f"{type(event).__name__} is not a video engine event")


def _decode_video_request(payload: bytes) -> wire.VideoRequestFrame:
    fields = _VIDEO_REQUEST.unpack_from(payload)
    number, name, outputs, *clip, keyframes, count, refs = fields
    at = _VIDEO_REQUEST.size + name
    dit = bytes(payload[_VIDEO_REQUEST.size : at]).decode()
    tokens = struct.unpack_from(f"<{count}I", payload, at)
    at += 4 * count
    headers = []
    for _ in range(refs):
        headers.append(_VIDEO_REFERENCE.unpack_from(payload, at))
        at += _VIDEO_REFERENCE.size
    # The keyframes after the reference headers, each height x width x 3
    # bytes, then each reference's pixels and soundtrack.
    canvas = clip[1] * clip[2] * 3
    frames = []
    for flag in wire.VideoKeyframeFlag:
        size = canvas if keyframes & flag else 0
        frames.append(bytes(payload[at : at + size]))
        at += size
    references = []
    for kind, n, width, height, samples in headers:
        pixels = bytes(payload[at : at + n * height * width * 3])
        at += len(pixels)
        sound = bytes(payload[at : at + 8 * samples])
        at += len(sound)
        references.append(wire.VideoReference(wire.VideoReferenceKind(kind), n, width,
                                              height, pixels, sound))  # fmt: skip
    return wire.VideoRequestFrame(number, dit,
                                  wire.VideoOutputFlag(outputs), *clip, tokens, *frames,
                                  tuple(references))  # fmt: skip


def decode_client_frame(frame_type: int, payload: bytes) -> wire.ClientMessage:
    """One client frame's message, read as the video engine reads it but
    without checking the rules the server's encoder already enforced."""
    match frame_type:
        case wire.VideoFrameType.VIDEO_REQUEST:
            return _decode_video_request(payload)
        case wire.VideoFrameType.CANCEL:
            return protocol.CancelFrame(*_ID.unpack(payload))
    raise ValueError(f"frame type {frame_type:#06x} is not a video client frame")


class ClientFrameReader:
    """Splits the server's byte stream into video client frames."""

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data) -> list[tuple[wire.ClientMessage, bytes]]:
        """Each complete frame in what has arrived so far, as its message and
        its raw bytes."""
        self._buffer.extend(data)
        frames = []
        while len(self._buffer) >= _HEADER.size:
            magic, version, frame_type, payload_bytes = _HEADER.unpack_from(
                self._buffer
            )
            if (magic, version) != (b"SPLV", wire.VIDEO_PROTOCOL_VERSION):
                raise ValueError("video client frame header is invalid")
            end = _HEADER.size + payload_bytes
            if len(self._buffer) < end:
                break
            raw = bytes(self._buffer[:end])
            del self._buffer[:end]
            frames.append((decode_client_frame(frame_type, raw[_HEADER.size :]), raw))
        return frames
