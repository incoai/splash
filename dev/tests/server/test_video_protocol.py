"""The video engine's stream as the server writes and reads it
(server/video_protocol.py): the golden bytes the engine's codec agrees on
(dev/tests/engine/video_protocol_test.cpp), each frame's rules, the parser's
failures, and the engine protocol's payloads it carries."""

import struct
import unittest
from dataclasses import replace
from pathlib import Path

from dev.tests import native_peer, video_peer
from server import protocol
from server import video_protocol as v

ROOT = Path(__file__).parents[3]
GOLDEN = ROOT / "dev/tests/engine/video_protocol_golden.txt"

VIDEO_REQUEST = v.VideoRequestFrame(
    92, "turbo", v.VideoOutputFlag.PREVIEW | v.VideoOutputFlag.AUDIO,
    73, 1344, 768, 0, 42, (5, 6, 7),
)  # fmt: skip
# The default DiT's clip without audio or preview.
VIDEO_DEFAULT_REQUEST = replace(
    VIDEO_REQUEST, dit="default", outputs=v.VideoOutputFlag(0),
    steps=50, prompt_tokens=(0xFFFFFFFF,),
)  # fmt: skip
# A clip of a 4 x 2 canvas from its first and last frames.
VIDEO_KEYFRAME_REQUEST = v.VideoRequestFrame(
    92, "turbo", v.VideoOutputFlag.AUDIO, 73, 4, 2, 0, 42, (5,),
    bytes(range(24)), bytes(255 - n for n in range(24)),
)  # fmt: skip
# A Ref2VA clip of a 4 x 2 canvas from a 2 x 2 video of two frames with a
# soundtrack of three samples, a 2 x 1 image and an audio reference of one
# sample.
VIDEO_REFERENCE_REQUEST = v.VideoRequestFrame(
    92, "pika-ref2va-6", v.VideoOutputFlag.AUDIO, 73, 4, 2, 0, 7, (5, 6),
    references=(
        v.VideoReference(v.VideoReferenceKind.VIDEO, 2, 2, 2, bytes(3 * n for n in range(24)),
                         struct.pack("<6f", 0.5, -0.25, 1.0, 0.0, -1.0, 0.125)),
        v.VideoReference(v.VideoReferenceKind.IMAGE, 1, 2, 1, bytes((9, 8, 7, 6, 5, 4))),
        v.VideoReference(v.VideoReferenceKind.AUDIO, 0, 0, 0, b"", struct.pack("<2f", 0.75, -0.5)),
    ),
)  # fmt: skip


def client_messages():
    return {
        "video_request": VIDEO_REQUEST,
        "video_request_default": VIDEO_DEFAULT_REQUEST,
        "video_request_keyframes": VIDEO_KEYFRAME_REQUEST,
        "video_request_references": VIDEO_REFERENCE_REQUEST,
        "cancel": protocol.CancelFrame(92),
    }


def engine_events():
    return {
        "video_ready": v.VideoReadyEvent(1, 2048),
        "video_progress": v.VideoProgressEvent(
            92, v.VideoPhase.DENOISE, 3, 8, 312_000_000, 15_200_000_000
        ),
        "video_output": v.VideoOutputEvent(
            92, v.VideoOutputKind.VIDEO, 73, 345_000_000, b"\x00\x00\x00\x18ftyp"
        ),
        "error_request": protocol.ErrorEvent(
            protocol.FailureClass.REQUEST_ERROR,
            92,
            False,
            b"cancelled",
            b"the video request was cancelled",
        ),  # fmt: skip
        "error_engine": protocol.ErrorEvent(
            protocol.FailureClass.ENGINE_UNHEALTHY,
            0,
            False,
            b"video_engine_failed",
            b"the video VAE failed",
        ),  # fmt: skip
        "start_failed": protocol.StartFailedEvent(
            protocol.StartFailureKind.DEVICE,
            protocol.ShortfallBound.DEVICE_MEMORY,
            0,
            0,
            0,
            b"the video engine runs on an Apple10 GPU (M5) or later",
        ),  # fmt: skip
    }


def read_golden():
    golden = {}
    for line in GOLDEN.read_text().splitlines():
        if line and not line.startswith("#"):
            name, data = line.split()
            golden[name] = bytes.fromhex(data)
    return golden


def parse_all(data, parser=None):
    parser = parser or v.FrameParser()
    frames = []
    offset = 0
    while offset < len(data):
        step = parser.consume(memoryview(data)[offset:])
        if step.issue:
            raise protocol.ProtocolError(step.issue)
        if not step.consumed_bytes:
            raise AssertionError("parser made no progress")
        offset += step.consumed_bytes
        if step.frame:
            frames.append(step.frame)
    if issue := parser.finish():
        raise protocol.ProtocolError(issue)
    return frames


def parser_issue(data, parser=None):
    parser = parser or v.FrameParser()
    offset = 0
    while offset < len(data):
        step = parser.consume(memoryview(data)[offset:])
        offset += step.consumed_bytes
        if step.issue:
            return step.issue
        if not step.consumed_bytes:
            break
    issue = parser.finish()
    if not issue:
        raise AssertionError("expected parser failure")
    return issue


def decode_event(data):
    """The server's view of one event the video engine wrote."""
    (frame,) = parse_all(data)
    return v.decode_frame(frame)


def mutate(data, offset, code, value):
    result = bytearray(data)
    struct.pack_into(code, result, offset, value)
    return bytes(result)


class VideoProtocolTests(unittest.TestCase):
    def assert_protocol_error(self, failure_class, code, callback):
        with self.assertRaises(protocol.ProtocolError) as caught:
            callback()
        self.assertEqual(caught.exception.issue.failure_class, failure_class)
        self.assertEqual(caught.exception.issue.code, code)
        return caught.exception.issue

    def assert_fatal_event(self, code, event):
        """An event that breaks the protocol fails the stream when read."""
        return self.assert_protocol_error(
            protocol.FailureClass.PROTOCOL_FATAL,
            code,
            lambda: decode_event(video_peer.serialize_event(event)),
        )

    def test_client_frames_match_golden(self):
        golden = read_golden()
        for name, message in client_messages().items():
            with self.subTest(name=name):
                self.assertEqual(v.serialize_message(message).hex(), golden[name].hex())
                ((read, _),) = video_peer.ClientFrameReader().feed(golden[name])
                self.assertEqual(read, message)

    def test_events_decode_from_golden(self):
        golden = read_golden()
        self.assertEqual(set(golden), {*client_messages(), *engine_events()})
        for name, event in engine_events().items():
            with self.subTest(name=name):
                self.assertEqual(decode_event(golden[name]), event)
                self.assertEqual(video_peer.serialize_event(event), golden[name])

    def test_cancel_error_and_start_failure_are_the_engine_protocols(self):
        """Their frames are the engine protocol's but for the magic and the
        version, as protocol_golden.txt holds that protocol's."""
        golden = read_golden()
        cancel = protocol.serialize_message(protocol.CancelFrame(92))
        frames = {"cancel": cancel}
        for name in ("error_request", "error_engine", "start_failed"):
            frames[name] = native_peer.serialize_event(engine_events()[name])
        for name, frame in frames.items():
            with self.subTest(name=name):
                self.assertEqual(golden[name][:6], b"SPLV\x01\x00")
                self.assertEqual(golden[name][6:], bytes(frame[6:]))

    def test_video_frames_are_validated(self):
        for code, change in (
            (protocol.IssueCode.LIMIT_EXCEEDED, {"prompt_tokens": ()}),
            (protocol.IssueCode.INVALID_COUNT, {"frames": 0}),
            (protocol.IssueCode.INVALID_ENUM_VALUE, {"outputs": v.VideoOutputFlag(4)}),
            (protocol.IssueCode.INVALID_ENUM_VALUE, {"dit": ""}),
            (protocol.IssueCode.INVALID_ENUM_VALUE, {"dit": ".."}),
            (protocol.IssueCode.INVALID_ENUM_VALUE, {"dit": "Turbo"}),
            (protocol.IssueCode.INVALID_COUNT, {"first_frame": bytes(23)}),
        ):
            with self.subTest(change=change):
                invalid = replace(VIDEO_REQUEST, **change)
                self.assert_protocol_error(protocol.FailureClass.REQUEST_ERROR, code,
                                           lambda: v.serialize_message(invalid))  # fmt: skip
        video, image, audio = VIDEO_REFERENCE_REQUEST.references
        for code, references in (
            (protocol.IssueCode.INVALID_COUNT, (replace(video, pixels=bytes(23)),)),
            (protocol.IssueCode.INVALID_COUNT, (replace(image, frames=2, pixels=bytes(12)),)),
            (protocol.IssueCode.INVALID_COUNT, (replace(image, soundtrack=bytes(8)),)),
            (protocol.IssueCode.INVALID_COUNT, (replace(video, soundtrack=bytes(12)),)),
            (protocol.IssueCode.INVALID_ENUM_VALUE, (replace(image, kind=3),)),
            (protocol.IssueCode.INVALID_COUNT, (replace(audio, soundtrack=b""),)),
            (protocol.IssueCode.INVALID_COUNT, (replace(audio, frames=1),)),
            (protocol.IssueCode.LIMIT_EXCEEDED, (image,) * (v.MAX_VIDEO_REFERENCES + 1)),
        ):  # fmt: skip
            with self.subTest(references=len(references), code=code):
                invalid = replace(VIDEO_REFERENCE_REQUEST, references=references)
                self.assert_protocol_error(protocol.FailureClass.REQUEST_ERROR, code,
                                           lambda: v.serialize_message(invalid))  # fmt: skip
        # Without its id, a request's failure fails the stream.
        self.assert_protocol_error(
            protocol.FailureClass.PROTOCOL_FATAL,
            protocol.IssueCode.INVALID_REQUEST_ID,
            lambda: v.serialize_message(replace(VIDEO_REQUEST, request_id=0)),
        )
        self.assert_protocol_error(
            protocol.FailureClass.PROTOCOL_FATAL,
            protocol.IssueCode.UNKNOWN_FRAME_TYPE,
            lambda: v.serialize_message(protocol.StopFrame(92)),
        )

    def test_video_events_are_validated(self):
        for code, event in (
            (protocol.IssueCode.INVALID_COUNT, v.VideoReadyEvent(0, 2048)),
            (protocol.IssueCode.INVALID_COUNT, v.VideoReadyEvent(1, 0)),
            (protocol.IssueCode.INVALID_COUNT,
             v.VideoProgressEvent(92, v.VideoPhase.DENOISE, 9, 8, 0, 0)),
            (protocol.IssueCode.INVALID_ENUM_VALUE,
             v.VideoProgressEvent(92, 8, 1, 8, 0, 0)),
            (protocol.IssueCode.LIMIT_EXCEEDED,
             v.VideoOutputEvent(92, v.VideoOutputKind.VIDEO, 73, 0, b"")),
            (protocol.IssueCode.INVALID_ENUM_VALUE,
             v.VideoOutputEvent(92, 2, 73, 0, b"mp4")),
            # The engine protocol's rules hold for its Error in the video
            # stream too.
            (protocol.IssueCode.INVALID_ERROR_CLASSIFICATION,
             protocol.ErrorEvent(protocol.FailureClass.REQUEST_ERROR, 0, False, b"bad", b"")),
        ):  # fmt: skip
            with self.subTest(event=event):
                self.assert_fatal_event(code, event)

    def test_header_failures(self):
        valid = read_golden()["video_progress"]
        for code, data in (
            (protocol.IssueCode.BAD_MAGIC, b"X" + valid[1:]),
            (protocol.IssueCode.UNSUPPORTED_VERSION, mutate(valid, 4, "<H", 2)),
            (protocol.IssueCode.UNKNOWN_FRAME_TYPE, mutate(valid, 6, "<H", 0x7777)),
            # The engine never sends what the server sends, nor the engine
            # protocol's other events.
            (protocol.IssueCode.UNKNOWN_FRAME_TYPE, mutate(valid, 6, "<H", 0x0007)),
            (protocol.IssueCode.UNKNOWN_FRAME_TYPE, mutate(valid, 6, "<H", 0x0100)),
            (protocol.IssueCode.FRAME_TOO_LARGE, mutate(valid, 8, "<Q", 34)),
            (protocol.IssueCode.INVALID_PAYLOAD_LENGTH, mutate(valid, 8, "<Q", 32)),
        ):
            with self.subTest(code=code, data=data[:16].hex()):
                issue = parser_issue(data)
                self.assertEqual(
                    issue.failure_class, protocol.FailureClass.PROTOCOL_FATAL
                )
                self.assertEqual(issue.code, code)

    def test_a_frame_sent_to_the_other_engine_fails_at_its_magic(self):
        """Either way: an engine protocol event in the video stream, and a
        video event in the engine protocol's."""
        error = engine_events()["error_request"]
        for parser, data in (
            (v.FrameParser(), native_peer.serialize_event(error)),
            (protocol.FrameParser(), read_golden()["error_request"]),
        ):
            with self.subTest(parser=type(parser).__module__):
                issue = parser_issue(data, parser)
                self.assertEqual(issue.code, protocol.IssueCode.BAD_MAGIC)
        self.assertTrue(v.starts_engine_protocol(native_peer.serialize_event(error)))
        self.assertFalse(v.starts_engine_protocol(read_golden()["error_request"]))

    def test_truncation_at_every_boundary(self):
        data = read_golden()["video_output"]
        for cut in range(1, len(data)):
            with self.subTest(cut=cut):
                issue = parser_issue(data[:cut])
                self.assertEqual(issue.code, protocol.IssueCode.TRUNCATED_FRAME)
        self.assertIsNone(v.FrameParser().finish())

    def test_events_arrive_one_after_another(self):
        golden = read_golden()
        stream = b"".join(golden[name] for name in engine_events())
        self.assertEqual([v.decode_frame(frame) for frame in parse_all(stream)],
                         list(engine_events().values()))  # fmt: skip
        frames = []
        parser = v.FrameParser()
        for byte in stream:
            step = parser.consume(bytes((byte,)))
            self.assertEqual((step.consumed_bytes, step.issue), (1, None))
            if step.frame:
                frames.append(step.frame)
        self.assertEqual(len(frames), len(engine_events()))

    def test_dit_names(self):
        for name, valid in (("turbo", True), ("pika-ref2va-6", True), ("a" * 64, True),
                            ("", False), ("a" * 65, False), ("..", False), ("a..b", False),
                            ("Turbo", False), ("a/b", False), (None, False)):  # fmt: skip
            with self.subTest(name=name):
                self.assertIs(v.valid_dit_name(name), valid)


if __name__ == "__main__":
    unittest.main()
