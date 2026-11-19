"""The base protocol's codec: the framing, the client frames and engine
events both versions share, and the parser the audio protocol builds on."""

import array
import random
import struct
import unittest

from dev.tests import native_peer
from server import protocol as p


def client_messages():
    return {
        "cancel": p.CancelFrame(91),
        "status_request": p.StatusRequestFrame(808),
    }


def engine_events():
    return {
        "ready": p.ReadyEvent(4, 524_288, False),
        "ready_vision": p.ReadyEvent(4, 524_288, True),
        "start": p.StartEvent(91, 2, 4096),
        "tokens": p.TokensEvent(91, 17, (10, 11, 12)),
        "done": p.DoneEvent(91, p.FinishReason.STOP, 4096, 512, 1000, 2000, 3500, ()),
        "done_length": p.DoneEvent(
            91, p.FinishReason.LENGTH, 4096, 512, 1000, 2000, 3500, ()
        ),
        "done_cancelled": p.DoneEvent(
            91, p.FinishReason.CANCELLED, 4096, 128, 1000, 500, 1500, ()
        ),
        "error_request": p.ErrorEvent(
            p.FailureClass.REQUEST_ERROR,
            91,
            True,
            b"deadline_exceeded",
            b"request deadline expired",
        ),
        "error_engine": p.ErrorEvent(
            p.FailureClass.ENGINE_UNHEALTHY,
            0,
            False,
            b"gpu_fault",
            b"Metal command buffer failed",
        ),
        "error_protocol": p.ErrorEvent(
            p.FailureClass.PROTOCOL_FATAL,
            0,
            False,
            b"bad_frame",
            b"stream framing cannot be trusted",
        ),
        "status_json": p.StatusJsonEvent(
            808, b'{\n  "schema_version": 6, "ready": true\n}'
        ),
    }


def parse_all(data):
    parser = p.FrameParser()
    frames = []
    offset = 0
    while offset < len(data):
        step = parser.consume(memoryview(data)[offset:])
        if step.issue:
            raise p.ProtocolError(step.issue)
        if not step.consumed_bytes:
            raise AssertionError("parser made no progress")
        offset += step.consumed_bytes
        if step.frame:
            frames.append(step.frame)
    if issue := parser.finish():
        raise p.ProtocolError(issue)
    return frames


def parser_issue(data):
    parser = p.FrameParser()
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
    """The server's view of one event the engine wrote."""
    (frame,) = parse_all(data)
    return p.decode_frame(frame)


def decode_client(data):
    """The engine's view of one client frame the server wrote."""
    ((message, _raw),) = native_peer.ClientFrameReader().feed(data)
    return message


def mutate_u16(data, offset, value):
    result = bytearray(data)
    struct.pack_into("<H", result, offset, value)
    return bytes(result)


def mutate_u32(data, offset, value):
    result = bytearray(data)
    struct.pack_into("<I", result, offset, value)
    return bytes(result)


def mutate_u64(data, offset, value):
    result = bytearray(data)
    struct.pack_into("<Q", result, offset, value)
    return bytes(result)


def exact_json(size):
    if size < 8:
        raise ValueError("JSON size is too small")
    return b'{"x":"' + b"a" * (size - 8) + b'"}'


class ProtocolPythonTests(unittest.TestCase):
    def assert_protocol_error(self, failure_class, code, callback):
        with self.assertRaises(p.ProtocolError) as caught:
            callback()
        self.assertEqual(caught.exception.issue.failure_class, failure_class)
        self.assertEqual(caught.exception.issue.code, code)
        return caught.exception.issue

    def assert_fatal_event(self, code, event):
        """An event that breaks the protocol fails the stream when read."""
        return self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            code,
            lambda: decode_event(native_peer.serialize_event(event)),
        )

    def test_frame_types_and_limits_are_the_shared_ones(self):
        # The values are the engine's; the audio protocol keeps them and adds
        # its own request and audio frames.
        self.assertEqual(
            {frame_type.name: int(frame_type) for frame_type in p.FrameType},
            {
                "CANCEL": 0x0002,
                "STATUS_REQUEST": 0x0004,
                "READY": 0x0100,
                "START": 0x0101,
                "TOKENS": 0x0102,
                "DONE": 0x0104,
                "ERROR": 0x0105,
                "STATUS_JSON": 0x0107,
            },
        )
        self.assertEqual(p.PROTOCOL_VERSION, 8)
        self.assertEqual(p.FRAME_HEADER_BYTES, 16)
        self.assertEqual(p.MAX_TOKEN_BATCH, 4096)
        self.assertEqual(p.MAX_FRAME_PAYLOAD_BYTES, 256 * 1024 * 1024)
        self.assertEqual(p.frame_type_name(p.FrameType.STATUS_JSON), "status_json")
        self.assertEqual(
            p.failure_class_name(p.FailureClass.ENGINE_UNHEALTHY), "engine_unhealthy"
        )
        self.assertEqual(p.issue_code_name(p.IssueCode.BAD_MAGIC), "bad_magic")
        issue = p.ProtocolIssue(
            p.FailureClass.REQUEST_ERROR, p.IssueCode.LIMIT_EXCEEDED, 7, "too long"
        )
        self.assertEqual(
            issue.describe(), "request_error:limit_exceeded request=7: too long"
        )

    def test_client_frames_carry_the_header_and_their_id(self):
        cancel = p.serialize_message(p.CancelFrame(91))
        self.assertEqual(cancel[:4], b"SPLH")
        self.assertEqual(
            struct.unpack_from("<HHQ", cancel, 4),
            (p.PROTOCOL_VERSION, int(p.FrameType.CANCEL), 8),
        )
        self.assertEqual(struct.unpack_from("<Q", cancel, p.FRAME_HEADER_BYTES), (91,))
        self.assertEqual(decode_client(cancel), p.CancelFrame(91))
        status = p.serialize_message(p.StatusRequestFrame(808))
        self.assertEqual(
            struct.unpack_from("<HHQ", status, 4),
            (p.PROTOCOL_VERSION, int(p.FrameType.STATUS_REQUEST), 8),
        )
        self.assertEqual(decode_client(status), p.StatusRequestFrame(808))
        # Status correlation 0 is the engine's own; the server may name it.
        self.assertEqual(
            decode_client(p.serialize_message(p.StatusRequestFrame(0))),
            p.StatusRequestFrame(0),
        )
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.UNKNOWN_FRAME_TYPE,
            lambda: p.serialize_message(p.StartEvent(1, 0, 0)),
        )

    def test_unnamed_request_failures_are_fatal(self):
        """A client frame's failure is its request's error only when the
        frame names the request; one that names none fails the stream, as
        the engine classifies it."""
        for message, code in (
            (p.CancelFrame(0), p.IssueCode.INVALID_REQUEST_ID),
            (p.CancelFrame(-1), p.IssueCode.INVALID_REQUEST_ID),
            (p.CancelFrame(1 << 64), p.IssueCode.INVALID_REQUEST_ID),
            (p.CancelFrame(True), p.IssueCode.INVALID_REQUEST_ID),
            (p.StatusRequestFrame(-1), p.IssueCode.INTEGER_OVERFLOW),
            (p.StatusRequestFrame(1 << 64), p.IssueCode.INTEGER_OVERFLOW),
        ):
            with self.subTest(message=message):
                issue = self.assert_protocol_error(
                    p.FailureClass.PROTOCOL_FATAL,
                    code,
                    lambda message=message: p.serialize_message(message),
                )
                self.assertEqual(issue.request_id, 0)

    def test_done_event_carries_counts_and_no_logits(self):
        done = engine_events()["done"]
        wire = native_peer.serialize_event(done)
        self.assertEqual(len(wire), p.FRAME_HEADER_BYTES + 45)
        self.assertEqual(decode_event(wire), done)
        for event in (
            engine_events()["done_length"],
            engine_events()["done_cancelled"],
        ):
            with self.subTest(event=event):
                decoded = decode_event(native_peer.serialize_event(event))
                self.assertEqual(decoded, event)
                self.assertIs(type(decoded.reason), p.FinishReason)
        # A logit count without its logits fails closed, not truncated.
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: decode_event(mutate_u32(wire, p.FRAME_HEADER_BYTES + 41, 2)),
        )
        # Logits do not fit a Done frame; the header refuses them.
        scored = native_peer.serialize_event(
            p.DoneEvent(91, p.FinishReason.STOP, 4096, 0, 1000, 0, 3500, (1.5, -2.25))
        )
        self.assertEqual(parser_issue(scored).code, p.IssueCode.FRAME_TOO_LARGE)
        # A Done payload without its logit count (41 bytes) is below the minimum.
        short = mutate_u64(wire[: p.FRAME_HEADER_BYTES + 41], 8, 41)
        self.assertEqual(parser_issue(short).code, p.IssueCode.INVALID_PAYLOAD_LENGTH)
        invalid_reason = bytearray(wire)
        invalid_reason[p.FRAME_HEADER_BYTES + 8] = 3
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_ENUM_VALUE,
            lambda: decode_event(bytes(invalid_reason)),
        )
        self.assert_fatal_event(
            p.IssueCode.INVALID_REQUEST_ID,
            p.DoneEvent(0, p.FinishReason.STOP, 4096, 512, 1000, 2000, 3500, ()),
        )

    def test_start_and_tokens_event_validation(self):
        tokens = engine_events()["tokens"]
        self.assertEqual(decode_event(native_peer.serialize_event(tokens)), tokens)
        self.assertEqual(
            decode_event(native_peer.serialize_event(p.StartEvent(91, 2, 4096))),
            p.StartEvent(91, 2, 4096),
        )
        self.assert_fatal_event(p.IssueCode.INVALID_REQUEST_ID, p.StartEvent(0, 0, 0))
        self.assert_fatal_event(
            p.IssueCode.INVALID_REQUEST_ID, p.TokensEvent(0, 0, (1,))
        )
        self.assert_fatal_event(p.IssueCode.LIMIT_EXCEEDED, p.TokensEvent(7, 0, ()))
        self.assert_fatal_event(
            p.IssueCode.INTEGER_OVERFLOW, p.TokensEvent(7, 0xFFFFFFFF, (1, 2))
        )
        batch = p.TokensEvent(7, 0, (1,) * p.MAX_TOKEN_BATCH)
        self.assertEqual(decode_event(native_peer.serialize_event(batch)), batch)
        too_many = native_peer.serialize_event(
            p.TokensEvent(7, 0, (1,) * (p.MAX_TOKEN_BATCH + 1))
        )
        self.assertEqual(parser_issue(too_many).code, p.IssueCode.FRAME_TOO_LARGE)

    def test_client_frames_and_events_each_share_one_stream(self):
        clients = client_messages().values()
        stream = b"".join(p.serialize_message(message) for message in clients)
        self.assertEqual(
            [message for message, _raw in native_peer.ClientFrameReader().feed(stream)],
            list(clients),
        )
        events = engine_events().values()
        stream = b"".join(native_peer.serialize_event(event) for event in events)
        self.assertEqual(
            [p.decode_frame(frame) for frame in parse_all(stream)], list(events)
        )

    def test_one_byte_incremental_parser(self):
        events = list(engine_events().values())
        parser = p.FrameParser()
        frames = []
        for byte in b"".join(native_peer.serialize_event(event) for event in events):
            step = parser.consume(bytes((byte,)))
            self.assertEqual(step.consumed_bytes, 1)
            self.assertIsNone(step.issue)
            if step.frame:
                frames.append(step.frame)
        self.assertIsNone(parser.finish())
        self.assertEqual([p.decode_frame(frame) for frame in frames], events)

    def test_header_failures_are_protocol_fatal(self):
        valid = native_peer.serialize_event(engine_events()["tokens"])
        mutations = []
        bad_magic = bytearray(valid)
        bad_magic[0] = ord("X")
        mutations.append((bytes(bad_magic), p.IssueCode.BAD_MAGIC))
        mutations.extend(
            [
                (mutate_u16(valid, 4, 1), p.IssueCode.UNSUPPORTED_VERSION),
                (mutate_u16(valid, 4, 9), p.IssueCode.UNSUPPORTED_VERSION),
                (mutate_u16(valid, 6, 0x7777), p.IssueCode.UNKNOWN_FRAME_TYPE),
                # The frame types the base no longer defines: version 8's
                # request, mask response, mask request and prompt progress.
                (mutate_u16(valid, 6, 0x0001), p.IssueCode.UNKNOWN_FRAME_TYPE),
                (mutate_u16(valid, 6, 0x0003), p.IssueCode.UNKNOWN_FRAME_TYPE),
                (mutate_u16(valid, 6, 0x0103), p.IssueCode.UNKNOWN_FRAME_TYPE),
                (mutate_u16(valid, 6, 0x0108), p.IssueCode.UNKNOWN_FRAME_TYPE),
                (
                    mutate_u64(valid, 8, 0xFFFFFFFFFFFFFFFF),
                    p.IssueCode.FRAME_TOO_LARGE,
                ),
                (mutate_u64(valid, 8, 15), p.IssueCode.INVALID_PAYLOAD_LENGTH),
                (
                    b"ready\n".ljust(p.FRAME_HEADER_BYTES, b"r"),
                    p.IssueCode.BAD_MAGIC,
                ),
            ]
        )
        # The server never receives the frames it sends.
        mutations.extend(
            (p.serialize_message(message), p.IssueCode.UNKNOWN_FRAME_TYPE)
            for message in client_messages().values()
        )
        for wire, code in mutations:
            with self.subTest(code=code, header=wire[:12].hex()):
                issue = parser_issue(wire)
                self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
                self.assertEqual(issue.code, code)

        # A caller that keeps feeding a failed parser is a bug.
        parser = p.FrameParser()
        first = parser.consume(mutations[0][0])
        self.assertEqual(first.issue.code, p.IssueCode.BAD_MAGIC)
        with self.assertRaisesRegex(RuntimeError, "used after it failed"):
            parser.consume(valid)
        with self.assertRaisesRegex(RuntimeError, "used after it failed"):
            parser.finish()

    def test_every_nonempty_truncation_is_fatal(self):
        wire = native_peer.serialize_event(engine_events()["error_request"])
        for cut in range(1, len(wire)):
            issue = parser_issue(wire[:cut])
            self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
            self.assertEqual(issue.code, p.IssueCode.TRUNCATED_FRAME)
        self.assertIsNone(p.FrameParser().finish())

    def test_malformed_payloads_have_strict_classification(self):
        tokens = native_peer.serialize_event(p.TokensEvent(7, 0, (1, 2)))
        bad_tokens = mutate_u32(tokens, p.FRAME_HEADER_BYTES + 12, 3)
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: decode_event(bad_tokens),
        )

        error = native_peer.serialize_event(
            p.ErrorEvent(p.FailureClass.REQUEST_ERROR, 7, False, b"bad", b"message")
        )
        bad_error = mutate_u32(error, p.FRAME_HEADER_BYTES + 10, 0xFFFFFFFF)
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: decode_event(bad_error),
        )
        self.assert_fatal_event(
            p.IssueCode.INVALID_ERROR_CLASSIFICATION,
            p.ErrorEvent(p.FailureClass.ENGINE_UNHEALTHY, 7, False, b"bad", b""),
        )
        self.assert_fatal_event(
            p.IssueCode.INVALID_ERROR_CLASSIFICATION,
            p.ErrorEvent(p.FailureClass.REQUEST_ERROR, 0, False, b"bad", b""),
        )
        self.assert_fatal_event(
            p.IssueCode.LIMIT_EXCEEDED,
            p.ErrorEvent(p.FailureClass.REQUEST_ERROR, 7, False, b"", b"no code"),
        )
        # A frame the decoder is handed must be one the parser framed.
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.UNKNOWN_FRAME_TYPE,
            lambda: p.decode_frame(p.Frame(0x0001, b"")),
        )
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.UNKNOWN_FRAME_TYPE,
            lambda: p.decode_frame(p.Frame(p.FrameType.CANCEL, bytes(8))),
        )
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: p.decode_frame(p.Frame(p.FrameType.START, bytearray(16))),
        )
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.UNKNOWN_FRAME_TYPE,
            lambda: p.serialize_frame(p.Frame(0x0003, b"")),
        )

    def test_status_json_is_opaque_length_delimited_and_bounded(self):
        event = p.StatusJsonEvent(99, exact_json(p.MAX_STATUS_JSON_BYTES))
        self.assertEqual(decode_event(native_peer.serialize_event(event)), event)
        self.assert_fatal_event(p.IssueCode.LIMIT_EXCEEDED, p.StatusJsonEvent(99, b""))
        # One byte past the limit is refused at the header, before its payload.
        header = native_peer.serialize_event(p.StatusJsonEvent(99, b"{}"))[
            : p.FRAME_HEADER_BYTES
        ]
        too_large = mutate_u64(header, 8, 8 + p.MAX_STATUS_JSON_BYTES + 1)
        issue = parser_issue(too_large)
        self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
        self.assertEqual(issue.code, p.IssueCode.FRAME_TOO_LARGE)

    def test_failure_taxonomy(self):
        errors = (
            p.ErrorEvent(
                p.FailureClass.REQUEST_ERROR,
                5,
                True,
                b"busy",
                b"retry this request",
            ),
            p.ErrorEvent(
                p.FailureClass.ENGINE_UNHEALTHY,
                0,
                False,
                b"metal_error",
                b"replace engine",
            ),
            p.ErrorEvent(
                p.FailureClass.PROTOCOL_FATAL,
                0,
                False,
                b"framing_error",
                b"close stream",
            ),
        )
        for expected in errors:
            self.assertEqual(
                decode_event(native_peer.serialize_event(expected)), expected
            )
        self.assert_fatal_event(p.IssueCode.INVALID_COUNT, p.ReadyEvent(0, 4096, False))
        self.assert_fatal_event(p.IssueCode.INVALID_COUNT, p.ReadyEvent(4, 0, False))
        self.assert_fatal_event(
            p.IssueCode.INVALID_ENUM_VALUE, p.ReadyEvent(4, 4096, 2)
        )

    def test_serialized_frames_round_trip_through_the_parser(self):
        for event in engine_events().values():
            with self.subTest(event=event):
                (frame,) = parse_all(native_peer.serialize_event(event))
                self.assertEqual(
                    p.serialize_frame(frame), native_peer.serialize_event(event)
                )

    def test_value_helpers_the_audio_codec_builds_on(self):
        self.assertEqual(
            p._words((1, 2, 0xFFFFFFFF), "x"), array.array("I", (1, 2, 0xFFFFFFFF))
        )
        for bad, message in (
            ([1, 2], "must be a tuple of uint32"),
            ((1, -1), "x element must be an integer in [0, 4294967295]"),
            ((1, 1 << 32), "x element must be an integer"),
            ((True,), "x element must be an integer"),
            ((1.0,), "x element must be an integer"),
            (("1",), "x element must be an integer"),
        ):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, message.replace("[", "\\[")):
                    p._words(bad, "x")
        self.assertEqual(
            p._float32(0.1, "x"), struct.unpack("<f", struct.pack("<f", 0.1))[0]
        )
        for bad in (True, "1", None, 1e39, float("nan").hex()):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "must be a float32 value"):
                    p._float32(bad, "x")
        self.assertEqual(
            p._tail(struct.pack("<2f", 1.5, -2.0), 0, 2, "f", "value", 0), (1.5, -2.0)
        )
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: p._tail(bytes(8), 0, 3, "I", "value", 7),
        )
        with self.assertRaisesRegex(ValueError, "must be bytes"):
            p._bytes(bytearray(4), "x")

    def test_overflow_lengths_fail_before_allocation(self):
        valid = native_peer.serialize_event(engine_events()["tokens"])
        claimed = struct.unpack_from("<Q", valid, 8)[0]
        issue = parser_issue(mutate_u64(valid, 8, claimed + 1))
        self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
        self.assertEqual(issue.code, p.IssueCode.TRUNCATED_FRAME)

        enormous = mutate_u64(valid, 8, 0xFFFFFFFFFFFFFFFF)
        issue = parser_issue(enormous[: p.FRAME_HEADER_BYTES])
        self.assertEqual(issue.code, p.IssueCode.FRAME_TOO_LARGE)

    def test_random_malformed_inputs_and_valid_frame_mutations_do_not_crash(self):
        random_source = random.Random(0x5EED1234)
        for _ in range(3000):
            data = random_source.randbytes(random_source.randrange(257))
            parser = p.FrameParser()
            offset = 0
            steps = 0
            failed = False
            while offset < len(data) and not failed:
                size = min(random_source.randrange(1, 32), len(data) - offset)
                step = parser.consume(memoryview(data)[offset : offset + size])
                self.assertLessEqual(step.consumed_bytes, size)
                failed = step.issue is not None
                self.assertTrue(step.consumed_bytes or failed)
                offset += step.consumed_bytes
                if step.frame:
                    try:
                        p.decode_frame(step.frame)
                    except p.ProtocolError:
                        pass
                steps += 1
                self.assertLess(steps, 1024)
            if not failed:
                parser.finish()

        valid = native_peer.serialize_event(engine_events()["error_request"])
        for _ in range(2000):
            mutated = bytearray(valid)
            for _ in range(random_source.randrange(1, 5)):
                mutated[random_source.randrange(len(mutated))] = (
                    random_source.randrange(256)
                )
            parser = p.FrameParser()
            offset = 0
            failed = False
            while offset < len(mutated) and not failed:
                step = parser.consume(memoryview(mutated)[offset:])
                failed = step.issue is not None
                self.assertTrue(step.consumed_bytes or failed)
                offset += step.consumed_bytes
                if step.frame:
                    try:
                        p.decode_frame(step.frame)
                    except p.ProtocolError:
                        pass
            if not failed:
                parser.finish()


if __name__ == "__main__":
    unittest.main()
