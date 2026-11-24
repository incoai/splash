"""The version-11 codec against the bytes docs/omni-design.md specifies."""

import struct
import unittest

from dev.tests import native_peer
from server import audio_protocol as p
from server import protocol as v8


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def header(frame_type, payload_bytes):
    return b"SPLH" + struct.pack("<HHQ", 11, frame_type, payload_bytes)


def u32s(*values):
    return struct.pack(f"<{len(values)}I", *values)


def example_speech():
    return p.SpeechRequest(
        request_id=0x0123456789ABCDEF,
        text_tokens=(0, 1, 42, 0xFFFFFFFF),
        language_id=2050,
        temperature=f32(0.9),
        top_k=50,
        top_p=f32(0.95),
        repetition_penalty=f32(1.05),
        predictor_temperature=f32(0.8),
        predictor_top_k=40,
        predictor_top_p=1.0,
        seed=0xFEDCBA9876543210,
        max_frames=100,
        chunk_frames=25,
        reference_text_tokens=(7,),
        reference_codes=tuple(range(16)),
        flags=p.SPEECH_FLAG_CODES,
        instruction_tokens=(151644, 872),
        preset_speaker=3065,
    )


# The example speech request's frame, by field (little-endian).
SPEECH_FRAME = (
    header(0x0011, 176)
    + bytes.fromhex("efcdab8967452301")  # request id
    + bytes.fromhex("01000000")  # flags: the codes
    + bytes.fromhex("02080000")  # language id 2050
    + bytes.fromhex("6666663f")  # temperature 0.9f
    + bytes.fromhex("32000000")  # top_k 50
    + bytes.fromhex("3333733f")  # top_p 0.95f
    + bytes.fromhex("6666863f")  # repetition penalty 1.05f
    + bytes.fromhex("cdcc4c3f")  # predictor temperature 0.8f
    + bytes.fromhex("28000000")  # predictor top_k 40
    + bytes.fromhex("0000803f")  # predictor top_p 1.0f
    + bytes.fromhex("1032547698badcfe")  # seed
    + bytes.fromhex("64000000")  # max frames 100
    + bytes.fromhex("19000000")  # chunk frames 25
    + bytes.fromhex("04000000")  # text count
    + bytes.fromhex("01000000")  # reference text count
    + bytes.fromhex("01000000")  # reference frame count
    + bytes.fromhex("00000000")  # speaker present
    + bytes.fromhex("02000000")  # instruction count
    + bytes.fromhex("f90b0000")  # preset speaker 3065
    + bytes.fromhex("00000000010000002a000000ffffffff")  # text
    + bytes.fromhex("5c50020068030000")  # instruction 151644, 872
    + bytes.fromhex("07000000")  # reference text
    + u32s(*range(16))  # reference codes, one frame
)


def example_transcribe():
    return p.TranscribeRequest(
        request_id=91,
        prefix_tokens=(151644, 8948),
        suffix_tokens=(151670,),
        samples=struct.pack("<2f", 0.5, -0.25),
    )


TRANSCRIBE_FRAME = (
    header(0x0012, 48)
    + bytes.fromhex("5b00000000000000")  # request id 91
    + bytes.fromhex("00000000")  # flags
    + bytes.fromhex("00000000")  # max tokens: default
    + bytes.fromhex("02000000")  # prefix count
    + bytes.fromhex("01000000")  # suffix count
    + bytes.fromhex("02000000")  # sample count
    + bytes.fromhex("5c500200f4220000")  # prefix 151644, 8948
    + bytes.fromhex("76500200")  # suffix 151670
    + bytes.fromhex("0000003f000080be")  # samples 0.5f, -0.25f
)
CANCEL_FRAME = header(0x0002, 8) + bytes.fromhex("5b00000000000000")
STATUS_REQUEST_FRAME = header(0x0004, 8) + bytes.fromhex("2803000000000000")

READY_PAYLOAD = bytes.fromhex("080000000020000001")
START_PAYLOAD = bytes.fromhex("5b000000000000000200000000000000")
TOKENS_PAYLOAD = bytes.fromhex(
    "5b0000000000000011000000030000000a0000000b0000000c000000"
)
AUDIO_PAYLOAD = bytes.fromhex("5b00000000000000800700000300000064009cffff7f")
CODES_PAYLOAD = bytes.fromhex("5b0000000000000001000000") + u32s(*range(2032, 2048))
DONE_PAYLOAD = bytes.fromhex(
    "5b00000000000000"
    "00"
    "0c000000"
    "05000000"
    "e803000000000000"
    "d007000000000000"
    "ac0d000000000000"
    "00000000"
)
ERROR_PAYLOAD = (
    bytes.fromhex("01015b000000000000000e00000008000000")
    + b"limit_exceeded"
    + b"too long"
)
STATUS_JSON_PAYLOAD = bytes.fromhex("2803000000000000") + b'{"ready":true}'


def engine_events():
    return {
        "ready": (0x0100, READY_PAYLOAD, p.ReadyEvent(8, 8192, p.ModelKind.TTS)),
        "start": (0x0101, START_PAYLOAD, p.StartEvent(91, 2, 0)),
        "tokens": (0x0102, TOKENS_PAYLOAD, p.TokensEvent(91, 17, (10, 11, 12))),
        "audio": (
            0x0110,
            AUDIO_PAYLOAD,
            p.AudioEvent(91, 1920, struct.pack("<3h", 100, -100, 32767)),
        ),
        "codes": (0x0111, CODES_PAYLOAD, p.CodesEvent(91, tuple(range(2032, 2048)))),
        "done": (
            0x0104,
            DONE_PAYLOAD,
            p.DoneEvent(91, p.FinishReason.STOP, 12, 5, 1000, 2000, 3500, ()),
        ),
        "error": (
            0x0105,
            ERROR_PAYLOAD,
            p.ErrorEvent(
                p.FailureClass.REQUEST_ERROR, 91, True, b"limit_exceeded", b"too long"
            ),
        ),
        "status_json": (
            0x0107,
            STATUS_JSON_PAYLOAD,
            p.StatusJsonEvent(808, b'{"ready":true}'),
        ),
    }


class AudioProtocolGoldenTests(unittest.TestCase):
    def test_frame_types_and_version_are_the_documents(self):
        self.assertEqual(p.PROTOCOL_VERSION, 11)
        self.assertEqual(p.FRAME_HEADER_BYTES, 16)
        self.assertEqual(
            {name: int(value) for name, value in p.FrameType.__members__.items()},
            {
                "CANCEL": 0x0002,
                "STATUS_REQUEST": 0x0004,
                "SPEECH_REQUEST": 0x0011,
                "TRANSCRIBE_REQUEST": 0x0012,
                "READY": 0x0100,
                "START": 0x0101,
                "TOKENS": 0x0102,
                "DONE": 0x0104,
                "ERROR": 0x0105,
                "STATUS_JSON": 0x0107,
                "AUDIO": 0x0110,
                "CODES": 0x0111,
            },
        )
        self.assertEqual(p.SPEECH_FIXED_BYTES, 84)
        self.assertEqual(p.TRANSCRIBE_FIXED_BYTES, 28)
        self.assertEqual(p.MAX_SAMPLES, 19_200_000)
        self.assertEqual(p.MAX_TEXT_TOKENS, 4096)
        self.assertEqual(p.MAX_INSTRUCTION_TOKENS, 512)
        self.assertEqual(p.MAX_FRAME_PAYLOAD_BYTES, 256 * 1024 * 1024)
        self.assertEqual(p.SPEECH_FLAG_CODES, 1)

    def test_client_frames_serialize_to_their_golden_bytes(self):
        for name, message, expected in (
            ("speech", example_speech(), SPEECH_FRAME),
            ("transcribe", example_transcribe(), TRANSCRIBE_FRAME),
            ("cancel", p.CancelFrame(91), CANCEL_FRAME),
            ("status_request", p.StatusRequestFrame(808), STATUS_REQUEST_FRAME),
        ):
            with self.subTest(name=name):
                encoded = bytes(p.serialize_message(message))
                self.assertEqual(encoded.hex(), expected.hex())
                # The engine's reader (native_peer) reads the same message back.
                ((decoded, raw),) = native_peer.ClientFrameReader(audio=True).feed(
                    encoded
                )
                self.assertEqual(raw, expected)
                self.assertEqual(decoded, message)

    def test_speech_request_with_speaker_embedding(self):
        request = p.SpeechRequest(
            request_id=5,
            text_tokens=(1,),
            speaker_embedding=tuple(float(i) for i in range(p.SPEAKER_EMBEDDING_DIMS)),
        )
        encoded = bytes(p.serialize_message(request))
        self.assertEqual(len(encoded), 16 + 84 + 4 + 4 * p.SPEAKER_EMBEDDING_DIMS)
        self.assertEqual(encoded[16 + 72 : 16 + 76], u32s(1))
        ((decoded, _),) = native_peer.ClientFrameReader(audio=True).feed(encoded)
        self.assertEqual(decoded.speaker_embedding, request.speaker_embedding)

    def test_engine_events_decode_from_their_golden_bytes(self):
        for name, (frame_type, payload, expected) in engine_events().items():
            with self.subTest(name=name):
                frame = p.Frame(p.FrameType(frame_type), payload)
                self.assertEqual(p.decode_frame(frame), expected)
                encoded = p.serialize_frame(frame)
                self.assertEqual(encoded, header(frame_type, len(payload)) + payload)
                self.assertEqual(native_peer.serialize_audio_event(expected), encoded)

    def test_parser_splits_a_stream_of_events_at_any_boundary(self):
        events = engine_events()
        stream = b"".join(
            header(frame_type, len(payload)) + payload
            for frame_type, payload, _ in events.values()
        )
        for chunk_size in (1, 7, 16, 23, len(stream)):
            with self.subTest(chunk_size=chunk_size):
                parser = p.FrameParser()
                decoded = []
                for start in range(0, len(stream), chunk_size):
                    chunk = stream[start : start + chunk_size]
                    offset = 0
                    while offset < len(chunk):
                        step = parser.consume(memoryview(chunk)[offset:])
                        self.assertIsNone(step.issue)
                        offset += step.consumed_bytes
                        if step.frame:
                            decoded.append(p.decode_frame(step.frame))
                self.assertEqual(decoded, [event for _, _, event in events.values()])
                self.assertIsNone(parser.finish())

    def test_parser_refuses_version_8_and_client_frames(self):
        parser = p.FrameParser()
        step = parser.consume(
            native_peer.serialize_event(v8.ReadyEvent(4, 1024, False))
        )
        self.assertEqual(step.issue.code, p.IssueCode.UNSUPPORTED_VERSION)
        for frame in (SPEECH_FRAME, CANCEL_FRAME):
            parser = p.FrameParser()
            step = parser.consume(frame)
            self.assertEqual(step.issue.code, p.IssueCode.UNKNOWN_FRAME_TYPE)
        parser = p.FrameParser()
        step = parser.consume(header(0x0104, 44))
        self.assertEqual(step.issue.code, p.IssueCode.INVALID_PAYLOAD_LENGTH)
        parser = p.FrameParser()
        step = parser.consume(header(0x0110, p.MAX_FRAME_PAYLOAD_BYTES + 1))
        self.assertEqual(step.issue.code, p.IssueCode.FRAME_TOO_LARGE)
        parser = p.FrameParser()
        parser.consume(header(0x0110, 20)[:10])
        self.assertEqual(parser.finish().code, p.IssueCode.TRUNCATED_FRAME)


class AudioProtocolValidationTests(unittest.TestCase):
    def assert_request_issue(self, message, code, request_id=None):
        with self.assertRaises(p.ProtocolError) as refused:
            p.serialize_message(message)
        issue = refused.exception.issue
        self.assertEqual(issue.code, code, issue.describe())
        if request_id is None:
            self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
        else:
            self.assertEqual(issue.failure_class, p.FailureClass.REQUEST_ERROR)
            self.assertEqual(issue.request_id, request_id)

    def test_speech_request_rules(self):
        good = example_speech()
        cases = (
            (dict(request_id=0), p.IssueCode.INVALID_REQUEST_ID, None),
            (dict(flags=2), p.IssueCode.INVALID_ENUM_VALUE, good.request_id),
            (dict(language_id=-1), p.IssueCode.INVALID_ENUM_VALUE, good.request_id),
            (dict(text_tokens=()), p.IssueCode.LIMIT_EXCEEDED, good.request_id),
            (
                dict(text_tokens=(1,) * (p.MAX_TEXT_TOKENS + 1)),
                p.IssueCode.LIMIT_EXCEEDED,
                good.request_id,
            ),
            (
                dict(text_tokens=(1, 1 << 32)),
                p.IssueCode.LIMIT_EXCEEDED,
                good.request_id,
            ),
            (
                dict(instruction_tokens=(1,) * (p.MAX_INSTRUCTION_TOKENS + 1)),
                p.IssueCode.LIMIT_EXCEEDED,
                good.request_id,
            ),
            (dict(preset_speaker=-1), p.IssueCode.INVALID_ENUM_VALUE, good.request_id),
            (
                dict(reference_codes=(1,) * 15),
                p.IssueCode.INVALID_COUNT,
                good.request_id,
            ),
            (
                dict(speaker_embedding=(0.0,) * 2047),
                p.IssueCode.INVALID_COUNT,
                good.request_id,
            ),
            (
                dict(speaker_embedding=(float("nan"),) * 2048),
                p.IssueCode.INVALID_COUNT,
                good.request_id,
            ),
            (dict(temperature=-0.1), p.IssueCode.INVALID_SAMPLING, good.request_id),
            (dict(temperature=1e-45), p.IssueCode.INVALID_SAMPLING, good.request_id),
            (
                dict(temperature=float("nan")),
                p.IssueCode.INVALID_SAMPLING,
                good.request_id,
            ),
            (dict(top_p=0.0), p.IssueCode.INVALID_SAMPLING, good.request_id),
            (dict(top_p=1.5), p.IssueCode.INVALID_SAMPLING, good.request_id),
            (dict(top_k=-1), p.IssueCode.INVALID_SAMPLING, good.request_id),
            (
                dict(repetition_penalty=0.0),
                p.IssueCode.INVALID_SAMPLING,
                good.request_id,
            ),
            (
                dict(predictor_top_p=2.0),
                p.IssueCode.INVALID_SAMPLING,
                good.request_id,
            ),
            (dict(seed=1 << 64), p.IssueCode.INTEGER_OVERFLOW, good.request_id),
            (dict(max_frames=1 << 32), p.IssueCode.INTEGER_OVERFLOW, good.request_id),
        )
        for changes, code, request_id in cases:
            with self.subTest(changes=changes):
                self.assert_request_issue(
                    p.SpeechRequest(**{**vars_of(good), **changes}), code, request_id
                )
        self.assertEqual(
            bytes(p.serialize_message(p.SpeechRequest(1, (1,), temperature=0.0)))[
                16 + 16 : 16 + 20
            ],
            bytes(4),
        )

    def test_transcribe_request_rules(self):
        good = example_transcribe()
        cases = (
            (dict(request_id=0), p.IssueCode.INVALID_REQUEST_ID, None),
            (dict(flags=2), p.IssueCode.INVALID_ENUM_VALUE, 91),
            (dict(samples=b""), p.IssueCode.LIMIT_EXCEEDED, 91),
            (dict(samples=b"\x00" * 6), p.IssueCode.LIMIT_EXCEEDED, 91),
            (
                dict(samples=bytes(4 * (p.MAX_SAMPLES + 1))),
                p.IssueCode.LIMIT_EXCEEDED,
                91,
            ),
            (
                dict(prefix_tokens=(1,) * p.MAX_TEXT_TOKENS, suffix_tokens=(1,)),
                p.IssueCode.LIMIT_EXCEEDED,
                91,
            ),
            (dict(max_tokens=-1), p.IssueCode.INTEGER_OVERFLOW, 91),
        )
        for changes, code, request_id in cases:
            with self.subTest(changes=changes):
                self.assert_request_issue(
                    p.TranscribeRequest(**{**vars_of(good), **changes}),
                    code,
                    request_id,
                )
        self.assertEqual(p.float_samples(example_transcribe().samples), (0.5, -0.25))
        self.assertEqual(p.pcm_samples(struct.pack("<2h", 7, -7)), (7, -7))

    def test_other_messages_are_refused(self):
        # The base protocol's events are not client messages.
        self.assert_request_issue(
            v8.StartEvent(1, 0, 0), p.IssueCode.UNKNOWN_FRAME_TYPE
        )
        self.assert_request_issue(p.CancelFrame(0), p.IssueCode.INVALID_REQUEST_ID)

    def test_event_rules(self):
        cases = (
            (
                0x0100,
                bytes.fromhex("080000000020000003"),
                p.IssueCode.INVALID_ENUM_VALUE,
            ),
            (
                0x0100,
                bytes.fromhex("000000000020000001"),
                p.IssueCode.INVALID_COUNT,
            ),
            (0x0110, AUDIO_PAYLOAD[:-2], p.IssueCode.INVALID_PAYLOAD_LENGTH),
            (
                0x0110,
                bytes(8) + AUDIO_PAYLOAD[8:],
                p.IssueCode.INVALID_REQUEST_ID,
            ),
            (
                0x0110,
                bytes.fromhex("5b00000000000000ffffffff010000000000"),
                p.IssueCode.INTEGER_OVERFLOW,
            ),
            (
                0x0104,
                DONE_PAYLOAD[:-4] + u32s(1) + struct.pack("<f", 1.0),
                p.IssueCode.FRAME_TOO_LARGE,
            ),
            (0x0111, CODES_PAYLOAD[:-4], p.IssueCode.INVALID_PAYLOAD_LENGTH),
            (0x0111, bytes(8) + CODES_PAYLOAD[8:], p.IssueCode.INVALID_REQUEST_ID),
            (0x0111, CODES_PAYLOAD[:8], p.IssueCode.INVALID_PAYLOAD_LENGTH),
            (0x0101, bytes(16), p.IssueCode.INVALID_REQUEST_ID),
            (0x0102, TOKENS_PAYLOAD[:-4], p.IssueCode.INVALID_PAYLOAD_LENGTH),
            (0x0011, SPEECH_FRAME[16:], p.IssueCode.UNKNOWN_FRAME_TYPE),
        )
        for frame_type, payload, code in cases:
            with self.subTest(frame_type=frame_type, code=code):
                with self.assertRaises(p.ProtocolError) as refused:
                    p.decode_frame(p.Frame(frame_type, payload))
                self.assertEqual(refused.exception.issue.code, code)
        # An empty audio chunk is taken as it is; a Done event's logit count
        # of zero is the only one its layout allows.
        self.assertEqual(
            p.decode_frame(p.Frame(0x0110, AUDIO_PAYLOAD[:12] + u32s(0))).sample_count,
            0,
        )
        with self.assertRaises(p.ProtocolError) as refused:
            p.decode_frame(p.Frame(0x0104, DONE_PAYLOAD[:-4] + u32s(2)))
        self.assertEqual(
            refused.exception.issue.code, p.IssueCode.INVALID_PAYLOAD_LENGTH
        )


def vars_of(message):
    return {name: getattr(message, name) for name in type(message).__dataclass_fields__}


if __name__ == "__main__":
    unittest.main()
