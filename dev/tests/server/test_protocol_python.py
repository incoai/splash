import itertools
import random
import struct
import tracemalloc
import unittest
from dataclasses import astuple, replace
from pathlib import Path
from unittest import mock

from dev.tests import native_peer
from server import protocol as p

ROOT = Path(__file__).parents[3]
GOLDEN = ROOT / "dev/tests/engine/protocol_golden.txt"

# The request's fixed fields in wire order, and each one's payload offset.
REQUEST_FIELDS = (
    "request_id",
    "priority",
    "constraint",
    "absolute_deadline",
    "remaining_deadline",
    "max_output",
    "prompt_count",
    "image_span_count",
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "min_p",
    "seed",
    "score_count",
    "generation_prompt_tokens",
    "flags",
)
assert len(REQUEST_FIELDS) == len(p._REQUEST.format) - 1
OFFSET = dict(
    zip(
        REQUEST_FIELDS,
        itertools.accumulate(
            (struct.calcsize("<" + code) for code in p._REQUEST.format[1:]),
            initial=0,
        ),
    )
)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def example_request():
    return native_peer.request_frame(
        request_id=0x0123456789ABCDEF,
        priority=p.RequestPriority.FOREGROUND,
        absolute_deadline_unix_micros=1_800_000_000_000_000,
        remaining_deadline_micros=45_000_000,
        logical_max_output_tokens=32_768,
        prompt_tokens=(0, 1, 42, 0x80000000, 0xFFFFFFFF),
        sampling=p.SamplingParameters(
            f32(0.8), f32(0.95), 32, 1.5, -0.25, f32(1.1), f32(0.05)
        ),
        seed=0xFEDCBA9876543210,
        constraint=p.ConstraintMode.TOKEN_MASK,
        generation_prompt_tokens=2,
    )


def example_image_request():
    span = p.ImageSpan(1, 1, 2, 2, 0x1111222233334444, 0x5555666677778888)
    return replace(
        example_request(),
        prompt_tokens=(7, 3, 9),
        image_spans=(span,),
        image_pixels=bytes((i * 7 + 1) & 0xFF for i in range(span.pixel_bytes)),
    )


def example_score_request():
    return replace(
        example_request(),
        logical_max_output_tokens=0,
        prompt_tokens=(5, 6, 7),
        sampling=p.SamplingParameters(),
        constraint=p.ConstraintMode.NONE,
        score_tokens=(101, 202, 303),
    )


def example_ignore_eos_request():
    return replace(
        example_request(),
        constraint=p.ConstraintMode.NONE,
        flags=p.RequestFlag.IGNORE_END_OF_SEQUENCE,
    )


def example_progress_request():
    return replace(example_request(), flags=p.RequestFlag.RETURN_PROGRESS)


def client_messages():
    return {
        "request": example_request(),
        "request_image": example_image_request(),
        "request_score": example_score_request(),
        "request_ignore_eos": example_ignore_eos_request(),
        "request_progress": example_progress_request(),
        "cancel": p.CancelFrame(91),
        "mask_response": p.MaskResponseFrame(
            91, 7, struct.pack("<3I", 0xFFFFFFFF, 0, 0xA5A5A5A5)
        ),
        "status_request": p.StatusRequestFrame(808),
    }


def engine_events():
    return {
        "ready": p.ReadyEvent(4, 524_288, False),
        "ready_vision": p.ReadyEvent(4, 524_288, True),
        "start": p.StartEvent(91, 2, 4096),
        "prompt_progress": p.PromptProgressEvent(91, 2048, 123456),
        "tokens": p.TokensEvent(91, 17, (10, 11, 12)),
        "mask_request_initial": p.MaskRequestEvent(91, 6, 4, ()),
        "mask_request_verify": p.MaskRequestEvent(91, 7, 4, (101, 102, 103)),
        "done": p.DoneEvent(91, p.FinishReason.STOP, 4096, 512, 1000, 2000, 3500, ()),
        "done_scored": p.DoneEvent(
            91, p.FinishReason.STOP, 4096, 0, 1000, 0, 3500, (1.5, -2.25, 0.5)
        ),
        "done_scored_readout": p.DoneEvent(
            91,
            p.FinishReason.STOP,
            4096,
            0,
            1000,
            0,
            3500,
            (1.5, -2.25),
            12.75,
            (7, 3, 11),
            (9.5, 9.25, -1.0),
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


def read_golden():
    golden = {}
    for line in GOLDEN.read_text().splitlines():
        if line and not line.startswith("#"):
            name, data = line.split()
            golden[name] = bytes.fromhex(data)
    return golden


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

    def test_client_frames_match_golden(self):
        golden = read_golden()
        for name, message in client_messages().items():
            with self.subTest(name=name):
                self.assertEqual(p.serialize_message(message).hex(), golden[name].hex())

    def test_events_decode_from_golden(self):
        golden = read_golden()
        for name, event in engine_events().items():
            with self.subTest(name=name):
                self.assertEqual(decode_event(golden[name]), event)

    def test_progress_wire_validation(self):
        # A request asks for its prompt's progress with a flag bit.
        request = example_progress_request()
        wire = p.serialize_message(request)
        self.assertEqual(
            struct.unpack_from("<I", wire, p.FRAME_HEADER_BYTES + OFFSET["flags"])[0], 2
        )
        self.assertEqual(decode_client(wire), request)
        for event in (
            p.PromptProgressEvent(0, 1, 0),
            p.PromptProgressEvent(1, p.MAX_PROMPT_TOKENS + 1, 0),
        ):
            with self.subTest(event=event):
                self.assert_fatal_event(p.IssueCode.INVALID_COUNT, event)

    def test_score_request_wire_layout_and_roundtrip(self):
        request = example_score_request()
        wire = p.serialize_message(request)
        self.assertEqual(
            struct.unpack_from(
                "<I", wire, p.FRAME_HEADER_BYTES + OFFSET["score_count"]
            )[0],
            len(request.score_tokens),
        )
        self.assertEqual(
            struct.unpack_from(
                "<3I", wire, p.FRAME_HEADER_BYTES + p.REQUEST_FIXED_BYTES + 4 * 3
            ),
            request.score_tokens,
        )
        self.assertEqual(decode_client(wire), request)

    def test_maximum_score_domain_roundtrips_without_truncation(self):
        request = replace(example_score_request(), score_tokens=tuple(range(255)))
        self.assertEqual(decode_client(p.serialize_message(request)), request)
        done = p.DoneEvent(
            91,
            p.FinishReason.STOP,
            3,
            0,
            1000,
            0,
            1000,
            tuple(float(index) for index in range(255)),
        )
        self.assertEqual(decode_event(native_peer.serialize_event(done)), done)

    def test_penalty_and_min_p_ranges_are_refused_by_the_encoder(self):
        request = example_request()
        wire = p.serialize_message(request)
        # presence, frequency, repetition and min_p follow top_k in the frame.
        offset = p.FRAME_HEADER_BYTES + OFFSET["presence_penalty"]
        names = ("presence_penalty", "frequency_penalty", "repetition_penalty", "min_p")
        self.assertEqual(
            struct.unpack_from("<4f", wire, offset),
            (1.5, -0.25, f32(1.1), f32(0.05)),
        )
        nan, inf = float("nan"), float("inf")
        for field, value in (
            (0, 2.5),
            (0, nan),
            (1, -2.5),
            (1, -inf),
            (2, 0.0),
            (2, -1.0),
            (2, inf),
            (3, -0.01),
            (3, 1.01),
            (3, nan),
            (3, inf),
        ):
            with self.subTest(field=field, value=value):
                invalid = replace(
                    request, sampling=replace(request.sampling, **{names[field]: value})
                )
                self.assert_protocol_error(
                    p.FailureClass.REQUEST_ERROR,
                    p.IssueCode.INVALID_SAMPLING,
                    lambda invalid=invalid: p.serialize_message(invalid),
                )
        # The kernels divide by a sampling temperature, and Metal flushes a
        # subnormal one to zero; one that rounds to zero in float32 is greedy.
        for value in (1e-40, float.fromhex("0x1.fffffcp-127")):
            with self.subTest(temperature=value):
                invalid = replace(
                    request, sampling=replace(request.sampling, temperature=value)
                )
                self.assert_protocol_error(
                    p.FailureClass.REQUEST_ERROR,
                    p.IssueCode.INVALID_SAMPLING,
                    lambda invalid=invalid: p.serialize_message(invalid),
                )
        for value in (float.fromhex("0x1p-126"), 1e-46):
            with self.subTest(temperature=value):
                valid = replace(
                    request, sampling=replace(request.sampling, temperature=value)
                )
                decoded = decode_client(p.serialize_message(valid))
                self.assertEqual(decoded.sampling.temperature, f32(value))
        # The limits, and any top_k, 0 keeping every token.
        for limit in (
            p.SamplingParameters(f32(0.8), f32(0.95), 32, -2.0, 2.0, 2.0**-149, 0.0),
            p.SamplingParameters(f32(0.8), f32(0.95), 32, 2.0, -2.0, f32(3.4e38), 1.0),
            p.SamplingParameters(f32(0.8), f32(0.95), 0),
            p.SamplingParameters(1.0, 1.0, 0xFFFFFFFF),
        ):
            valid = replace(request, sampling=limit)
            self.assertEqual(decode_client(p.serialize_message(valid)), valid)
        score = example_score_request()
        for name in names:
            with self.subTest(score=name):
                invalid = replace(
                    score, sampling=replace(score.sampling, **{name: 0.5})
                )
                self.assert_protocol_error(
                    p.FailureClass.REQUEST_ERROR,
                    p.IssueCode.INVALID_SAMPLING,
                    lambda invalid=invalid: p.serialize_message(invalid),
                )

    def test_request_frame_copies_pixels_once(self):
        span = p.ImageSpan(1, 56 * 48, 112, 96, 1, 2)
        request = replace(
            example_request(),
            prompt_tokens=tuple(range(span.tokens + 2)),
            image_spans=(span,),
            image_pixels=bytes(range(256)) * (span.pixel_bytes // 256),
        )
        tracemalloc.start()
        try:
            frame = p.serialize_message(request)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLessEqual(peak, len(frame) + 1024 * 1024)
        payload = (
            struct.pack(
                p._REQUEST.format,
                request.request_id,
                int(request.priority),
                int(request.constraint),
                request.absolute_deadline_unix_micros,
                request.remaining_deadline_micros,
                request.logical_max_output_tokens,
                len(request.prompt_tokens),
                len(request.image_spans),
                *astuple(request.sampling),
                request.seed,
                len(request.score_tokens),
                request.generation_prompt_tokens,
                int(request.flags),
            )
            + struct.pack(f"<{len(request.prompt_tokens)}I", *request.prompt_tokens)
            + struct.pack(p._IMAGE_SPAN.format, *astuple(span))
            + request.image_pixels
        )
        header = struct.pack(
            p._HEADER.format,
            b"SPLH",
            p.PROTOCOL_VERSION,
            int(p.FrameType.REQUEST),
            len(payload),
        )
        self.assertEqual(frame, header + payload)

    def test_request_validation_runs_once(self):
        labels = []
        words = p._words

        def counted(values, label):
            labels.append(label)
            return words(values, label)

        with mock.patch.object(p, "_words", counted):
            p.serialize_message(example_score_request())
        self.assertEqual(sorted(labels), ["prompt tokens", "score tokens"])

    def test_score_request_rejects_generation_combinations(self):
        base = example_score_request()
        cases = (
            (replace(base, logical_max_output_tokens=8), p.IssueCode.INVALID_COUNT),
            (replace(base, score_tokens=(101,)), p.IssueCode.INVALID_COUNT),
            (replace(base, score_tokens=(101, 101)), p.IssueCode.INVALID_COUNT),
            (replace(base, score_tokens=tuple(range(256))), p.IssueCode.INVALID_COUNT),
            (
                replace(base, sampling=p.SamplingParameters(0.5, 1.0, 8)),
                p.IssueCode.INVALID_SAMPLING,
            ),
            (
                replace(base, sampling=p.SamplingParameters(0.0, 0.5, 0)),
                p.IssueCode.INVALID_SAMPLING,
            ),
            (
                replace(base, constraint=p.ConstraintMode.TOKEN_MASK),
                p.IssueCode.INVALID_CONSTRAINT,
            ),
            (
                replace(
                    base,
                    image_spans=(p.ImageSpan(0, 1, 2, 2, 1, 2),),
                    image_pixels=bytes(48),
                ),
                p.IssueCode.INVALID_COUNT,
            ),
        )
        for request, code in cases:
            with self.subTest(request=request):
                issue = self.assert_protocol_error(
                    p.FailureClass.REQUEST_ERROR,
                    code,
                    lambda request=request: p.serialize_message(request),
                )
                self.assertEqual(issue.request_id, request.request_id)
        # Ordinary generation still requires a positive output budget.
        self.assert_protocol_error(
            p.FailureClass.REQUEST_ERROR,
            p.IssueCode.LIMIT_EXCEEDED,
            lambda: p.serialize_message(
                replace(example_request(), logical_max_output_tokens=0)
            ),
        )

    def test_done_option_logits_wire_layout_and_roundtrip(self):
        done = p.DoneEvent(
            91, p.FinishReason.STOP, 4096, 0, 1000, 0, 3500, (1.5, -2.25, 0.5)
        )
        wire = native_peer.serialize_event(done)
        self.assertEqual(decode_event(wire), done)
        # A wrong logit count must fail closed, not truncate.
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: decode_event(mutate_u32(wire, p.FRAME_HEADER_BYTES + 41, 2)),
        )
        # A Done payload without its option-logit count (41 bytes) is below
        # the minimum.
        short = mutate_u64(wire[: p.FRAME_HEADER_BYTES + 41], 8, 41)
        self.assertEqual(parser_issue(short).code, p.IssueCode.INVALID_PAYLOAD_LENGTH)

    def test_done_option_logits_validation(self):
        base = dict(
            request_id=91,
            reason=p.FinishReason.STOP,
            prompt_tokens=4,
            completion_tokens=0,
            prefill_micros=100,
            decode_micros=0,
            wall_micros=200,
        )
        # Each scored-done invariant fails closed on the wire: too few or
        # non-finite logits, a finish other than stop, completion tokens or
        # decode time.
        for logits in ((1.0,), (1.0, float("nan")), (1.0, float("inf"))):
            with self.subTest(logits=logits):
                self.assert_fatal_event(
                    p.IssueCode.INVALID_COUNT,
                    p.DoneEvent(**base, option_logits=logits),
                )
        for field, value in (
            ("reason", p.FinishReason.LENGTH),
            ("reason", p.FinishReason.CANCELLED),
            ("completion_tokens", 1),
            ("decode_micros", 1),
        ):
            with self.subTest(field=field, value=value):
                self.assert_fatal_event(
                    p.IssueCode.INVALID_COUNT,
                    p.DoneEvent(**{**base, field: value}, option_logits=(1.0, 2.0)),
                )
        # More logits than score options do not fit a Done frame.
        self.assert_fatal_event(
            p.IssueCode.FRAME_TOO_LARGE,
            p.DoneEvent(**base, option_logits=tuple(float(i) for i in range(256))),
        )
        # Scored, generation and cancelled done events decode unchanged.
        for event in (
            p.DoneEvent(**base, option_logits=(1.0, 2.0)),
            p.DoneEvent(91, p.FinishReason.STOP, 4096, 512, 1000, 2000, 3500, ()),
            p.DoneEvent(91, p.FinishReason.LENGTH, 4096, 512, 1000, 2000, 3500, ()),
            p.DoneEvent(91, p.FinishReason.CANCELLED, 4096, 128, 1000, 500, 1500, ()),
        ):
            with self.subTest(event=event):
                decoded = decode_event(native_peer.serialize_event(event))
                self.assertEqual(decoded, event)
                self.assertIs(type(decoded.reason), p.FinishReason)

    def test_request_header_and_binary_prompt(self):
        request = example_request()
        wire = p.serialize_message(request)
        self.assertEqual(wire[:4], b"SPLH")
        self.assertEqual(
            struct.unpack_from("<HHQ", wire, 4),
            (
                p.PROTOCOL_VERSION,
                int(p.FrameType.REQUEST),
                p.REQUEST_FIXED_BYTES + 4 * len(request.prompt_tokens),
            ),
        )
        self.assertEqual(
            struct.unpack_from("<Q", wire, p.FRAME_HEADER_BYTES)[0], request.request_id
        )
        self.assertEqual(
            struct.unpack_from(
                "<I", wire, p.FRAME_HEADER_BYTES + OFFSET["prompt_count"]
            )[0],
            5,
        )
        self.assertEqual(
            struct.unpack_from(
                "<I", wire, p.FRAME_HEADER_BYTES + OFFSET["image_span_count"]
            )[0],
            0,
        )
        self.assertEqual(
            struct.unpack_from(
                "<II", wire, p.FRAME_HEADER_BYTES + OFFSET["generation_prompt_tokens"]
            ),
            (request.generation_prompt_tokens, request.flags),
        )
        self.assertEqual(
            struct.unpack_from(
                "<5I", wire, p.FRAME_HEADER_BYTES + p.REQUEST_FIXED_BYTES
            ),
            request.prompt_tokens,
        )

        image = example_image_request()
        wire = p.serialize_message(image)
        span_offset = (
            p.FRAME_HEADER_BYTES + p.REQUEST_FIXED_BYTES + 4 * len(image.prompt_tokens)
        )
        self.assertEqual(
            struct.unpack_from(
                "<I", wire, p.FRAME_HEADER_BYTES + OFFSET["image_span_count"]
            )[0],
            1,
        )
        self.assertEqual(
            struct.unpack_from("<IIIIQQ", wire, span_offset),
            (1, 1, 2, 2, 0x1111222233334444, 0x5555666677778888),
        )
        self.assertEqual(wire[span_offset + 32 :], image.image_pixels)
        self.assertEqual(decode_client(wire), image)

    def test_generation_prompt_must_leave_a_prompt_token(self):
        request = example_request()
        for tokens in (len(request.prompt_tokens), 0xFFFFFFFF):
            with self.subTest(tokens=tokens):
                issue = self.assert_protocol_error(
                    p.FailureClass.REQUEST_ERROR,
                    p.IssueCode.INVALID_COUNT,
                    lambda tokens=tokens: p.serialize_message(
                        replace(request, generation_prompt_tokens=tokens)
                    ),
                )
                self.assertEqual(issue.request_id, request.request_id)

    def test_request_flags_follow_the_generation_prompt(self):
        request = example_ignore_eos_request()
        wire = p.serialize_message(request)
        self.assertEqual(
            struct.unpack_from("<I", wire, p.FRAME_HEADER_BYTES + OFFSET["flags"])[0], 1
        )
        self.assertEqual(decode_client(wire), request)
        # An undefined bit, or ignoring end-of-sequence where a grammar or
        # scoring decides the output, fails the request.
        cases = [
            (replace(request, flags=flags), p.IssueCode.INVALID_ENUM_VALUE)
            for flags in (1 << 2, 1 << 31, 0xFFFFFFFF, True, -1, 1 << 32, 1.0)
        ] + [
            (
                replace(base, flags=p.RequestFlag.IGNORE_END_OF_SEQUENCE),
                p.IssueCode.INVALID_CONSTRAINT,
            )
            for base in (example_request(), example_score_request())
        ]
        for invalid, code in cases:
            with self.subTest(flags=invalid.flags, code=code):
                issue = self.assert_protocol_error(
                    p.FailureClass.REQUEST_ERROR,
                    code,
                    lambda invalid=invalid: p.serialize_message(invalid),
                )
                self.assertEqual(issue.request_id, request.request_id)

    def test_token_words_must_be_exact_uint32_ints(self):
        base = example_request()
        bad_values = (
            True,
            p.ConstraintMode.NONE,
            -1,
            0x100000000,
            1.0,
            "1",
            None,
        )
        for field, message in (
            ("prompt_tokens", "prompt tokens element"),
            ("score_tokens", "score tokens element"),
        ):
            # A bad word is caught wherever it sits, including the last one.
            for position in (0, 4):
                for bad in bad_values:
                    words = [*range(5)]
                    words[position] = bad
                    with self.subTest(field=field, position=position, bad=bad):
                        request = replace(base, **{field: tuple(words)})
                        with self.assertRaises(p.ProtocolError) as raised:
                            p.serialize_message(request)
                        self.assertEqual(
                            raised.exception.issue.code, p.IssueCode.LIMIT_EXCEEDED
                        )
                        self.assertIn(
                            f"{message} must be an integer in [0, 4294967295]",
                            raised.exception.issue.message,
                        )
        with self.assertRaises(p.ProtocolError) as raised:
            p.serialize_message(replace(base, prompt_tokens=[1, 2, 3]))
        self.assertIn("must be a tuple of uint32", raised.exception.issue.message)

    def test_refresh_request_deadline_changes_only_the_absolute_deadline(self):
        request = example_request()
        wire = p.serialize_message(request)
        now = 1_900_000_000_000_000
        refreshed = p.refresh_request_deadline(wire, now)
        self.assertEqual(
            decode_client(refreshed),
            replace(
                request,
                absolute_deadline_unix_micros=now + request.remaining_deadline_micros,
            ),
        )
        offset = p.FRAME_HEADER_BYTES + OFFSET["absolute_deadline"]
        self.assertEqual(refreshed[:offset], wire[:offset])
        self.assertEqual(refreshed[offset + 8 :], wire[offset + 8 :])
        saturated = p.refresh_request_deadline(wire, 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(
            decode_client(saturated).absolute_deadline_unix_micros,
            0xFFFFFFFFFFFFFFFF,
        )
        status = p.serialize_message(p.StatusRequestFrame(808))
        self.assertEqual(p.refresh_request_deadline(status, now), status)

    def test_refresh_request_deadline_leaves_an_unparsable_frame_alone(self):
        truncated = mutate_u16(
            p.serialize_message(p.StatusRequestFrame(808)),
            6,
            int(p.FrameType.REQUEST),
        )
        now = 1_900_000_000_000_000
        self.assertEqual(p.refresh_request_deadline(truncated, now), truncated)
        self.assertEqual(p.refresh_request_deadline(truncated[:8], now), truncated[:8])
        request = p.serialize_message(example_request())
        for size in range(p.FRAME_HEADER_BYTES + p.REQUEST_FIXED_BYTES):
            with self.subTest(size=size):
                self.assertEqual(
                    p.refresh_request_deadline(request[:size], now), request[:size]
                )

    def test_refresh_request_deadline_preserves_invalid_sampling_bits(self):
        request = example_request()
        wire = bytearray(p.serialize_message(request))
        # Signaling NaNs would be quieted by float32 -> Python float -> float32.
        struct.pack_into(
            "<II",
            wire,
            p.FRAME_HEADER_BYTES + OFFSET["temperature"],
            0x7F800001,
            0xFF800001,
        )
        now = 1_900_000_000_000_000
        expected = bytearray(wire)
        struct.pack_into(
            "<Q",
            expected,
            p.FRAME_HEADER_BYTES + OFFSET["absolute_deadline"],
            now + request.remaining_deadline_micros,
        )
        self.assertEqual(p.refresh_request_deadline(bytes(wire), now), bytes(expected))

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
                (mutate_u16(valid, 6, 0x7777), p.IssueCode.UNKNOWN_FRAME_TYPE),
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

        invalid = replace(
            example_request(),
            sampling=p.SamplingParameters(0.8, 0.0, 32),
        )
        self.assert_protocol_error(
            p.FailureClass.REQUEST_ERROR,
            p.IssueCode.INVALID_SAMPLING,
            lambda: p.serialize_message(invalid),
        )
        self.assert_fatal_event(p.IssueCode.INVALID_COUNT, p.ReadyEvent(0, 4096, False))
        self.assert_fatal_event(
            p.IssueCode.INVALID_ENUM_VALUE, p.ReadyEvent(4, 4096, 2)
        )

    def test_unnamed_request_failures_are_fatal(self):
        """A client frame's failure is its request's error only when the
        frame names the request; one that names none fails the stream, as
        the engine classifies it (protocol_test.cpp)."""
        for message, code, failure_class in (
            (
                replace(example_request(), request_id=0),
                p.IssueCode.INVALID_REQUEST_ID,
                p.FailureClass.PROTOCOL_FATAL,
            ),
            (
                replace(example_request(), request_id=-1),
                p.IssueCode.INVALID_REQUEST_ID,
                p.FailureClass.PROTOCOL_FATAL,
            ),
            (
                p.CancelFrame(0),
                p.IssueCode.INVALID_REQUEST_ID,
                p.FailureClass.PROTOCOL_FATAL,
            ),
            (
                p.MaskResponseFrame(0, 7, bytes(4)),
                p.IssueCode.INVALID_REQUEST_ID,
                p.FailureClass.PROTOCOL_FATAL,
            ),
            (
                p.MaskResponseFrame(91, 0, bytes(4)),
                p.IssueCode.INVALID_REQUEST_ID,
                p.FailureClass.REQUEST_ERROR,
            ),
            (
                p.StatusRequestFrame(-1),
                p.IssueCode.INTEGER_OVERFLOW,
                p.FailureClass.PROTOCOL_FATAL,
            ),
        ):
            with self.subTest(message=message):
                issue = self.assert_protocol_error(
                    failure_class,
                    code,
                    lambda message=message: p.serialize_message(message),
                )
                self.assertEqual(
                    issue.request_id,
                    91 if failure_class is p.FailureClass.REQUEST_ERROR else 0,
                )

    def test_overflow_lengths_fail_before_allocation(self):
        valid = native_peer.serialize_event(engine_events()["tokens"])
        claimed = struct.unpack_from("<Q", valid, 8)[0]
        issue = parser_issue(mutate_u64(valid, 8, claimed + 1))
        self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
        self.assertEqual(issue.code, p.IssueCode.TRUNCATED_FRAME)

        enormous = mutate_u64(valid, 8, 0xFFFFFFFFFFFFFFFF)
        issue = parser_issue(enormous[: p.FRAME_HEADER_BYTES])
        self.assertEqual(issue.code, p.IssueCode.FRAME_TOO_LARGE)

        # A request past the frame limit is that request's error, found
        # before its frame is allocated.
        with mock.patch.object(p, "MAX_FRAME_PAYLOAD_BYTES", 1024):
            issue = self.assert_protocol_error(
                p.FailureClass.REQUEST_ERROR,
                p.IssueCode.FRAME_TOO_LARGE,
                lambda: p.serialize_message(
                    replace(example_request(), prompt_tokens=(7,) * 300)
                ),
            )
        self.assertEqual(issue.request_id, example_request().request_id)

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

        valid = native_peer.serialize_event(engine_events()["done_scored"])
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
