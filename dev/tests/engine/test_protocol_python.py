import os
import random
import shutil
import struct
import subprocess
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path

from server import protocol as p

ROOT = Path(__file__).parents[3]


REQUEST_GOLDEN = (
    "53504c480600180001000000540000000000000000000000efcdab8967452301"
    "000201008098281765060040a5ae0200000000008000000500000000000000cd"
    "cc4c3f3333733f200000001032547698badcfe00000000000000000001000000"
    "2a00000000000080ffffffff"
)
ERROR_GOLDEN = (
    "53504c4806001800050100002700000000000000000000000200000000000000"
    "0000090000000c0000006770755f6661756c744d6574616c206661696c6564"
)
STATUS_GOLDEN = (
    "53504c4806001800070100002d00000000000000000000002803000000000000"
    "050000007b22736368656d615f76657273696f6e223a342c227265616479223a"
    "747275657d"
)
INITIAL_MASK_GOLDEN = (
    "53504c4806001800030100001800000000000000000000005b00000000000000"
    "06000000000000000400000000000000"
)

CPP_GOLDEN_SOURCE = r"""
#include "engine/Protocol.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace splash::protocol;

void show(const Message &message) {
    auto result = serializeMessage(message);
    if (!result) {
        std::cerr << result.issue->describe() << "\n";
        std::exit(1);
    }
    for (uint8_t byte : *result.value) {
        std::cout << std::hex << std::setfill('0') << std::setw(2)
                  << unsigned(byte);
    }
    std::cout << "\n";
}

int main() {
    RequestFrame request;
    request.requestId = 0x0123456789abcdefULL;
    request.priority = RequestPriority::Foreground;
    request.absoluteDeadlineUnixMicros = 1800000000000000ULL;
    request.remainingDeadlineMicros = 45000000;
    request.logicalMaxOutputTokens = 32768;
    request.promptTokens = {0, 1, 42, 0x80000000U, 0xffffffffU};
    request.sampling = {0.8f, 0.95f, 32};
    request.seed = 0xfedcba9876543210ULL;
    request.cohort = Cohort::Constrained;
    request.constraint = ConstraintMode::TokenMask;
    show(request);
    RequestFrame image = request;
    image.promptTokens = {7, 3, 9};
    image.imageSpans = {{1, 1, 2, 2, 0x1111222233334444ULL, 0x5555666677778888ULL}};
    image.imagePixels.resize(image.imageSpans[0].pixelBytes());
    for (size_t i = 0; i < image.imagePixels.size(); ++i) {
        image.imagePixels[i] = static_cast<uint8_t>(i * 7 + 1);
    }
    show(image);
    RequestFrame score = request;
    score.promptTokens = {5, 6, 7};
    score.logicalMaxOutputTokens = 0;
    score.sampling = {0.0f, 1.0f, 0};
    score.cohort = Cohort::Greedy;
    score.constraint = ConstraintMode::None;
    score.scoreTokens = {101, 202, 303};
    show(score);
    show(CancelFrame{91});
    show(MaskResponseFrame{91, 7, {0xffffffffU, 0, 0xa5a5a5a5U}});
    show(StatusRequestFrame{808});
    show(ReadyEvent{1001, 4, 524288,
                    uint64_t(FeatureCancellation) |
                        uint64_t(FeatureTokenMasks) |
                        uint64_t(FeatureStatusJson) |
                        uint64_t(FeatureMultiplexing)});
    show(StartEvent{91, CacheDisposition::PrefixHit, 2, 4096, 131072});
    show(PromptProgressEvent{91, 2048, 123456});
    show(TokensEvent{91, 17, {10, 11, 12}});
    show(MaskRequestEvent{91, 6, 4, {}});
    show(MaskRequestEvent{91, 7, 4, {101, 102, 103}});
    show(DoneEvent{91, FinishReason::Stop, 4096, 512,
                   1000, 2000, 3500});
    DoneEvent scored{91, FinishReason::Stop, 4096, 0, 1000, 0, 3500};
    scored.optionLogits = {1.5f, -2.25f, 0.5f};
    show(scored);
    show(ErrorEvent{FailureClass::RequestError, 91, true,
                    "deadline_exceeded", "request deadline expired"});
    show(ErrorEvent{FailureClass::EngineUnhealthy, 0, false,
                    "gpu_fault", "Metal command buffer failed"});
    show(ErrorEvent{FailureClass::ProtocolFatal, 0, false,
                    "bad_frame", "stream framing cannot be trusted"});
    show(CapacityExhaustedEvent{92, 40, 12, 50000});
    show(StatusJsonEvent{
        808, kStatusSchemaVersion,
        "{\n  \"schema_version\": 4, \"ready\": true\n}"});
}
"""


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def example_request():
    return p.RequestFrame(
        request_id=0x0123456789ABCDEF,
        priority=p.RequestPriority.FOREGROUND,
        absolute_deadline_unix_micros=1_800_000_000_000_000,
        remaining_deadline_micros=45_000_000,
        logical_max_output_tokens=32_768,
        prompt_tokens=(0, 1, 42, 0x80000000, 0xFFFFFFFF),
        sampling=p.SamplingParameters(f32(0.8), f32(0.95), 32),
        seed=0xFEDCBA9876543210,
        cohort=p.Cohort.CONSTRAINED,
        constraint=p.ConstraintMode.TOKEN_MASK,
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
        cohort=p.Cohort.GREEDY,
        constraint=p.ConstraintMode.NONE,
        score_tokens=(101, 202, 303),
    )


def all_messages():
    return [
        example_request(),
        example_image_request(),
        example_score_request(),
        p.CancelFrame(91),
        p.MaskResponseFrame(91, 7, (0xFFFFFFFF, 0, 0xA5A5A5A5)),
        p.StatusRequestFrame(808),
        p.ReadyEvent(
            1001,
            4,
            524_288,
            int(
                p.ReadyFeature.CANCELLATION
                | p.ReadyFeature.TOKEN_MASKS
                | p.ReadyFeature.STATUS_JSON
                | p.ReadyFeature.MULTIPLEXING
            ),
        ),
        p.StartEvent(91, p.CacheDisposition.PREFIX_HIT, 2, 4096, 131_072),
        p.PromptProgressEvent(91, 2048, 123456),
        p.TokensEvent(91, 17, (10, 11, 12)),
        p.MaskRequestEvent(91, 6, 4, ()),
        p.MaskRequestEvent(91, 7, 4, (101, 102, 103)),
        p.DoneEvent(91, p.FinishReason.STOP, 4096, 512, 1000, 2000, 3500),
        p.DoneEvent(
            91,
            p.FinishReason.STOP,
            4096,
            0,
            1000,
            0,
            3500,
            (1.5, -2.25, 0.5),
        ),
        p.ErrorEvent(
            p.FailureClass.REQUEST_ERROR,
            91,
            True,
            b"deadline_exceeded",
            b"request deadline expired",
        ),
        p.ErrorEvent(
            p.FailureClass.ENGINE_UNHEALTHY,
            0,
            False,
            b"gpu_fault",
            b"Metal command buffer failed",
        ),
        p.ErrorEvent(
            p.FailureClass.PROTOCOL_FATAL,
            0,
            False,
            b"bad_frame",
            b"stream framing cannot be trusted",
        ),
        p.CapacityExhaustedEvent(92, 40, 12, 50_000),
        p.StatusJsonEvent(
            808,
            p.STATUS_SCHEMA_VERSION,
            b'{\n  "schema_version": 4, "ready": true\n}',
        ),
    ]


def parse_all(data, limits=p.ProtocolLimits()):
    parser = p.FrameParser(limits)
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


def parser_issue(data, limits=p.ProtocolLimits()):
    parser = p.FrameParser(limits)
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

    def test_fixed_golden_vectors(self):
        messages = (
            example_request(),
            p.ErrorEvent(
                p.FailureClass.ENGINE_UNHEALTHY,
                0,
                False,
                b"gpu_fault",
                b"Metal failed",
            ),
            p.StatusJsonEvent(
                808,
                p.STATUS_SCHEMA_VERSION,
                b'{"schema_version":4,"ready":true}',
            ),
            p.MaskRequestEvent(91, 6, 4, ()),
        )
        goldens = (
            REQUEST_GOLDEN,
            ERROR_GOLDEN,
            STATUS_GOLDEN,
            INITIAL_MASK_GOLDEN,
        )
        for message, golden in zip(messages, goldens, strict=True):
            wire = p.serialize_message(message)
            self.assertEqual(wire.hex(), golden)
            frame = parse_all(bytes.fromhex(golden))[0]
            self.assertEqual(p.decode_frame(frame), message)

    def test_cpp_codec_matches_fixed_golden_vectors(self):
        xcrun = shutil.which("xcrun")
        compiler = shutil.which(os.environ.get("CXX", "clang++"))
        if xcrun:
            command = [xcrun, "-sdk", "macosx", "clang++"]
        elif compiler:
            command = [compiler]
        else:
            self.skipTest("no C++ compiler is available for golden-vector check")

        with tempfile.TemporaryDirectory(prefix="splash-protocol-") as temp:
            binary = Path(temp) / "golden"
            command.extend(
                [
                    "-std=c++20",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(ROOT / "runtime/engine/Protocol.cpp"),
                    "-I",
                    str(ROOT / "runtime"),
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    str(binary),
                ]
            )
            subprocess.run(
                command,
                input=textwrap.dedent(CPP_GOLDEN_SOURCE),
                text=True,
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                [binary], text=True, check=True, capture_output=True
            )
        self.assertEqual(
            result.stdout.splitlines(),
            [p.serialize_message(message).hex() for message in all_messages()],
        )

    def test_progress_wire_validation(self):
        request = replace(example_request(), return_progress=True)
        self.assertEqual(
            [
                p.decode_frame(frame)
                for frame in parse_all(p.serialize_message(request))
            ],
            [request],
        )
        frame = p.encode_message(request)
        payload = bytearray(frame.payload)
        payload[59] = 2
        with self.assertRaises(p.ProtocolError) as raised:
            p.decode_frame(p.Frame(p.FrameType.REQUEST, bytes(payload)))
        self.assertEqual(
            raised.exception.issue.failure_class, p.FailureClass.REQUEST_ERROR
        )
        self.assertEqual(raised.exception.issue.code, p.IssueCode.INVALID_ENUM_VALUE)
        for event in (
            p.PromptProgressEvent(0, 1, 0),
            p.PromptProgressEvent(1, p.ProtocolLimits().max_prompt_tokens + 1, 0),
        ):
            with self.subTest(event=event), self.assertRaises(p.ProtocolError):
                p.serialize_message(event)

    def test_score_request_wire_layout_and_roundtrip(self):
        request = example_score_request()
        wire = p.serialize_message(request)
        self.assertEqual(
            struct.unpack_from("<I", wire, 24 + 60)[0], len(request.score_tokens)
        )
        self.assertEqual(
            struct.unpack_from("<3I", wire, 24 + 64 + 4 * 3),
            request.score_tokens,
        )
        self.assertEqual(
            p.decode_frame(parse_all(wire)[0]),
            request,
        )
        # A wrong score count desynchronizes the tail and must fail closed.
        bad = mutate_u32(wire, 24 + 60, 2)
        self.assert_protocol_error(
            p.FailureClass.REQUEST_ERROR,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: p.decode_frame(parse_all(bad)[0]),
        )

    def test_maximum_score_domain_roundtrips_without_truncation(self):
        request = replace(example_score_request(), score_tokens=tuple(range(255)))
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
        for message in (request, done):
            encoded = p.serialize_message(message)
            self.assertEqual(p.decode_frame(parse_all(encoded)[0]), message)

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
                p.IssueCode.INVALID_COHORT_CONSTRAINT,
            ),
            (
                replace(base, cohort=p.Cohort.SAMPLING),
                p.IssueCode.INVALID_COHORT_CONSTRAINT,
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

    def test_malformed_score_frames_preserve_request_error_codes(self):
        request = example_score_request()
        payload = p.encode_message(request).payload
        score_offset = 64 + 4 * len(request.prompt_tokens)
        for tokens, output_tokens in (
            ((101,), 0),
            ((101, 101), 0),
            (tuple(range(256)), 0),
            (request.score_tokens, 1),
        ):
            with self.subTest(
                tokens=tokens[:4], count=len(tokens), output=output_tokens
            ):
                malformed = bytearray(payload[:score_offset])
                struct.pack_into("<I", malformed, 27, output_tokens)
                struct.pack_into("<I", malformed, 60, len(tokens))
                malformed.extend(struct.pack(f"<{len(tokens)}I", *tokens))
                wire = p.serialize_frame(p.Frame(p.FrameType.REQUEST, bytes(malformed)))
                issue = self.assert_protocol_error(
                    p.FailureClass.REQUEST_ERROR,
                    p.IssueCode.INVALID_COUNT,
                    lambda: p.decode_frame(parse_all(wire)[0]),
                )
                self.assertEqual(issue.request_id, request.request_id)

    def test_oversized_score_count_rejected_before_tail_decode(self):
        request = example_score_request()
        wire = p.serialize_message(request)
        for count in (p.MAX_SCORE_TOKENS + 1, 1_000_000, 0xFFFFFFFF):
            with self.subTest(count=count):
                bad = mutate_u32(wire, 24 + 60, count)
                issue = self.assert_protocol_error(
                    p.FailureClass.REQUEST_ERROR,
                    p.IssueCode.INVALID_COUNT,
                    lambda: p.decode_frame(parse_all(bad)[0]),
                )
                self.assertEqual(issue.request_id, request.request_id)

    def test_done_option_logits_wire_layout_and_roundtrip(self):
        done = p.DoneEvent(
            91, p.FinishReason.STOP, 4096, 0, 1000, 0, 3500, (1.5, -2.25, 0.5)
        )
        wire = p.serialize_message(done)
        self.assertEqual(struct.unpack_from("<I", wire, 24 + 41)[0], 3)
        self.assertEqual(struct.unpack_from("<3f", wire, 24 + 45), done.option_logits)
        self.assertEqual(p.decode_frame(parse_all(wire)[0]), done)
        # A wrong logit count must fail closed, not truncate.
        bad = mutate_u32(wire, 24 + 41, 2)
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: p.decode_frame(parse_all(bad)[0]),
        )
        # A v5-shaped 41-byte Done payload is below the v6 minimum.
        short = wire[: 24 + 41]
        short = mutate_u64(short, 12, 41)
        issue = parser_issue(short)
        self.assertEqual(issue.code, p.IssueCode.INVALID_PAYLOAD_LENGTH)
        # Each scored-done invariant fails closed on the wire: non-stop
        # finish, completion tokens, decode time, and sub-minimum or
        # non-finite logits are all INVALID_COUNT.
        non_stop = bytearray(wire)
        non_stop[24 + 8] = int(p.FinishReason.LENGTH)
        mutations = (
            bytes(non_stop),
            mutate_u32(wire, 24 + 13, 1),
            mutate_u64(wire, 24 + 25, 1),
        )
        # A wire-consistent count below the score minimum.
        one_logit = wire[: 24 + 49]
        one_logit = mutate_u64(one_logit, 12, 49)
        one_logit = mutate_u32(one_logit, 24 + 41, 1)
        mutations += (one_logit, mutate_u32(wire, 24 + 45, 0x7FC00000))
        for mutated in mutations:
            with self.subTest(mutated=mutated.hex()):
                self.assert_protocol_error(
                    p.FailureClass.PROTOCOL_FATAL,
                    p.IssueCode.INVALID_COUNT,
                    lambda mutated=mutated: p.decode_frame(parse_all(mutated)[0]),
                )

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
        for logits in (
            (1.0,),
            tuple(float(i) for i in range(256)),
            (1.0, float("nan")),
            (1.0, float("inf")),
            (1.0, 1e300),
        ):
            with self.subTest(logits=logits[:4]):
                self.assert_protocol_error(
                    p.FailureClass.ENGINE_UNHEALTHY,
                    p.IssueCode.INVALID_COUNT,
                    lambda logits=logits: p.serialize_message(
                        p.DoneEvent(**base, option_logits=logits)
                    ),
                )
        # Each scored-done invariant is independently invalid: a non-stop
        # finish reason, completion tokens, or decode time.
        for field, value in (
            ("reason", p.FinishReason.LENGTH),
            ("reason", p.FinishReason.CANCELLED),
            ("completion_tokens", 1),
            ("decode_micros", 1),
        ):
            with self.subTest(field=field, value=value):
                self.assert_protocol_error(
                    p.FailureClass.ENGINE_UNHEALTHY,
                    p.IssueCode.INVALID_COUNT,
                    lambda field=field, value=value: p.serialize_message(
                        p.DoneEvent(
                            **{**base, field: value},
                            option_logits=(1.0, 2.0),
                        )
                    ),
                )
        # Valid scored, generation, and cancelled done events still encode
        # and decode unchanged; a raw integer finish reason is normalized to
        # the enum and round-trips like the enum form.
        for event in (
            p.DoneEvent(**base, option_logits=(1.0, 2.0)),
            p.DoneEvent(**{**base, "reason": 0}, option_logits=(1.0, 2.0)),
            p.DoneEvent(91, p.FinishReason.STOP, 4096, 512, 1000, 2000, 3500),
            p.DoneEvent(91, p.FinishReason.LENGTH, 4096, 512, 1000, 2000, 3500),
            p.DoneEvent(91, p.FinishReason.CANCELLED, 4096, 128, 1000, 500, 1500),
        ):
            with self.subTest(event=event):
                self.assertEqual(
                    p.decode_frame(parse_all(p.serialize_message(event))[0]),
                    event,
                )

    def test_request_header_and_binary_prompt(self):
        request = example_request()
        wire = p.serialize_message(request)
        self.assertEqual(wire[:4], b"SPLH")
        self.assertEqual(
            struct.unpack_from("<HHHHQI", wire, 4),
            (p.PROTOCOL_VERSION, 24, int(p.FrameType.REQUEST), 0, 84, 0),
        )
        self.assertEqual(struct.unpack_from("<Q", wire, 24)[0], request.request_id)
        self.assertEqual(struct.unpack_from("<I", wire, 24 + 31)[0], 5)
        self.assertEqual(struct.unpack_from("<I", wire, 24 + 35)[0], 0)
        self.assertEqual(
            struct.unpack_from("<5I", wire, 24 + 64), request.prompt_tokens
        )

        image = example_image_request()
        wire = p.serialize_message(image)
        span_offset = 24 + 64 + 4 * len(image.prompt_tokens)
        self.assertEqual(struct.unpack_from("<I", wire, 24 + 35)[0], 1)
        self.assertEqual(
            struct.unpack_from("<IIIIQQ", wire, span_offset),
            (1, 1, 2, 2, 0x1111222233334444, 0x5555666677778888),
        )
        self.assertEqual(wire[span_offset + 32 :], image.image_pixels)
        self.assertEqual(p.decode_frame(parse_all(wire)[0]), image)

    def test_refresh_request_deadline_changes_only_the_absolute_deadline(self):
        request = example_request()
        wire = p.serialize_message(request)
        now = 1_900_000_000_000_000
        refreshed = p.refresh_request_deadline(wire, now)
        self.assertEqual(
            p.decode_frame(parse_all(refreshed)[0]),
            replace(
                request,
                absolute_deadline_unix_micros=now + request.remaining_deadline_micros,
            ),
        )
        offset = p.FRAME_HEADER_BYTES + 11
        self.assertEqual(refreshed[:offset], wire[:offset])
        self.assertEqual(refreshed[offset + 8 :], wire[offset + 8 :])
        saturated = p.refresh_request_deadline(wire, 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(
            p.decode_frame(parse_all(saturated)[0]).absolute_deadline_unix_micros,
            0xFFFFFFFFFFFFFFFF,
        )
        status = p.serialize_message(p.StatusRequestFrame(808))
        self.assertEqual(p.refresh_request_deadline(status, now), status)

    def test_refresh_request_deadline_leaves_an_unparsable_frame_alone(self):
        truncated = mutate_u16(
            p.serialize_message(p.StatusRequestFrame(808)),
            8,
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
        struct.pack_into("<II", wire, p.FRAME_HEADER_BYTES + 39, 0x7F800001, 0xFF800001)
        now = 1_900_000_000_000_000
        expected = bytearray(wire)
        struct.pack_into(
            "<Q",
            expected,
            p.FRAME_HEADER_BYTES + 11,
            now + request.remaining_deadline_micros,
        )
        self.assertEqual(p.refresh_request_deadline(bytes(wire), now), bytes(expected))

    def test_every_message_round_trips_in_one_multiplexed_stream(self):
        expected = all_messages()
        stream = b"".join(p.serialize_message(message) for message in expected)
        frames = parse_all(stream)
        self.assertEqual(len(frames), len(expected))
        self.assertEqual(
            [p.decode_frame(frame) for frame in frames],
            expected,
        )

    def test_one_byte_incremental_parser(self):
        request = example_request()
        parser = p.FrameParser()
        frame = None
        for byte in p.serialize_message(request):
            step = parser.consume(bytes((byte,)))
            self.assertEqual(step.consumed_bytes, 1)
            self.assertIsNone(step.issue)
            if step.frame:
                self.assertIsNone(frame)
                frame = step.frame
        self.assertIsNotNone(frame)
        self.assertIsNone(parser.finish())
        self.assertEqual(p.decode_frame(frame), request)

    def test_header_failures_are_protocol_fatal(self):
        valid = p.serialize_message(example_request())
        mutations = []
        bad_magic = bytearray(valid)
        bad_magic[0] = ord("X")
        mutations.append((bytes(bad_magic), p.IssueCode.BAD_MAGIC))
        mutations.extend(
            [
                (mutate_u16(valid, 4, 1), p.IssueCode.UNSUPPORTED_VERSION),
                (mutate_u16(valid, 6, 23), p.IssueCode.INVALID_HEADER_SIZE),
                (mutate_u16(valid, 8, 0x7777), p.IssueCode.UNKNOWN_FRAME_TYPE),
                (mutate_u16(valid, 10, 1), p.IssueCode.NON_ZERO_HEADER_FLAGS),
                (mutate_u32(valid, 20, 1), p.IssueCode.NON_ZERO_RESERVED_FIELD),
                (
                    mutate_u64(valid, 12, 0xFFFFFFFFFFFFFFFF),
                    p.IssueCode.FRAME_TOO_LARGE,
                ),
                (mutate_u64(valid, 12, 54), p.IssueCode.INVALID_PAYLOAD_LENGTH),
                (b"ready\n".ljust(24, b"r"), p.IssueCode.BAD_MAGIC),
            ]
        )
        for wire, code in mutations:
            issue = parser_issue(wire)
            self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
            self.assertEqual(issue.code, code)

        parser = p.FrameParser()
        first = parser.consume(mutations[0][0])
        self.assertEqual(first.issue.code, p.IssueCode.BAD_MAGIC)
        second = parser.consume(valid)
        self.assertEqual(second.consumed_bytes, 0)
        self.assertEqual(second.issue.code, p.IssueCode.PARSER_ALREADY_FAILED)

    def test_every_nonempty_truncation_is_fatal(self):
        wire = p.serialize_message(example_request())
        for cut in range(1, len(wire)):
            issue = parser_issue(wire[:cut])
            self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
            self.assertEqual(issue.code, p.IssueCode.TRUNCATED_FRAME)
        self.assertIsNone(p.FrameParser().finish())

    def test_bad_request_is_recoverable_and_next_frame_remains_aligned(self):
        valid = p.serialize_message(example_request())
        invalid = bytearray(valid)
        invalid[p.FRAME_HEADER_BYTES + 8] = 0xFF
        frames = parse_all(bytes(invalid) + valid)
        self.assertEqual(len(frames), 2)
        self.assert_protocol_error(
            p.FailureClass.REQUEST_ERROR,
            p.IssueCode.INVALID_ENUM_VALUE,
            lambda: p.decode_frame(frames[0]),
        )
        self.assertEqual(p.decode_frame(frames[1]), example_request())

    def test_malformed_payloads_have_strict_classification(self):
        wire = p.serialize_message(example_request())
        cases = (
            (
                mutate_u32(wire, p.FRAME_HEADER_BYTES + 31, 0xFFFFFFFF),
                p.FailureClass.REQUEST_ERROR,
                p.IssueCode.LIMIT_EXCEEDED,
            ),
            (
                mutate_u64(wire, p.FRAME_HEADER_BYTES + 11, 0),
                p.FailureClass.REQUEST_ERROR,
                p.IssueCode.INVALID_DEADLINE,
            ),
            (
                mutate_u32(wire, p.FRAME_HEADER_BYTES + 39, 0x7FC00000),
                p.FailureClass.REQUEST_ERROR,
                p.IssueCode.INVALID_SAMPLING,
            ),
        )
        constraint = bytearray(wire)
        constraint[p.FRAME_HEADER_BYTES + 10] = int(p.ConstraintMode.NONE)
        cases += (
            (
                bytes(constraint),
                p.FailureClass.REQUEST_ERROR,
                p.IssueCode.INVALID_COHORT_CONSTRAINT,
            ),
        )
        for mutated, failure, code in cases:
            frame = parse_all(mutated)[0]
            self.assert_protocol_error(
                failure, code, lambda frame=frame: p.decode_frame(frame)
            )

        tokens = p.serialize_message(p.TokensEvent(7, 0, (1, 2)))
        bad_tokens = mutate_u32(tokens, p.FRAME_HEADER_BYTES + 12, 3)
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: p.decode_frame(parse_all(bad_tokens)[0]),
        )

        error = p.serialize_message(
            p.ErrorEvent(p.FailureClass.REQUEST_ERROR, 7, False, b"bad", b"message")
        )
        bad_error = mutate_u32(error, p.FRAME_HEADER_BYTES + 10, 0xFFFFFFFF)
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_PAYLOAD_LENGTH,
            lambda: p.decode_frame(parse_all(bad_error)[0]),
        )

        status = p.serialize_message(
            p.StatusJsonEvent(0, p.STATUS_SCHEMA_VERSION, b'{"x":1}')
        )
        bad_status = mutate_u32(status, p.FRAME_HEADER_BYTES + 8, 2)
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.INVALID_STATUS_SCHEMA,
            lambda: p.decode_frame(parse_all(bad_status)[0]),
        )

    def test_status_json_is_opaque_length_delimited_and_bounded(self):
        limits = p.ProtocolLimits(
            max_frame_payload_bytes=128 * 1024,
            max_status_json_bytes=128 * 1024 - 12,
            max_error_string_bytes=1024,
            max_prompt_tokens=1024,
            max_logical_output_tokens=1024,
            max_token_batch=128,
            max_simulation_tokens=8,
            max_mask_words=4096,
        )
        event = p.StatusJsonEvent(
            99, p.STATUS_SCHEMA_VERSION, exact_json(limits.max_status_json_bytes)
        )
        frame = parse_all(p.serialize_message(event, limits), limits)[0]
        self.assertEqual(p.decode_frame(frame, limits), event)

        too_large = p.StatusJsonEvent(99, p.STATUS_SCHEMA_VERSION, event.json + b" ")
        self.assert_protocol_error(
            p.FailureClass.ENGINE_UNHEALTHY,
            p.IssueCode.LIMIT_EXCEEDED,
            lambda: p.serialize_message(too_large, limits),
        )

        normal = p.serialize_message(
            p.StatusJsonEvent(1, p.STATUS_SCHEMA_VERSION, exact_json(4096))
        )
        tiny = p.ProtocolLimits(
            max_frame_payload_bytes=128 * 1024,
            max_status_json_bytes=100,
            max_error_string_bytes=1024,
            max_prompt_tokens=1024,
            max_logical_output_tokens=1024,
            max_token_batch=128,
            max_simulation_tokens=8,
            max_mask_words=4096,
        )
        issue = parser_issue(normal, tiny)
        self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
        self.assertEqual(issue.code, p.IssueCode.FRAME_TOO_LARGE)

    def test_failure_taxonomy_and_capacity_event(self):
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
            frame = parse_all(p.serialize_message(expected))[0]
            self.assertEqual(p.decode_frame(frame), expected)

        capacity = p.CapacityExhaustedEvent(8, 24, 3, 100_000)
        frame = parse_all(p.serialize_message(capacity))[0]
        self.assertEqual(p.decode_frame(frame), capacity)

        invalid = replace(
            example_request(),
            sampling=p.SamplingParameters(0.8, 0.0, 32),
        )
        self.assert_protocol_error(
            p.FailureClass.REQUEST_ERROR,
            p.IssueCode.INVALID_SAMPLING,
            lambda: p.serialize_message(invalid),
        )
        self.assert_protocol_error(
            p.FailureClass.ENGINE_UNHEALTHY,
            p.IssueCode.INVALID_COUNT,
            lambda: p.serialize_message(p.ReadyEvent(0, 4, 4096, 0)),
        )

    def test_overflow_lengths_and_limits_fail_before_allocation(self):
        invalid_limits = p.ProtocolLimits(max_frame_payload_bytes=0xFFFFFFFFFFFFFFFF)
        parser = p.FrameParser(invalid_limits)
        self.assertTrue(parser.failed)
        step = parser.consume(b"")
        self.assertEqual(step.issue.code, p.IssueCode.PARSER_ALREADY_FAILED)
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.LIMIT_EXCEEDED,
            lambda: p.serialize_message(example_request(), invalid_limits),
        )
        # A frame limit below the fixed request size must fail on its own:
        # zero the status/error limits and use an 8-byte CancelFrame so no
        # other limit or message size can be the real cause.
        small_limits = p.ProtocolLimits(
            max_frame_payload_bytes=p.REQUEST_FIXED_BYTES - 1,
            max_status_json_bytes=0,
            max_error_string_bytes=0,
        )
        self.assertTrue(p.FrameParser(small_limits).failed)
        self.assert_protocol_error(
            p.FailureClass.PROTOCOL_FATAL,
            p.IssueCode.LIMIT_EXCEEDED,
            lambda: p.serialize_message(p.CancelFrame(91), small_limits),
        )
        # The floor itself accepts a parser and round-trips the same frame.
        floor_limits = p.ProtocolLimits(
            max_frame_payload_bytes=p.REQUEST_FIXED_BYTES,
            max_status_json_bytes=0,
            max_error_string_bytes=0,
        )
        cancel = p.CancelFrame(91)
        encoded = p.serialize_message(cancel, floor_limits)
        self.assertEqual(
            p.decode_frame(parse_all(encoded, floor_limits)[0], floor_limits),
            cancel,
        )

        valid = p.serialize_message(example_request())
        claimed = struct.unpack_from("<Q", valid, 12)[0]
        issue = parser_issue(mutate_u64(valid, 12, claimed + 1))
        self.assertEqual(issue.failure_class, p.FailureClass.PROTOCOL_FATAL)
        self.assertEqual(issue.code, p.IssueCode.TRUNCATED_FRAME)

        enormous = mutate_u64(valid, 12, 0xFFFFFFFFFFFFFFFF)
        issue = parser_issue(enormous[:24])
        self.assertEqual(issue.code, p.IssueCode.FRAME_TOO_LARGE)

    def test_random_malformed_inputs_and_valid_frame_mutations_do_not_crash(self):
        random_source = random.Random(0x5EED1234)
        for _ in range(3000):
            data = random_source.randbytes(random_source.randrange(257))
            parser = p.FrameParser()
            offset = 0
            steps = 0
            while offset < len(data) and not parser.failed:
                size = min(random_source.randrange(1, 32), len(data) - offset)
                step = parser.consume(memoryview(data)[offset : offset + size])
                self.assertLessEqual(step.consumed_bytes, size)
                self.assertTrue(step.consumed_bytes or step.issue)
                offset += step.consumed_bytes
                if step.frame:
                    try:
                        p.decode_frame(step.frame)
                    except p.ProtocolError:
                        pass
                steps += 1
                self.assertLess(steps, 1024)
            parser.finish()

        valid = p.serialize_message(example_request())
        for _ in range(2000):
            mutated = bytearray(valid)
            for _ in range(random_source.randrange(1, 5)):
                mutated[random_source.randrange(len(mutated))] = (
                    random_source.randrange(256)
                )
            parser = p.FrameParser()
            offset = 0
            while offset < len(mutated) and not parser.failed:
                step = parser.consume(memoryview(mutated)[offset:])
                self.assertTrue(step.consumed_bytes or step.issue)
                offset += step.consumed_bytes
                if step.frame:
                    try:
                        p.decode_frame(step.frame)
                    except p.ProtocolError:
                        pass
            parser.finish()


if __name__ == "__main__":
    unittest.main()
