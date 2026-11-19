"""The audio server's HTTP endpoints over the fake audio engine: speech
requests and their wire frames, complete and streamed audio answers,
transcription uploads, the control endpoints, errors, cancellation and the
serving process's startup."""

import http.client
import io
import json
import math
import os
import signal
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dev.tests.audio_fixtures import (
    ASR_LANGUAGES,
    ASR_PROFILE,
    CUSTOM_VOICE_PROFILE,
    MODEL,
    TTS_PROFILE,
    VOICE_DESIGN_PROFILE,
    AudioHarnessTestCase,
    AudioPlan,
    FakeAudioRuntime,
    FakeAudioTokenizer,
    audio_main_args,
    multipart,
    pcm,
    sine,
    speech_body,
    transcription_upload,
    wav_upload,
)
from server import audio_frontend, audio_io, audio_server, diagnostics, serve_options
from server import audio_protocol as audio_wire
from server import runtime as engine_runtime

FRAME = audio_wire.SAMPLES_PER_FRAME


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    return condition()


# The bytes a speech answer holds back until it ends (its faded end).
TAIL = 2 * round(audio_io.SpeechEdges.FADE_OUT_SECONDS * audio_io.TTS_SAMPLE_RATE)


def edged(*chunks):
    """The answer the server sends for the engine's chunks: its end faded
    (audio_io.SpeechEdges)."""
    edges = audio_io.SpeechEdges(False)
    return b"".join(edges.push(chunk) for chunk in chunks) + edges.finish()


class SpeechRequestTests(AudioHarnessTestCase):
    def test_request_maps_to_the_wire_frame(self):
        runtime = FakeAudioRuntime()
        harness = self.harness(runtime)
        status, _, _ = harness.request(
            "POST",
            "/v1/audio/speech",
            speech_body(
                input="Hello, world!",
                language="en",
                temperature=0.5,
                top_k=10,
                top_p=0.8,
                repetition_penalty=1.2,
                seed=7,
                max_frames=50,
                voice="alloy",
            ),
        )
        self.assertEqual(status, 200)
        (request,) = runtime.requests
        frame = request.frame
        self.assertIsInstance(frame, audio_wire.SpeechRequest)
        self.assertEqual(frame.request_id, 0)
        self.assertEqual(
            frame.text_tokens, tuple(harness.tokenizer.encode_text("Hello, world!"))
        )
        self.assertEqual(
            harness.tokenizer.calls[-1],
            ("Hello, world!", {"add_special_tokens": False}),
        )
        self.assertEqual(frame.language_id, 2050)
        self.assertEqual(
            (frame.temperature, frame.top_k, frame.top_p, frame.repetition_penalty),
            (0.5, 10, 0.8, 1.2),
        )
        # The code predictor follows the talker's sampling.
        self.assertEqual(
            (frame.predictor_temperature, frame.predictor_top_k, frame.predictor_top_p),
            (0.5, 10, 0.8),
        )
        self.assertEqual((frame.seed, frame.max_frames, frame.chunk_frames), (7, 50, 0))
        self.assertEqual(frame.reference_text_tokens, ())
        self.assertEqual(frame.speaker_embedding, ())
        # A Base model's prompt has no instruction and no preset speaker.
        self.assertEqual((frame.instruction_tokens, frame.preset_speaker), ((), 0))
        self.assertEqual(request.deadline, math.inf if False else request.deadline)

    def test_defaults_are_the_models_generation_config(self):
        runtime = FakeAudioRuntime()
        harness = self.harness(runtime)
        status, _, _ = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 200)
        frame = runtime.requests[0].frame
        self.assertEqual(frame.language_id, 0)
        self.assertAlmostEqual(frame.temperature, 0.9)
        self.assertEqual(frame.top_k, 50)
        self.assertEqual(frame.top_p, 1.0)
        self.assertAlmostEqual(frame.repetition_penalty, 1.05)
        self.assertAlmostEqual(frame.predictor_temperature, 0.9)
        self.assertEqual(frame.predictor_top_k, 50)
        self.assertEqual(frame.max_frames, 0)
        self.assertTrue(0 < frame.seed < 2**64)
        # Greedy throughout when the request asks for it.
        harness.request(
            "POST", "/v1/audio/speech", speech_body(temperature=0, language="Chinese")
        )
        frame = runtime.requests[1].frame
        self.assertEqual((frame.temperature, frame.predictor_temperature), (0.0, 0.0))
        self.assertEqual(frame.language_id, 2055)

    def test_a_complete_wav_answer(self):
        first, second = pcm([100] * FRAME), pcm([-100] * (FRAME // 2))
        harness = self.harness(FakeAudioRuntime(AudioPlan(chunks=[first, second])))
        status, headers, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body()
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        self.assertEqual(headers["Content-Length"], str(len(payload)))
        self.assertEqual(payload[:4], b"RIFF")
        self.assertEqual(struct.unpack_from("<I", payload, 40)[0], len(first + second))
        self.assertEqual(payload[44:], edged(first, second))
        self.assertEqual(headers["X-Splash-Frames"], "2")
        self.assertEqual(
            headers["X-Splash-Audio-Seconds"], f"{1.5 * FRAME / 24000:.3f}"
        )
        self.assertEqual(headers["X-Splash-Finish-Reason"], "stop")
        self.assertGreater(float(headers["X-Splash-Time-To-First-Audio-Ms"]), 0)

    def test_a_pcm_answer(self):
        chunk = pcm([7] * FRAME)
        harness = self.harness(FakeAudioRuntime(AudioPlan(chunks=[chunk])))
        status, headers, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(response_format="pcm")
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/pcm")
        self.assertEqual(payload, edged(chunk))
        self.assertEqual(headers["X-Splash-Frames"], "1")

    def test_a_streamed_wav_arrives_chunk_by_chunk(self):
        first, second = pcm([1] * FRAME), pcm([2] * FRAME)
        plan = AudioPlan(chunks=[first, second], stepped=True)
        harness = self.harness(FakeAudioRuntime(plan))
        connection, response = harness.open_stream(
            "/v1/audio/speech", speech_body(stream=True)
        )
        self.addCleanup(connection.close)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "audio/wav")
        self.assertEqual(response.getheader("Transfer-Encoding"), "chunked")
        self.assertIsNone(response.getheader("Content-Length"))
        self.assertIn("X-Splash-Time-To-First-Audio-Ms", dict(response.getheaders()))
        header = response.read(audio_io.WAV_HEADER_BYTES)
        self.assertEqual(header[:4], b"RIFF")
        self.assertEqual(struct.unpack_from("<I", header, 4)[0], 0xFFFFFFFF)
        self.assertEqual(struct.unpack_from("<I", header, 40)[0], 0xFFFFFFFF)
        # The first chunk is here before the engine decodes the second, all
        # but the end held back until it is known not to be the answer's.
        self.assertEqual(response.read(len(first) - TAIL), first[:-TAIL])
        plan.step.set()
        self.assertEqual(response.read(), edged(first, second)[len(first) - TAIL :])

    def test_each_chunk_is_written_before_the_engine_is_done(self):
        # The engine emits one chunk, then holds Done back: the client has
        # the header and the chunk while the request is still running.
        for response_format in ("wav", "pcm"):
            with self.subTest(response_format=response_format):
                chunk = pcm([5] * FRAME)
                plan = AudioPlan(chunks=[chunk], hold_done=True)
                runtime = FakeAudioRuntime(plan)
                harness = self.harness(runtime)
                connection, response = harness.open_stream(
                    "/v1/audio/speech",
                    speech_body(stream=True, response_format=response_format),
                )
                self.addCleanup(connection.close)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("Transfer-Encoding"), "chunked")
                if response_format == "wav":
                    header = response.read(audio_io.WAV_HEADER_BYTES)
                    self.assertEqual(header, audio_io.wav_header(None))
                self.assertEqual(response.read(len(chunk) - TAIL), chunk[:-TAIL])
                self.assertTrue(plan.terminal.is_set())
                self.assertFalse(runtime.calls[0].done)
                self.assertFalse(response.isclosed())
                plan.terminal_release.set()
                self.assertEqual(response.read(), edged(chunk)[len(chunk) - TAIL :])
                self.assertTrue(wait_until(lambda: runtime.calls[0].done))

    def test_chunk_frames_reach_the_engine(self):
        runtime = FakeAudioRuntime()
        harness = self.harness(runtime)
        status, _, _ = harness.request(
            "POST", "/v1/audio/speech", speech_body(stream=True, chunk_frames=5)
        )
        self.assertEqual(status, 200)
        self.assertEqual(runtime.requests[0].frame.chunk_frames, 5)
        status, _, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(chunk_frames=-1)
        )
        self.assertEqual(status, 400)
        self.assertIn("chunk_frames", json.loads(payload)["error"]["message"])

    def test_a_streamed_pcm_answer_ends_cleanly(self):
        chunks = [pcm([3] * 10), pcm([4] * 20)]
        harness = self.harness(FakeAudioRuntime(AudioPlan(chunks=chunks)))
        status, headers, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(stream=True, response_format="pcm")
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/pcm")
        self.assertEqual(payload, edged(*chunks))

    def test_an_error_before_the_first_chunk_has_its_status(self):
        failure = engine_runtime.RequestFailed(
            1, b"invalid_request", b"text is not speakable"
        )
        harness = self.harness(FakeAudioRuntime(AudioPlan(exception=failure)))
        status, headers, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(stream=True)
        )
        self.assertEqual(status, 400)
        self.assertEqual(headers["Content-Type"], "application/json")
        error = json.loads(payload)["error"]
        self.assertEqual(error["code"], "invalid_request")
        self.assertEqual(error["message"], "text is not speakable")

    def test_an_engine_failure_after_the_first_chunk_cuts_the_stream_short(self):
        plan = AudioPlan(
            chunks=[pcm([1] * FRAME), pcm([2] * FRAME)],
            stepped=True,
            exception=None,
        )
        runtime = FakeAudioRuntime(plan)
        harness = self.harness(runtime)
        connection, response = harness.open_stream(
            "/v1/audio/speech", speech_body(stream=True)
        )
        self.addCleanup(connection.close)
        self.assertEqual(response.status, 200)
        response.read(audio_io.WAV_HEADER_BYTES + 2 * FRAME - TAIL)
        # The engine dies: its call ends with a failure, and the stream ends
        # without its final chunk.
        runtime.calls[0].complete(
            error=engine_runtime.EngineUnhealthy("native protocol reached EOF")
        )
        plan.cancelled.set()
        with self.assertRaises((http.client.IncompleteRead, ConnectionError, OSError)):
            response.read()

    def test_invalid_requests_are_refused(self):
        runtime = FakeAudioRuntime()
        harness = self.harness(runtime)
        cases = (
            (speech_body(input=""), 400, "input"),
            (speech_body(input=3), 400, "input"),
            ({"model": MODEL}, 400, "input"),
            (speech_body(model="other"), 404, "model other not found"),
            (speech_body(response_format="mp3"), 400, "response_format"),
            (speech_body(stream="yes"), 400, "stream"),
            (speech_body(language="klingon"), 400, "not supported"),
            (speech_body(language=3), 400, "language"),
            (speech_body(temperature=-1), 400, "temperature"),
            (speech_body(temperature="hot"), 400, "temperature"),
            (speech_body(top_k=1.5), 400, "top_k"),
            (speech_body(top_p=0), 400, "top_p"),
            (speech_body(top_p=2), 400, "top_p"),
            (speech_body(repetition_penalty=0), 400, "repetition_penalty"),
            (speech_body(seed=-1), 400, "seed"),
            (speech_body(seed=2**64), 400, "seed"),
            (speech_body(max_frames=-1), 400, "max_frames"),
            (speech_body(max_frames=2**32), 400, "max_frames"),
            (speech_body(voice=5), 400, "voice"),
            (speech_body(input=" ".join(["word"] * 4097)), 400, "limit is 4096"),
        )
        for body, expected, message in cases:
            with self.subTest(body=body):
                status, _, payload = harness.request("POST", "/v1/audio/speech", body)
                self.assertEqual(status, expected, payload)
                self.assertIn(message, json.loads(payload)["error"]["message"])
        self.assertEqual(runtime.requests, [])
        status, _, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(input=" ".join(["word"] * 4097))
        )
        self.assertEqual(
            json.loads(payload)["error"]["code"], "context_length_exceeded"
        )

    def test_a_transcription_model_does_not_speak(self):
        harness = self.asr_harness()
        status, _, payload = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 404)
        self.assertIn("does not speak", json.loads(payload)["error"]["message"])

    def test_engine_outcomes_map_to_statuses(self):
        cases = (
            (
                engine_runtime.RequestFailed(1, b"limit_exceeded", b"too many frames"),
                400,
                "limit_exceeded",
            ),
            (
                engine_runtime.RequestFailed(1, b"capacity_exhausted", b"no lane"),
                400,
                "capacity_exhausted",
            ),
            (
                engine_runtime.RequestFailed(1, b"gpu_fault", b"x", retryable=True),
                503,
                "gpu_fault",
            ),
            (engine_runtime.EngineUnhealthy("died"), 503, "runtime_unavailable"),
            (engine_runtime.ProtocolFatal("bad frame"), 503, "runtime_unavailable"),
        )
        for failure, status_code, code in cases:
            with self.subTest(code=code):
                harness = self.harness(FakeAudioRuntime(AudioPlan(exception=failure)))
                status, _, payload = harness.request(
                    "POST", "/v1/audio/speech", speech_body()
                )
                self.assertEqual(status, status_code)
                self.assertEqual(json.loads(payload)["error"]["code"], code)

    def test_a_recovering_engine_refuses_with_503(self):
        runtime = FakeAudioRuntime()
        harness = self.harness(runtime)
        runtime.ready = False
        status, headers, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body()
        )
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(payload)["error"]["code"], "engine_recovering")
        self.assertEqual(headers["Retry-After"], "1")
        status, _, _ = harness.request("GET", "/ready")
        self.assertEqual(status, 503)

    def test_cancelled_and_length_outcomes(self):
        harness = self.harness(
            FakeAudioRuntime(AudioPlan(chunks=[pcm([1] * FRAME)], reason="length"))
        )
        status, headers, _ = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Splash-Finish-Reason"], "length")

    def test_a_client_that_disconnects_cancels_its_request(self):
        plan = AudioPlan(block=True)
        runtime = FakeAudioRuntime(plan)
        harness = self.harness(runtime)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=5
        )
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(speech_body(stream=True)),
            {"Content-Type": "application/json"},
        )
        self.assertTrue(plan.started.wait(2))
        connection.close()
        self.assertTrue(wait_until(lambda: runtime.cancel_count == 1))
        self.assertTrue(plan.cancelled.is_set())
        self.assertTrue(
            wait_until(lambda: harness.server.requests.stats()["active"] == 0)
        )

    def test_a_request_timeout_answers_504(self):
        harness = self.harness(FakeAudioRuntime(AudioPlan(block=True)), timeout=0.2)
        status, _, payload = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 504)
        self.assertEqual(json.loads(payload)["error"]["code"], "request_timeout")

    def test_request_logger_sees_each_outcome(self):
        records = []
        harness = self.harness(
            FakeAudioRuntime(AudioPlan(chunks=[pcm([1] * FRAME)])),
            request_logger=records.append,
        )
        harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertTrue(wait_until(lambda: len(records) == 1))
        self.assertEqual(records[0]["outcome"], "stop")
        self.assertEqual(records[0]["kind"], "speech")
        self.assertEqual(records[0]["streamed"], FRAME)
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            audio_server.print_audio_request(records[0])
            self.assertTrue(diagnostics.CONSOLE.flush(5))
        self.assertIn("Done · text", stdout.getvalue())


def frame_codes(frames, start=0):
    """Codes of `frames` frames, each value its own."""
    return tuple(
        (start + index) % 2048 for index in range(frames * audio_wire.CODES_PER_FRAME)
    )


class SpeechReferenceTests(AudioHarnessTestCase):
    """reference_speech_id: a request continuing earlier speech."""

    def speak(self, harness, **fields):
        status, headers, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(**fields)
        )
        self.assertEqual(status, 200, payload)
        return headers["X-Splash-Speech-Id"]

    def test_a_reference_carries_the_earlier_texts_tokens_and_trimmed_codes(self):
        codes = frame_codes(10)
        runtime = FakeAudioRuntime(
            AudioPlan(chunks=[pcm([1] * FRAME)], frames=10, codes=codes)
        )
        harness = self.harness(runtime)
        first = self.speak(harness, input="The first paragraph.")
        self.assertRegex(first, r"^speech-[0-9a-f]{24}$")
        second = self.speak(harness, input="The second one.", reference_speech_id=first)
        self.assertNotEqual(first, second)
        earlier, frame = (request.frame for request in runtime.requests)
        # Every request asks for its codes, which later requests may take.
        self.assertEqual(earlier.flags, audio_wire.SPEECH_FLAG_CODES)
        self.assertEqual(
            (earlier.reference_text_tokens, earlier.reference_codes), ((), ())
        )
        self.assertEqual(frame.flags, audio_wire.SPEECH_FLAG_CODES)
        self.assertEqual(
            frame.text_tokens, tuple(harness.tokenizer.encode_text("The second one."))
        )
        self.assertEqual(frame.reference_text_tokens, earlier.text_tokens)
        # The end of the reference's closing pause, its last frames, is dropped.
        frames = 10 - audio_frontend.REFERENCE_TRIM_FRAMES
        self.assertEqual(
            frame.reference_codes, codes[: frames * audio_wire.CODES_PER_FRAME]
        )
        # A continuation is itself a reference: its own text and frames.
        self.speak(harness, reference_speech_id=second)
        third = runtime.requests[2].frame
        self.assertEqual(third.reference_text_tokens, frame.text_tokens)
        # The default plan's codes: one frame, of which one is kept.
        self.assertEqual(third.reference_codes, frame_codes(1))

    def test_a_streamed_answer_names_its_speech_id_and_is_a_reference(self):
        runtime = FakeAudioRuntime(
            AudioPlan(chunks=[pcm([1] * FRAME)], frames=8, codes=frame_codes(8, 5))
        )
        harness = self.harness(runtime)
        connection, response = harness.open_stream(
            "/v1/audio/speech", speech_body(stream=True)
        )
        speech_id = response.getheader("X-Splash-Speech-Id")
        response.read()
        connection.close()
        self.assertRegex(speech_id, r"^speech-[0-9a-f]{24}$")
        self.speak(harness, reference_speech_id=speech_id)
        self.assertEqual(
            runtime.requests[1].frame.reference_codes,
            frame_codes(8 - audio_frontend.REFERENCE_TRIM_FRAMES, 5),
        )

    def test_a_request_naming_running_speech_waits_for_it(self):
        plan = AudioPlan(chunks=[pcm([1] * FRAME)], frames=8, stepped=True)
        plan.chunks.append(pcm([2] * FRAME))
        runtime = FakeAudioRuntime(plan)
        harness = self.harness(runtime)
        connection, response = harness.open_stream(
            "/v1/audio/speech", speech_body(stream=True)
        )
        speech_id = response.getheader("X-Splash-Speech-Id")
        results = []
        thread = threading.Thread(
            target=lambda: results.append(
                harness.request(
                    "POST",
                    "/v1/audio/speech",
                    speech_body(reference_speech_id=speech_id),
                )
            )
        )
        thread.start()
        # Not submitted while the speech it names is still running.
        time.sleep(0.1)
        self.assertEqual(len(runtime.requests), 1)
        plan.step.set()
        response.read()
        connection.close()
        thread.join(3)
        ((status, _, payload),) = results
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            len(runtime.requests[1].frame.reference_codes),
            (8 - audio_frontend.REFERENCE_TRIM_FRAMES) * 16,
        )

    def test_unknown_failed_and_malformed_references_are_refused(self):
        runtime = FakeAudioRuntime(
            AudioPlan(
                exception=engine_runtime.RequestFailed(
                    1,
                    b"invalid_request",
                    b"text token outside the vocabulary",
                    retryable=False,
                )
            )
        )
        harness = self.harness(runtime)
        status, headers, _ = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 400)
        failed = headers.get("X-Splash-Speech-Id")
        self.assertIsNone(failed)
        for value, expected, code in (
            ("speech-000000000000000000000000", 404, "reference_not_found"),
            ("", 400, "invalid_request_error"),
            (7, 400, "invalid_request_error"),
        ):
            with self.subTest(value=value):
                status, _, payload = harness.request(
                    "POST", "/v1/audio/speech", speech_body(reference_speech_id=value)
                )
                self.assertEqual(status, expected)
                error = json.loads(payload)["error"]
                self.assertEqual(error["code"], code)
                self.assertIn("reference_speech_id", error["message"])
        # Speech whose request failed or was cancelled is no reference.
        references = harness.app.references
        references.begin("speech-failed")
        references.finish("speech-failed", (1, 2), None)
        status, _, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(reference_speech_id="speech-failed")
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(payload)["error"]["code"], "reference_not_found")
        self.assertEqual(len(runtime.requests), 1)

    def test_a_running_reference_that_fails_is_refused(self):
        references = audio_frontend.SpeechReferences()
        references.begin("speech-a")
        timer = threading.Timer(0.05, references.finish, ("speech-a", (1,), None))
        timer.start()
        with self.assertRaises(audio_frontend.APIError) as refused:
            references.get("speech-a", time.monotonic() + 2)
        self.assertEqual(refused.exception.status, 404)
        self.assertIn("did not finish", refused.exception.message)
        # One still running when the request's time runs out: 504.
        references.begin("speech-b")
        with self.assertRaises(audio_frontend.APIError) as refused:
            references.get("speech-b", time.monotonic() + 0.05)
        self.assertEqual(refused.exception.status, 504)

    def test_kept_speech_is_bounded(self):
        references = audio_frontend.SpeechReferences(budget_bytes=4 * 40, capacity=8)
        for index in range(3):
            references.begin(f"speech-{index}")
            references.finish(f"speech-{index}", (1, 2, 3, 4), frame_codes(1))
        # Each keeps 20 values of 4 bytes: the budget holds two.
        with self.assertRaises(audio_frontend.APIError):
            references.get("speech-0", time.monotonic())
        text, codes = references.get("speech-2", time.monotonic())
        self.assertEqual((tuple(text), tuple(codes)), ((1, 2, 3, 4), frame_codes(1)))


class VoiceControlTests(AudioHarnessTestCase):
    """`instructions` and `voice` as each Qwen3-TTS model type takes them."""

    def instruction_turn(self, harness, text):
        return tuple(
            harness.tokenizer.encode_text(f"<|im_start|>user\n{text}<|im_end|>\n")
        )

    def test_a_base_model_refuses_instructions(self):
        runtime = FakeAudioRuntime()
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(instructions="Speak softly.")
        )
        self.assertEqual(status, 400)
        self.assertIn("takes no instructions", json.loads(payload)["error"]["message"])
        # None or an empty string asks for nothing.
        for instructions in (None, ""):
            status, _, _ = harness.request(
                "POST", "/v1/audio/speech", speech_body(instructions=instructions)
            )
            self.assertEqual(status, 200)
        self.assertEqual(
            [r.frame.instruction_tokens for r in runtime.requests], [(), ()]
        )

    def test_a_voice_design_model_speaks_in_the_voice_described(self):
        runtime = FakeAudioRuntime()
        harness = self.harness(runtime, profile=VOICE_DESIGN_PROFILE)
        for body in (
            speech_body(),
            speech_body(instructions="  "),
            speech_body(instructions=3),
        ):
            with self.subTest(body=body):
                status, _, payload = harness.request("POST", "/v1/audio/speech", body)
                self.assertEqual(status, 400)
                self.assertIn("instructions", json.loads(payload)["error"]["message"])
        self.assertEqual(runtime.requests, [])
        voice = "A calm male voice with a low pitch."
        status, _, _ = harness.request(
            "POST",
            "/v1/audio/speech",
            speech_body(instructions=voice, voice="alloy", language="en"),
        )
        self.assertEqual(status, 200)
        (request,) = runtime.requests
        # The whole user turn, tokenized at once; the voice is not read.
        self.assertEqual(
            request.frame.instruction_tokens, self.instruction_turn(harness, voice)
        )
        self.assertEqual(
            harness.tokenizer.calls[0],
            (f"<|im_start|>user\n{voice}<|im_end|>\n", {"add_special_tokens": False}),
        )
        self.assertEqual(request.frame.preset_speaker, 0)
        self.assertEqual(request.frame.language_id, 2050)
        status, _, payload = harness.request(
            "POST",
            "/v1/audio/speech",
            speech_body(instructions=" ".join(["word"] * 300)),
        )
        self.assertEqual(status, 400)
        self.assertEqual(
            json.loads(payload)["error"]["code"], "context_length_exceeded"
        )

    def test_a_custom_voice_model_speaks_as_a_preset_speaker(self):
        runtime = FakeAudioRuntime()
        harness = self.harness(runtime, profile=CUSTOM_VOICE_PROFILE)
        for voice in (None, "alloy", "default"):
            with self.subTest(voice=voice):
                status, _, payload = harness.request(
                    "POST", "/v1/audio/speech", speech_body(voice=voice)
                )
                self.assertEqual(status, 400)
                self.assertIn(
                    "speakers: dylan, vivian", json.loads(payload)["error"]["message"]
                )
        self.assertEqual(runtime.requests, [])
        status, _, _ = harness.request(
            "POST", "/v1/audio/speech", speech_body(voice="Vivian", language="en")
        )
        self.assertEqual(status, 200)
        frame = runtime.requests[0].frame
        self.assertEqual(
            (frame.preset_speaker, frame.instruction_tokens, frame.language_id),
            (3065, (), 2050),
        )
        style = "Very happy and excited."
        harness.request(
            "POST",
            "/v1/audio/speech",
            speech_body(voice="vivian", instructions=style),
        )
        frame = runtime.requests[1].frame
        self.assertEqual(
            frame.instruction_tokens, self.instruction_turn(harness, style)
        )
        self.assertEqual(frame.language_id, 0)
        # A dialect speaker speaks his dialect unless another language is asked.
        for language, expected in ((None, 2074), ("zh", 2074), ("English", 2050)):
            harness.request(
                "POST",
                "/v1/audio/speech",
                speech_body(voice="Dylan", language=language),
            )
            self.assertEqual(runtime.requests[-1].frame.language_id, expected)
            self.assertEqual(runtime.requests[-1].frame.preset_speaker, 2878)

    def test_models_and_voices_name_each_model_types_controls(self):
        for profile, controls, voices in (
            (
                TTS_PROFILE,
                {
                    "tts_model_type": "base",
                    "instructions": "unsupported",
                    "references": True,
                },
                [{"id": "default"}],
            ),
            (
                VOICE_DESIGN_PROFILE,
                {
                    "tts_model_type": "voice_design",
                    "instructions": "required",
                    "references": True,
                },
                [{"id": "default"}],
            ),
            (
                CUSTOM_VOICE_PROFILE,
                {
                    "tts_model_type": "custom_voice",
                    "instructions": "optional",
                    "references": True,
                    "voices": ["dylan", "vivian"],
                },
                [{"id": "dylan"}, {"id": "vivian"}],
            ),
        ):
            with self.subTest(model_type=profile.tts_model_type):
                harness = self.harness(profile=profile)
                _, _, payload = harness.request("GET", "/v1/models")
                (model,) = json.loads(payload)["data"]
                self.assertEqual({key: model.get(key) for key in controls}, controls)
                self.assertEqual("voices" in model, "voices" in controls)
                _, _, payload = harness.request("GET", "/v1/audio/voices")
                self.assertEqual(json.loads(payload), voices)
        _, _, payload = self.asr_harness().request("GET", "/v1/models")
        self.assertNotIn("tts_model_type", json.loads(payload)["data"][0])


class TranscriptionTests(AudioHarnessTestCase):
    def transcribe(self, harness, body, headers):
        return harness.request(
            "POST", "/v1/audio/transcriptions", data=body, headers=headers
        )

    def test_upload_becomes_prompt_tokens_and_samples(self):
        tokenizer = FakeAudioTokenizer()
        output = tokenizer.encode_text("language English<asr_text>hello world")
        runtime = FakeAudioRuntime(
            AudioPlan(tokens=[output[:3], output[3:]]), kind=audio_wire.ModelKind.ASR
        )
        harness = self.harness(runtime, tokenizer=tokenizer, profile=ASR_PROFILE)
        samples = sine(0.5)
        status, headers, payload = self.transcribe(
            harness, *transcription_upload(samples)
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(json.loads(payload), {"text": "hello world"})
        self.assertEqual(headers["X-Splash-Audio-Seconds"], "0.500")
        self.assertEqual(headers["X-Splash-Segments"], "1")
        self.assertEqual(headers["X-Splash-Output-Tokens"], str(len(output)))
        (request,) = runtime.requests
        frame = request.frame
        self.assertIsInstance(frame, audio_wire.TranscribeRequest)
        self.assertEqual(
            frame.prefix_tokens,
            tuple(tokenizer.encode_text(audio_frontend.ASR_PREFIX)),
        )
        self.assertEqual(
            frame.suffix_tokens,
            tuple(tokenizer.encode_text(audio_frontend.ASR_SUFFIX)),
        )
        self.assertEqual(frame.max_tokens, 0)
        decoded = np.frombuffer(frame.samples, dtype="<f4")
        self.assertEqual(len(decoded), 8000)
        self.assertLess(np.abs(decoded - samples).max(), 1e-3)

    def test_a_forced_language_extends_the_assistant_prefix(self):
        tokenizer = FakeAudioTokenizer()
        runtime = FakeAudioRuntime(
            AudioPlan(tokens=[tokenizer.encode_text("bonjour")]),
            kind=audio_wire.ModelKind.ASR,
        )
        harness = self.harness(runtime, tokenizer=tokenizer, profile=ASR_PROFILE)
        for language in ("fr", "French", "FRENCH"):
            with self.subTest(language=language):
                runtime.plans.append(
                    AudioPlan(tokens=[tokenizer.encode_text("bonjour")])
                )
                status, _, payload = self.transcribe(
                    harness,
                    *transcription_upload(
                        fields={"language": language, "response_format": "verbose_json"}
                    ),
                )
                self.assertEqual(status, 200, payload)
                answer = json.loads(payload)
                self.assertEqual(answer["language"], "French")
                self.assertEqual(answer["text"], "bonjour")
                frame = runtime.requests[-1].frame
                self.assertEqual(
                    frame.suffix_tokens,
                    tuple(
                        tokenizer.encode_text(
                            audio_frontend.ASR_SUFFIX + "language French<asr_text>"
                        )
                    ),
                )

    def test_response_formats(self):
        tokenizer = FakeAudioTokenizer()
        transcript = tokenizer.encode_text("language English<asr_text>hello world")
        plans = [AudioPlan(tokens=[transcript]) for _ in range(3)]
        harness = self.harness(
            FakeAudioRuntime(*plans, kind=audio_wire.ModelKind.ASR),
            tokenizer=tokenizer,
            profile=ASR_PROFILE,
        )
        status, headers, payload = self.transcribe(
            harness, *transcription_upload(fields={"response_format": "text"})
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/plain; charset=utf-8")
        self.assertEqual(payload, b"hello world")
        status, _, payload = self.transcribe(
            harness,
            *transcription_upload(
                sine(1.0),
                fields={"response_format": "verbose_json", "temperature": "0"},
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload),
            {
                "task": "transcribe",
                "language": "English",
                "duration": 1.0,
                "text": "hello world",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"}
                ],
            },
        )
        status, _, payload = self.transcribe(
            harness, *transcription_upload(fields={"response_format": "json"})
        )
        self.assertEqual(json.loads(payload), {"text": "hello world"})

    def test_a_transcript_without_the_language_prefix_is_taken_as_it_is(self):
        tokenizer = FakeAudioTokenizer()
        harness = self.harness(
            FakeAudioRuntime(
                AudioPlan(tokens=[tokenizer.encode_text("  just words <|im_end|>")]),
                kind=audio_wire.ModelKind.ASR,
            ),
            tokenizer=tokenizer,
            profile=ASR_PROFILE,
        )
        status, _, payload = self.transcribe(
            harness, *transcription_upload(fields={"response_format": "verbose_json"})
        )
        self.assertEqual(status, 200)
        answer = json.loads(payload)
        self.assertEqual(answer["text"], "just words")
        self.assertIsNone(answer["language"])

    def test_long_audio_is_split_and_transcribed_in_order(self):
        tokenizer = FakeAudioTokenizer()
        first = tokenizer.encode_text("language English<asr_text>first part")
        second = tokenizer.encode_text("language English<asr_text>second part")
        runtime = FakeAudioRuntime(
            AudioPlan(tokens=[first]),
            AudioPlan(tokens=[second]),
            kind=audio_wire.ModelKind.ASR,
        )
        harness = self.harness(runtime, tokenizer=tokenizer, profile=ASR_PROFILE)
        samples = np.concatenate([sine(0.9), np.zeros(1600), sine(0.9)])
        with (
            mock.patch.object(audio_io, "MAX_REQUEST_SECONDS", 1.0),
            mock.patch.object(audio_io, "SPLIT_SEARCH_SECONDS", 0.3),
        ):
            status, headers, payload = self.transcribe(
                harness,
                *transcription_upload(
                    samples, fields={"response_format": "verbose_json"}
                ),
            )
        self.assertEqual(status, 200, payload)
        self.assertEqual(headers["X-Splash-Segments"], "2")
        answer = json.loads(payload)
        self.assertEqual(answer["text"], "first part second part")
        self.assertEqual(len(answer["segments"]), 2)
        self.assertEqual(answer["segments"][0]["start"], 0.0)
        self.assertEqual(answer["segments"][1]["end"], answer["duration"])
        # The cut falls in the silence, and the requests carry the two spans.
        cut = answer["segments"][0]["end"]
        self.assertTrue(0.9 <= cut <= 1.0, cut)
        first_frame, second_frame = (request.frame for request in runtime.requests)
        self.assertEqual(
            first_frame.sample_count + second_frame.sample_count, len(samples)
        )
        self.assertEqual(first_frame.sample_count, round(cut * 16_000))

    def test_higher_rate_and_stereo_uploads_are_resampled(self):
        runtime = FakeAudioRuntime(kind=audio_wire.ModelKind.ASR)
        harness = self.harness(runtime, profile=ASR_PROFILE)
        data = wav_upload(sine(0.5, 48_000), 48_000, channels=2)
        body, content_type = multipart(
            {"model": MODEL}, {"file": ("stereo.wav", data, "audio/wav")}
        )
        status, headers, _ = self.transcribe(
            harness, body, {"Content-Type": content_type}
        )
        self.assertEqual(status, 200)
        self.assertEqual(runtime.requests[0].frame.sample_count, 8000)
        self.assertEqual(headers["X-Splash-Audio-Seconds"], "0.500")

    def test_invalid_uploads_are_refused(self):
        runtime = FakeAudioRuntime(kind=audio_wire.ModelKind.ASR)
        harness = self.harness(runtime, profile=ASR_PROFILE)
        wav = wav_upload(sine(0.1), 16_000)
        cases = (
            (multipart({"model": MODEL}), 400, "file field is required"),
            (
                multipart(
                    {"model": MODEL, "stream": "true"},
                    {"file": ("a.wav", wav, "audio/wav")},
                ),
                400,
                "streaming transcriptions are not supported",
            ),
            (
                multipart({"model": "other"}, {"file": ("a.wav", wav, "audio/wav")}),
                404,
                "model other not found",
            ),
            (
                multipart(
                    {"model": MODEL, "response_format": "srt"},
                    {"file": ("a.wav", wav, "audio/wav")},
                ),
                400,
                "response_format",
            ),
            (
                multipart(
                    {"model": MODEL, "language": "xx"},
                    {"file": ("a.wav", wav, "audio/wav")},
                ),
                400,
                "not supported",
            ),
            (
                multipart(
                    {"model": MODEL, "temperature": "warm"},
                    {"file": ("a.wav", wav, "audio/wav")},
                ),
                400,
                "temperature",
            ),
            (
                multipart({"model": MODEL}, {"file": ("a.wav", b"", "audio/wav")}),
                400,
                "empty",
            ),
        )
        for (body, content_type), expected, message in cases:
            with self.subTest(message=message):
                status, _, payload = self.transcribe(
                    harness, body, {"Content-Type": content_type}
                )
                self.assertEqual(status, expected, payload)
                self.assertIn(message, json.loads(payload)["error"]["message"])
        # A field given twice is refused rather than one taken.
        body, content_type = multipart(
            {"model": MODEL}, {"file": ("a.wav", wav, "audio/wav")}
        )
        boundary = content_type.partition("boundary=")[2].encode()
        repeated = body.replace(
            b"--" + boundary + b"--",
            b"--"
            + boundary
            + b'\r\nContent-Disposition: form-data; name="model"\r\n\r\nx\r\n--'
            + boundary
            + b"--",
        )
        status, _, payload = self.transcribe(
            harness, repeated, {"Content-Type": content_type}
        )
        self.assertEqual(status, 400)
        self.assertIn("repeated", json.loads(payload)["error"]["message"])
        body, content_type = multipart(
            {"model": MODEL}, {"file": ("a.mp3", b"ID3\x03\x00junk", "audio/mpeg")}
        )
        status, _, payload = self.transcribe(
            harness, body, {"Content-Type": content_type}
        )
        self.assertEqual(status, 415)
        self.assertEqual(json.loads(payload)["error"]["code"], "unsupported_media_type")

    def test_multipart_framing_is_strict(self):
        harness = self.asr_harness()
        body, content_type = transcription_upload()
        cases = (
            (body, {"Content-Type": "application/json"}, 415),
            (body, {"Content-Type": "multipart/form-data"}, 400),
            (b"no boundary here", {"Content-Type": content_type["Content-Type"]}, 400),
            (body[:-20], content_type, 400),
            (body.replace(b"\r\n\r\n", b"\n\n", 1), content_type, 400),
            (body.replace(b'name="file"', b""), content_type, 400),
        )
        for data, headers, expected in cases:
            with self.subTest(data=data[:40]):
                status, _, _ = self.transcribe(harness, data, headers)
                self.assertEqual(status, expected)
        status, _, payload = harness.request(
            "POST",
            "/v1/audio/transcriptions",
            data=b"{}",
            headers={
                "Content-Type": "multipart/form-data; boundary=x",
                "Transfer-Encoding": "chunked",
            },
        )
        self.assertEqual(status, 400)

    def test_uploads_are_bounded_by_the_request_size(self):
        harness = self.asr_harness(max_request_bytes=4096)
        body, headers = transcription_upload(sine(1.0))
        status, _, payload = self.transcribe(harness, body, headers)
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(payload)["error"]["code"], "request_too_large")
        body, headers = transcription_upload(sine(0.05))
        status, _, _ = self.transcribe(harness, body, headers)
        self.assertEqual(status, 200)

    def test_decoded_samples_are_charged_to_the_body_budget(self):
        harness = self.asr_harness()
        with mock.patch.object(
            harness.server.request_bodies, "capacity", 2 * 16_000 * 2 + 8192
        ):
            # 2 s of 16-bit samples upload (64 KiB) and decode to 128 KiB of
            # float32, more than the budget holds.
            status, _, payload = self.transcribe(
                harness, *transcription_upload(sine(2.0))
            )
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(payload)["error"]["code"], "frontend_overloaded")

    def test_a_speech_model_does_not_transcribe(self):
        harness = self.harness(FakeAudioRuntime())
        status, _, payload = self.transcribe(harness, *transcription_upload())
        self.assertEqual(status, 404)
        self.assertIn("does not transcribe", json.loads(payload)["error"]["message"])

    def test_an_engine_failure_mid_way_is_reported(self):
        failure = engine_runtime.RequestFailed(1, b"invalid_count", b"audio too long")
        harness = self.asr_harness(AudioPlan(exception=failure))
        status, _, payload = self.transcribe(harness, *transcription_upload())
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["code"], "invalid_count")


class ControlEndpointTests(AudioHarnessTestCase):
    def test_health_ready_status_and_metrics(self):
        harness = self.harness()
        status, _, payload = harness.request("GET", "/health")
        self.assertEqual((status, json.loads(payload)), (200, {"status": "ok"}))
        status, _, payload = harness.request("GET", "/ready")
        self.assertEqual((status, json.loads(payload)), (200, {"status": "ready"}))
        status, _, payload = harness.request("GET", "/status")
        self.assertEqual(status, 200)
        document = json.loads(payload)
        self.assertTrue(document["ready"])
        self.assertEqual(document["frontend"]["family"], "qwen3-tts")
        self.assertEqual(document["frontend"]["model"], MODEL)
        self.assertIn("English", document["frontend"]["languages"])
        self.assertEqual(document["instance"]["model"], MODEL)
        self.assertIn("http_request", document["latency"])
        self.assertEqual(document["http"]["requests"]["capacity"], 4)
        status, headers, payload = harness.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/plain"))
        self.assertIn("splash_ready 1", payload.decode())
        self.assertIn("splash_http_request_seconds_count", payload.decode())

    def test_models_and_voices(self):
        harness = self.harness(served_model_names=["tts"], announce_served_name=True)
        status, _, payload = harness.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        listing = json.loads(payload)
        self.assertEqual([item["id"] for item in listing["data"]], ["tts", MODEL])
        self.assertEqual(listing["data"][0]["family"], "qwen3-tts")
        self.assertEqual(listing["data"][0]["task"], "text-to-speech")
        self.assertEqual(listing["data"][1]["root"], "tts")
        status, _, payload = harness.request("GET", "/v1/models/tts")
        self.assertEqual((status, json.loads(payload)["id"]), (200, "tts"))
        status, _, _ = harness.request("GET", "/v1/models/other")
        self.assertEqual(status, 404)
        status, _, payload = harness.request("GET", "/v1/audio/voices")
        self.assertEqual((status, json.loads(payload)), (200, [{"id": "default"}]))
        asr = self.asr_harness()
        status, _, payload = asr.request("GET", "/v1/models")
        self.assertEqual(json.loads(payload)["data"][0]["task"], "transcription")
        self.assertEqual(
            json.loads(payload)["data"][0]["languages"], list(ASR_LANGUAGES)
        )

    def test_unknown_routes_and_methods(self):
        harness = self.harness()
        for method, path in (
            ("GET", "/"),
            ("GET", "/v1/chat/completions"),
            ("POST", "/v1/chat/completions"),
            ("DELETE", "/v1/audio/speech"),
            ("GET", "/v1/responses/resp_1"),
        ):
            with self.subTest(method=method, path=path):
                status, _, payload = harness.request(
                    method, path, {} if method == "POST" else None
                )
                self.assertEqual(status, 404, payload)
        status, headers, _ = harness.request("OPTIONS", "/v1/audio/speech")
        self.assertEqual(status, 204)
        self.assertEqual(headers["Allow"], "GET, HEAD, POST, OPTIONS")

    def test_api_key_guards_every_route_but_health_and_ready(self):
        harness = self.harness(api_key="secret")
        for path in ("/health", "/ready"):
            status, _, _ = harness.request("GET", path)
            self.assertEqual(status, 200)
        for method, path, body in (
            ("GET", "/status", None),
            ("GET", "/v1/models", None),
            ("POST", "/v1/audio/speech", speech_body()),
        ):
            with self.subTest(path=path):
                status, headers, _ = harness.request(method, path, body)
                self.assertEqual(status, 401)
                self.assertEqual(headers["WWW-Authenticate"], "Bearer")
                status, _, _ = harness.request(
                    method,
                    path,
                    body,
                    {
                        "Authorization": "Bearer secret",
                        "Content-Type": "application/json",
                    },
                )
                self.assertEqual(status, 200)

    def test_requests_beyond_the_queue_are_refused(self):
        plan = AudioPlan(block=True)
        runtime = FakeAudioRuntime(plan)
        harness = self.harness(runtime, queue_size=1)
        # The first request's answer is still owed when the second arrives.
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=5
        )
        self.addCleanup(connection.close)
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(speech_body()),
            {"Content-Type": "application/json"},
        )
        self.assertTrue(plan.started.wait(2))
        status, _, payload = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(payload)["error"]["code"], "frontend_overloaded")
        plan.release.set()


class ModelProfileTests(unittest.TestCase):
    def test_profiles_load_from_config_and_generation_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_tts",
                        "talker_config": {
                            "codec_language_id": {"English": 2050, "chinese": 2055}
                        },
                    }
                )
            )
            profile = audio_frontend.ModelProfile.load(root)
            self.assertEqual(profile.family, "qwen3-tts")
            self.assertIs(profile.kind, audio_wire.ModelKind.TTS)
            self.assertEqual(profile.languages, {"english": 2050, "chinese": 2055})
            self.assertEqual(profile.generation, audio_frontend.TTS_DEFAULTS)
            (root / "generation_config.json").write_text(
                json.dumps(
                    {
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_k": 20,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "subtalker_dosample": False,
                        "subtalker_temperature": 0.8,
                        "subtalker_top_k": 10,
                        "subtalker_top_p": 0.95,
                    }
                )
            )
            profile = audio_frontend.ModelProfile.load(root)
            self.assertEqual(
                profile.generation,
                {
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "predictor_temperature": 0.0,
                    "predictor_top_k": 10,
                    "predictor_top_p": 0.95,
                },
            )
            self.assertEqual(profile.language_id("EN"), 2050)
            self.assertEqual(profile.language_id(None), 0)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_asr",
                        "support_languages": ["Chinese", "English"],
                    }
                )
            )
            profile = audio_frontend.ModelProfile.load(root)
            self.assertEqual(profile.family, "qwen3-asr")
            self.assertIs(profile.kind, audio_wire.ModelKind.ASR)
            self.assertEqual(profile.languages, ("Chinese", "English"))
            for config in (
                {"model_type": "qwen3"},
                {"model_type": "qwen3_tts"},
                {"model_type": "qwen3_asr", "support_languages": "English"},
            ):
                (root / "config.json").write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    audio_frontend.ModelProfile.load(root)

    def test_speech_profiles_take_the_model_type_and_its_speakers(self):
        talker = {
            "codec_language_id": {
                "english": 2050,
                "chinese": 2055,
                "beijing_dialect": 2074,
            },
            "spk_id": {"Vivian": 3065, "dylan": 2878},
            "spk_is_dialect": {"Vivian": False, "dylan": "beijing_dialect"},
        }

        def load(**config):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "config.json").write_text(
                    json.dumps({"model_type": "qwen3_tts", **config})
                )
                return audio_frontend.ModelProfile.load(root)

        base = load(talker_config=talker)
        self.assertEqual(base.tts_model_type, "base")
        self.assertEqual((base.speakers, base.dialects), ({}, {}))
        # A dialect is not a language a request names.
        self.assertEqual(base.languages, {"english": 2050, "chinese": 2055})
        self.assertEqual(
            load(talker_config=talker, tts_model_type="voice_design").tts_model_type,
            "voice_design",
        )
        custom = load(talker_config=talker, tts_model_type="custom_voice")
        self.assertEqual(custom.speakers, {"vivian": 3065, "dylan": 2878})
        self.assertEqual(custom.dialects, {"dylan": 2074})
        self.assertEqual(custom.language_id(None, "dylan"), 2074)
        self.assertEqual(custom.language_id("Chinese", "dylan"), 2074)
        self.assertEqual(custom.language_id("en", "dylan"), 2050)
        self.assertEqual(custom.language_id(None, "vivian"), 0)
        for config in (
            {"talker_config": talker, "tts_model_type": "voice_clone"},
            {
                "talker_config": {**talker, "spk_id": {}},
                "tts_model_type": "custom_voice",
            },
            {
                "talker_config": {**talker, "spk_is_dialect": {"dylan": "wu_dialect"}},
                "tts_model_type": "custom_voice",
            },
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    load(**config)


class AudioMainTests(unittest.TestCase):
    def _run_main(self, args, readiness, profile=TTS_PROFILE, serve=None, status=None):
        status = mock.Mock() if status is None else status
        runtime = mock.Mock()
        runtime.readiness = readiness
        runtime.wait_ready.return_value = True
        server = mock.Mock(server_port=8000)
        if serve is not None:
            server.serve_forever.side_effect = serve
        with (
            mock.patch.object(audio_server, "parse_args", return_value=args),
            mock.patch.object(
                audio_server.ModelProfile, "load", return_value=profile
            ) as load,
            mock.patch.object(
                audio_server.AutoTokenizer, "from_pretrained", return_value=object()
            ) as tokenizer,
            mock.patch.object(
                audio_server.audio_runtime, "AudioRuntime", return_value=runtime
            ) as runtime_type,
            mock.patch.object(audio_server, "AudioBackend") as backend_type,
            mock.patch.object(audio_server, "AudioFrontend") as app_type,
            mock.patch.object(audio_server, "AudioServer", return_value=server),
            mock.patch.object(audio_server.signal, "signal"),
            mock.patch.object(audio_server, "print_status", status),
        ):
            audio_server.main()
        return mock.Mock(
            runtime=runtime,
            runtime_type=runtime_type,
            backend_type=backend_type,
            app_type=app_type,
            server=server,
            status=status,
            load=load,
            tokenizer=tokenizer,
        )

    def test_main_wires_the_engine_command_and_the_frontend(self):
        args = audio_main_args(
            max_memory=2**34,
            lanes=4,
            idle_release=600.0,
            request_timeout=30.0,
        )
        run = self._run_main(
            args, audio_wire.ReadyEvent(4, 8192, audio_wire.ModelKind.TTS)
        )
        run.runtime_type.assert_called_once_with(
            [
                "splash",
                "serve-omni",
                "model",
                "--max-memory",
                str(2**34),
                "--lanes",
                "4",
                "--idle-release",
                "600.0",
            ],
            startup_timeout=audio_server.NATIVE_START_TIMEOUT,
            pending_limit=1,
        )
        run.load.assert_called_once_with("model")
        run.tokenizer.assert_called_once_with(
            "tokenizer", local_files_only=True, trust_remote_code=False
        )
        run.backend_type.assert_called_once_with(
            run.runtime, request_logger=audio_server.print_audio_request
        )
        self.assertEqual(run.app_type.call_args.args[2:5], (MODEL, TTS_PROFILE, 30.0))
        self.assertEqual(run.app_type.call_args.args[5], 4)
        run.server.server_bind.assert_called_once_with()
        run.server.server_activate.assert_called_once_with()
        run.server.serve_forever.assert_called_once_with()
        run.server.server_close.assert_called_once_with()
        run.backend_type.return_value.close.assert_called_once_with()
        self.assertIn(
            mock.call(f"Ready · {MODEL} · qwen3-tts · 4 lanes · http://127.0.0.1:8000"),
            run.status.call_args_list,
        )

    def test_main_refuses_an_engine_of_the_other_kind(self):
        status = mock.Mock()
        with self.assertRaisesRegex(SystemExit, "1"):
            self._run_main(
                audio_main_args(),
                audio_wire.ReadyEvent(4, 8192, audio_wire.ModelKind.ASR),
                status=status,
            )
        (message,) = (
            call.args[0] for call in status.call_args_list if call.kwargs.get("error")
        )
        self.assertIn("serves transcription", message)
        self.assertIn("qwen3-tts", message)

    def test_main_serves_qwen3_asr_1_7b_with_the_default_draft(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, cache = Path(temporary) / "model", Path(temporary) / "hub"
            root.mkdir()
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_asr",
                        "thinker_config": {"text_config": {"hidden_size": 2048}},
                    }
                )
            )
            folder = cache / (
                "models--" + serve_options.DEFAULT_DRAFT_MODEL.replace("/", "--")
            )
            snapshot = folder / "snapshots" / ("c" * 40)
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            readiness = audio_wire.ReadyEvent(4, 8192, audio_wire.ModelKind.ASR)
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)}):
                for options, draft in (
                    ({}, ["--draft-model", str(snapshot)]),
                    ({"no_draft_model": True}, []),
                ):
                    with self.subTest(options=options):
                        run = self._run_main(
                            audio_main_args(model_root=str(root), **options),
                            readiness,
                            profile=ASR_PROFILE,
                        )
                        run.runtime_type.assert_called_once_with(
                            ["splash", "serve-omni", str(root), *draft],
                            startup_timeout=audio_server.NATIVE_START_TIMEOUT,
                            pending_limit=1,
                        )

    def test_main_stops_on_sigterm_through_the_cleanup_path(self):
        handlers = {}

        def install(signum, handler):
            handlers[signum] = handler

        def serve():
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        runtime = mock.Mock()
        runtime.readiness = audio_wire.ReadyEvent(4, 8192, audio_wire.ModelKind.TTS)
        runtime.wait_ready.return_value = True
        server = mock.Mock(server_port=8000)
        server.serve_forever.side_effect = serve
        backend = mock.Mock()
        with (
            mock.patch.object(
                audio_server, "parse_args", return_value=audio_main_args()
            ),
            mock.patch.object(
                audio_server.ModelProfile, "load", return_value=TTS_PROFILE
            ),
            mock.patch.object(
                audio_server.AutoTokenizer, "from_pretrained", return_value=object()
            ),
            mock.patch.object(
                audio_server.audio_runtime, "AudioRuntime", return_value=runtime
            ),
            mock.patch.object(audio_server, "AudioBackend", return_value=backend),
            mock.patch.object(audio_server, "AudioFrontend"),
            mock.patch.object(audio_server, "AudioServer", return_value=server),
            mock.patch.object(audio_server.signal, "signal", side_effect=install),
            mock.patch.object(audio_server, "print_status"),
        ):
            audio_server.main()
        backend.close.assert_called_once_with()
        server.server_close.assert_called_once_with()
        self.assertEqual(
            handlers, {signal.SIGTERM: signal.SIG_IGN, signal.SIGINT: signal.SIG_IGN}
        )

    def test_main_exits_unavailable_once_the_engine_stops_for_good(self):
        # A supervisor (launchd's KeepAlive) restarts a server that exits; one
        # that stays up refusing every request with engine_failed it never
        # would.
        runtime = mock.Mock()
        runtime.readiness = audio_wire.ReadyEvent(4, 8192, audio_wire.ModelKind.TTS)
        runtime.wait_ready.return_value = True
        server = mock.Mock(server_port=8000)
        backend = mock.Mock()
        server.serve_forever.side_effect = lambda: backend.on_stopped()
        with (
            mock.patch.object(
                audio_server, "parse_args", return_value=audio_main_args()
            ),
            mock.patch.object(
                audio_server.ModelProfile, "load", return_value=TTS_PROFILE
            ),
            mock.patch.object(
                audio_server.AutoTokenizer, "from_pretrained", return_value=object()
            ),
            mock.patch.object(
                audio_server.audio_runtime, "AudioRuntime", return_value=runtime
            ),
            mock.patch.object(audio_server, "AudioBackend", return_value=backend),
            mock.patch.object(audio_server, "AudioFrontend"),
            mock.patch.object(audio_server, "AudioServer", return_value=server),
            mock.patch.object(audio_server.signal, "signal"),
            mock.patch.object(audio_server, "print_status"),
            self.assertRaises(SystemExit) as exited,
        ):
            audio_server.main()
        self.assertEqual(exited.exception.code, 69)
        self.assertTrue(wait_until(lambda: server.shutdown.called))
        backend.close.assert_called_once_with()
        server.server_close.assert_called_once_with()

    def test_parse_args_takes_the_shared_options(self):
        args = audio_server.parse_args(
            [
                "model",
                "--tokenizer",
                "tok",
                "--model",
                "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
                "--port",
                "8123",
                "--lanes",
                "2",
                "--max-memory",
                "4G",
                "--idle-release",
                "off",
                "--queue-size",
                "3",
            ]
        )
        self.assertEqual(args.model_root, "model")
        self.assertEqual((args.port, args.lanes, args.queue_size), (8123, 2, 3))
        self.assertEqual(args.max_memory, 4 * 1024**3)
        self.assertEqual(args.idle_release, math.inf)
        self.assertEqual(
            audio_server._native_command(args),
            [
                args.binary,
                "serve-omni",
                "model",
                "--max-memory",
                str(4 * 1024**3),
                "--lanes",
                "2",
                "--idle-release",
                "off",
            ],
        )
        self.assertEqual(args.max_request_size, serve_options.DEFAULT_MAX_REQUEST_BYTES)
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                audio_server.parse_args(
                    ["model", "--tokenizer", "t", "--model", "x/y", "--lanes", "0"]
                )


if __name__ == "__main__":
    unittest.main()
