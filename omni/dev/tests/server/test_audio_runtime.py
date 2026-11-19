"""AudioRuntime over a fake audio engine process: frames written, events
delivered, and the rules a speech or transcription request holds the engine
to."""

import math
import struct
import threading
import time
import unittest

from dev.tests.server_fixtures import AUDIO_READY, FakeFactory
from server import audio_protocol as audio_wire
from server import audio_runtime
from server import runtime as engine_runtime

READY_STATUS = b'{"schema_version":6,"ready":true}'


def speech_request(text=(1, 2, 3), *, deadline=45.0, **fields):
    frame = audio_wire.SpeechRequest(0, tuple(text), **fields)
    return audio_runtime.AudioRequest(frame, time.monotonic() + deadline)


def transcribe_request(samples=4, *, deadline=45.0, **fields):
    frame = audio_wire.TranscribeRequest(
        0,
        (151644,),
        (151670,),
        struct.pack(f"<{samples}f", *([0.0] * samples)),
        **fields,
    )
    return audio_runtime.AudioRequest(frame, time.monotonic() + deadline)


def pcm(count, value=0):
    return struct.pack(f"<{count}h", *([value] * count))


class Recorder:
    def __init__(self):
        self.events = []
        self.completed = threading.Event()
        self.lock = threading.Lock()

    def on_event(self, call, event):
        with self.lock:
            self.events.append(event)

    def on_complete(self, call):
        self.completed.set()


class AudioRuntimeTests(unittest.TestCase):
    def runtime(self, handler=None, **options):
        factory = FakeFactory(handler, audio=True)
        runtime = audio_runtime.AudioRuntime(
            process_factory=factory, eager_start=True, pending_limit=4, **options
        )
        self.addCleanup(runtime.close)
        return runtime, factory

    def test_command_line_for_the_engine(self):
        self.assertEqual(
            audio_runtime.engine_command("splash", "/models/x"),
            ["splash", "serve-omni", "/models/x"],
        )
        self.assertEqual(
            audio_runtime.engine_command(
                "splash",
                "/models/x",
                max_memory=2**34,
                lanes=4,
                idle_release="off",
                disable_ane=True,
                draft_model="/models/draft",
                draft_tokens=5,
                speculative_lanes=2,
            ),
            [
                "splash",
                "serve-omni",
                "/models/x",
                "--max-memory",
                str(2**34),
                "--lanes",
                "4",
                "--idle-release",
                "off",
                "--disable-ane",
                "--draft-model",
                "/models/draft",
                "--draft-tokens",
                "5",
                "--speculative-lanes",
                "2",
            ],
        )

    def test_speech_request_streams_audio_then_done(self):
        runtime, factory = self.runtime()
        process = factory.processes[0]
        self.assertEqual(runtime.readiness, AUDIO_READY)
        recorder = Recorder()
        call = runtime.submit(
            speech_request((5, 6), temperature=0.5, top_k=3, seed=9, max_frames=4),
            on_event=recorder.on_event,
            on_complete=recorder.on_complete,
        )
        (frame,) = process.stdin.wait_for(audio_wire.SpeechRequest)
        self.assertEqual(frame.request_id, call.request_id)
        self.assertEqual(frame.text_tokens, (5, 6))
        self.assertEqual((frame.temperature, frame.top_k, frame.seed), (0.5, 3, 9))
        self.assertEqual(frame.max_frames, 4)
        process.send(audio_wire.StartEvent(call.request_id, 1, 0))
        first, second = pcm(1920, 1), pcm(960, 2)
        process.send(audio_wire.AudioEvent(call.request_id, 0, first))
        process.send(audio_wire.AudioEvent(call.request_id, 1920, second))
        process.send(
            audio_wire.DoneEvent(
                call.request_id, audio_wire.FinishReason.STOP, 12, 2, 10, 20, 30, ()
            )
        )
        done = call.result(1.0)
        self.assertEqual((done.prompt_tokens, done.completion_tokens), (12, 2))
        self.assertTrue(recorder.completed.wait(1.0))
        self.assertEqual(
            [type(event) for event in recorder.events],
            [audio_wire.StartEvent, audio_wire.AudioEvent, audio_wire.AudioEvent],
        )
        self.assertEqual(recorder.events[1].pcm, first)
        self.assertEqual(recorder.events[2].sample_offset, 1920)
        self.assertEqual(runtime.pending_count, 0)
        self.assertTrue(runtime.ready)

    def test_speech_request_asking_for_codes_gets_them_before_done(self):
        runtime, factory = self.runtime()
        process = factory.processes[0]
        recorder = Recorder()
        call = runtime.submit(
            speech_request(
                reference_text_tokens=(7, 8),
                reference_codes=tuple(range(16)),
                flags=audio_wire.SPEECH_FLAG_CODES,
            ),
            on_event=recorder.on_event,
        )
        (frame,) = process.stdin.wait_for(audio_wire.SpeechRequest)
        self.assertEqual(frame.flags, audio_wire.SPEECH_FLAG_CODES)
        self.assertEqual(frame.reference_text_tokens, (7, 8))
        self.assertEqual(frame.reference_codes, tuple(range(16)))
        codes = tuple(range(32))
        process.send(audio_wire.StartEvent(call.request_id, 0, 0))
        process.send(audio_wire.AudioEvent(call.request_id, 0, pcm(3840)))
        process.send(audio_wire.CodesEvent(call.request_id, codes))
        process.send(
            audio_wire.DoneEvent(
                call.request_id, audio_wire.FinishReason.STOP, 12, 2, 10, 20, 30, ()
            )
        )
        self.assertEqual(call.result(1.0).completion_tokens, 2)
        self.assertEqual(
            recorder.events[-1], audio_wire.CodesEvent(call.request_id, codes)
        )

    def test_codes_rules(self):
        flags = audio_wire.SPEECH_FLAG_CODES
        cases = {
            "not_asked": (0, [audio_wire.CodesEvent(1, tuple(range(16)))]),
            "frames_other_than_done": (
                flags,
                [
                    audio_wire.CodesEvent(1, tuple(range(16))),
                    audio_wire.DoneEvent(
                        1, audio_wire.FinishReason.STOP, 1, 2, 0, 0, 0, ()
                    ),
                ],
            ),
            "twice": (
                flags,
                [
                    audio_wire.CodesEvent(1, tuple(range(16))),
                    audio_wire.CodesEvent(1, tuple(range(16))),
                ],
            ),
        }
        for name, (request_flags, events) in cases.items():
            with self.subTest(name=name):
                runtime, factory = self.runtime()
                process = factory.processes[0]
                call = runtime.submit(speech_request(max_frames=4, flags=request_flags))
                process.stdin.wait_for(audio_wire.SpeechRequest)
                process.send(audio_wire.StartEvent(call.request_id, 0, 0))
                for event in events:
                    process.send(event)
                with self.assertRaises(engine_runtime.ProtocolFatal):
                    call.result(1.0)
                runtime.close()

    def test_transcription_streams_tokens_then_done(self):
        runtime, factory = self.runtime()
        process = factory.processes[0]
        recorder = Recorder()
        call = runtime.submit(transcribe_request(3), on_event=recorder.on_event)
        (frame,) = process.stdin.wait_for(audio_wire.TranscribeRequest)
        self.assertEqual(frame.sample_count, 3)
        self.assertEqual(frame.prefix_tokens, (151644,))
        process.send(audio_wire.StartEvent(call.request_id, 0, 0))
        process.send(audio_wire.TokensEvent(call.request_id, 0, (10, 11)))
        process.send(audio_wire.TokensEvent(call.request_id, 2, (12,)))
        process.send(
            audio_wire.DoneEvent(
                call.request_id, audio_wire.FinishReason.STOP, 30, 3, 1, 2, 3, ()
            )
        )
        self.assertEqual(call.result(1.0).completion_tokens, 3)
        self.assertEqual(recorder.events[1].tokens, (10, 11))

    def test_an_expired_deadline_is_refused_before_a_write(self):
        runtime, factory = self.runtime()
        with self.assertRaises(TimeoutError):
            runtime.submit(speech_request(deadline=-1.0))
        self.assertEqual(factory.processes[0].stdin.records, [])
        self.assertEqual(runtime.pending_count, 0)
        # An infinite deadline is a request without one.
        call = runtime.submit(speech_request(deadline=math.inf))
        self.assertEqual(runtime.pending_count, 1)
        call.cancel()

    def test_invalid_requests_fail_without_reaching_the_engine(self):
        runtime, factory = self.runtime()
        with self.assertRaises(engine_runtime.RequestFailed) as refused:
            runtime.submit(speech_request(()))
        self.assertEqual(refused.exception.code, b"limit_exceeded")
        self.assertEqual(factory.processes[0].stdin.records, [])

    def test_cancel_writes_a_version_11_cancel_frame(self):
        runtime, factory = self.runtime()
        process = factory.processes[0]
        call = runtime.submit(speech_request())
        process.stdin.wait_for(audio_wire.SpeechRequest)
        self.assertTrue(call.cancel())
        (cancel,) = process.stdin.wait_for(audio_wire.CancelFrame)
        self.assertEqual(cancel.request_id, call.request_id)
        _, raw = process.stdin.records_of(audio_wire.CancelFrame)[0]
        self.assertEqual(raw[4:6], (11).to_bytes(2, "little"))
        process.send(
            audio_wire.DoneEvent(
                call.request_id, audio_wire.FinishReason.CANCELLED, 0, 0, 0, 0, 0, ()
            )
        )
        self.assertIs(call.result(1.0).reason, audio_wire.FinishReason.CANCELLED)

    def test_status_requests_are_answered(self):
        def answer(process, message):
            if isinstance(message, audio_wire.StatusRequestFrame):
                process.send(
                    audio_wire.StatusJsonEvent(message.correlation_id, READY_STATUS)
                )

        runtime, _ = self.runtime(answer)
        self.assertEqual(runtime.status(timeout=1.0).json, READY_STATUS)

    def test_protocol_violations_fail_the_engine(self):
        cases = {
            "audio_gap": lambda process, call: [
                audio_wire.StartEvent(call.request_id, 0, 0),
                audio_wire.AudioEvent(call.request_id, 10, pcm(2)),
            ],
            "audio_before_start": lambda process, call: [
                audio_wire.AudioEvent(call.request_id, 0, pcm(2))
            ],
            "tokens_for_speech": lambda process, call: [
                audio_wire.StartEvent(call.request_id, 0, 0),
                audio_wire.TokensEvent(call.request_id, 0, (1,)),
            ],
            "too_many_frames": lambda process, call: [
                audio_wire.StartEvent(call.request_id, 0, 0),
                audio_wire.DoneEvent(
                    call.request_id, audio_wire.FinishReason.STOP, 1, 5, 0, 0, 0, ()
                ),
            ],
            "done_before_start": lambda process, call: [
                audio_wire.DoneEvent(
                    call.request_id, audio_wire.FinishReason.STOP, 1, 0, 0, 0, 0, ()
                )
            ],
            "unknown_request": lambda process, call: [
                audio_wire.AudioEvent(call.request_id + 7, 0, pcm(2))
            ],
        }
        for name, events in cases.items():
            with self.subTest(name=name):
                runtime, factory = self.runtime()
                process = factory.processes[0]
                call = runtime.submit(speech_request(max_frames=4))
                process.stdin.wait_for(audio_wire.SpeechRequest)
                for event in events(process, call):
                    process.send(event)
                with self.assertRaises(engine_runtime.ProtocolFatal):
                    call.result(1.0)
                self.assertFalse(runtime.ready)
                runtime.close()

    def test_transcription_rules(self):
        cases = {
            "count_mismatch": [
                audio_wire.TokensEvent(1, 0, (1, 2)),
                audio_wire.DoneEvent(
                    1, audio_wire.FinishReason.STOP, 3, 3, 0, 0, 0, ()
                ),
            ],
            "audio_for_transcription": [audio_wire.AudioEvent(1, 0, pcm(2))],
            "length_without_the_maximum": [
                audio_wire.TokensEvent(1, 0, (1,)),
                audio_wire.DoneEvent(
                    1, audio_wire.FinishReason.LENGTH, 3, 1, 0, 0, 0, ()
                ),
            ],
        }
        for name, events in cases.items():
            with self.subTest(name=name):
                runtime, factory = self.runtime()
                process = factory.processes[0]
                call = runtime.submit(transcribe_request(max_tokens=2))
                process.stdin.wait_for(audio_wire.TranscribeRequest)
                process.send(audio_wire.StartEvent(call.request_id, 0, 0))
                for event in events:
                    process.send(event)
                with self.assertRaises(engine_runtime.ProtocolFatal):
                    call.result(1.0)
                runtime.close()

    def test_a_relaunched_engine_must_announce_the_same_kind(self):
        runtime, factory = self.runtime(startup_timeout=2.0)
        first = factory.processes[0]
        # The engine dies; the next announces a transcription model.
        factory.initial_output = b""
        first.close_stdout()
        deadline = time.monotonic() + 2
        while runtime.ready and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(runtime.ready)

        def wait():
            try:
                runtime.wait_ready(timeout=2.0)
            except engine_runtime.EngineRuntimeError:
                pass

        started = threading.Thread(target=wait)
        started.start()
        deadline = time.monotonic() + 2
        while len(factory.processes) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        factory.processes[1].send(
            audio_wire.ReadyEvent(8, 8192, audio_wire.ModelKind.ASR)
        )
        started.join(3)
        self.assertIsNotNone(runtime.fatal_error)
        self.assertIn("model kind changed", str(runtime.fatal_error))


if __name__ == "__main__":
    unittest.main()
