import http.client
import io
import json
import socket
import struct
import threading
import time
import unittest
from unittest import mock

from dev.tests.audio_fixtures import (
    AudioHarnessTestCase,
    AudioPlan,
    FakeAudioRuntime,
    pcm,
    pcm_values,
    speech_body,
)
from dev.tests.server_fixtures import FOREVER
from server import audio_io, audio_runtime, audio_server, http_server

CHUNK = pcm([7] * 1920)


def plan(**options):
    """A speech plan of one chunk of audio."""
    return AudioPlan(chunks=[CHUNK], **options)


def read_cut_short(response):
    """What a stream cut short before its final chunk delivered."""
    try:
        return response.read()
    except http.client.IncompleteRead as cut:
        return cut.partial


class RequestFailureTests(AudioHarnessTestCase):
    def assert_served(self, harness):
        """The next speech request is answered with its plan's audio (its
        end faded, audio_io.SpeechEdges)."""
        status, _, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(response_format="pcm")
        )
        self.assertEqual(status, 200)
        edges = audio_io.SpeechEdges(False)
        expected = edges.push(struct.pack("<1920h", *(7,) * 1920)) + edges.finish()
        self.assertEqual(pcm_values(payload), pcm_values(expected))

    def test_stream_write_timeout_cancels_the_submitted_job(self):
        # The engine still decodes when the write fails.
        runtime = FakeAudioRuntime(AudioPlan(chunks=[CHUNK, CHUNK], delay=1))
        harness = self.harness(runtime)
        # The first write of the stream, the WAV header, times out.
        with mock.patch.object(audio_io, "wav_header", side_effect=TimeoutError):
            connection, response = harness.open_stream(
                "/v1/audio/speech", speech_body(stream=True)
            )
            self.addCleanup(connection.close)
            self.assertEqual(response.status, 200)
            self.assertEqual(read_cut_short(response), b"")
        deadline = time.monotonic() + 1
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)

    def test_very_large_timeout_does_not_overflow_wait(self):
        blocking = plan(block=True)
        harness = self.harness(FakeAudioRuntime(blocking), timeout=1e308)
        release = threading.Timer(0.03, blocking.release.set)
        release.start()
        self.addCleanup(release.cancel)
        status, _, _ = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 200)

    def test_pending_limit_returns_retryable_http_overload(self):
        blocking = plan(block=True)
        runtime = FakeAudioRuntime(blocking)
        harness = self.harness(runtime, queue_size=1)
        head = harness.app.prepare_speech(speech_body(), deadline=FOREVER).job
        harness.backend.submit(head)
        self.assertTrue(blocking.started.wait(1))

        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        self.addCleanup(connection.close)
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(speech_body()),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 503)
        self.assertEqual(response.getheader("Retry-After"), "1")
        self.assertEqual(
            json.loads(response.read())["error"]["code"], "frontend_overloaded"
        )
        self.assertEqual(len(runtime.requests), 1)
        blocking.release.set()

    def test_active_timeout_signals_and_next_request_runs(self):
        blocking = plan(block=True)
        runtime = FakeAudioRuntime(blocking, plan())
        harness = self.harness(runtime, timeout=0.03)
        status, _, _ = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 504)
        self.assertEqual(runtime.cancel_count, 1)
        harness.app.request_timeout = 2
        self.assert_served(harness)

    def test_http_stream_graceful_disconnect_cancels_and_next_request_runs(self):
        streaming = AudioPlan(chunks=[CHUNK] * 100, delay=0.02)
        runtime = FakeAudioRuntime(streaming, plan())
        harness = self.harness(runtime, timeout=10)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=2
        )
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(speech_body(stream=True)),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        response.read(64)
        response.close()
        connection.close()
        deadline = time.monotonic() + 2
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)
        self.assert_served(harness)

    def test_http_nonstream_disconnect_cancels_and_next_request_runs(self):
        blocking = plan(block=True)
        runtime = FakeAudioRuntime(blocking, plan())
        harness = self.harness(runtime, timeout=10)
        connection = http.client.HTTPConnection(*harness.server.server_address)
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(speech_body()),
            {"Content-Type": "application/json"},
        )
        self.assertTrue(blocking.started.wait(1))
        connection.close()
        deadline = time.monotonic() + 2
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)
        self.assert_served(harness)

    def test_stream_disconnect_before_headers_cancels(self):
        waiting = plan(block=True, before_start=True)
        runtime = FakeAudioRuntime(waiting)
        harness = self.harness(runtime, timeout=10)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=2
        )
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(speech_body(stream=True)),
            {"Content-Type": "application/json"},
        )
        deadline = time.monotonic() + 1
        while not runtime.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(runtime.calls), 1)
        connection.sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        connection.close()
        waiting.start_release.set()
        deadline = time.monotonic() + 2
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)

    def test_engine_refusal_does_not_poison_next_request(self):
        refused = AudioPlan(
            exception=audio_runtime.RequestFailed(
                1, b"invalid_request", b"text is empty", retryable=False
            )
        )
        runtime = FakeAudioRuntime(refused, plan())
        harness = self.harness(runtime)
        status, _, payload = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["code"], "invalid_request")
        self.assert_served(harness)

    def test_capacity_failure_is_a_bad_request_naming_the_limits(self):
        native = (
            "could not allocate KV pages: engine memory budget exceeded "
            "(additional_pages=12, free_pages=0)"
        )
        capacity = AudioPlan(
            exception=audio_runtime.RequestFailed(
                1, b"capacity_exhausted", native.encode(), retryable=False
            )
        )
        harness = self.harness(FakeAudioRuntime(capacity, plan()))
        # Retrying fails the same way, so no response invites a retry.
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        self.addCleanup(connection.close)
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(speech_body()),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        error = json.loads(response.read())["error"]
        self.assertEqual(response.status, 400)
        self.assertIsNone(response.getheader("Retry-After"))
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertEqual(error["code"], "capacity_exhausted")
        for text in ("--max-memory", "--lanes", native):
            self.assertIn(text, error["message"])
        self.assert_served(harness)

    def test_unexpected_runtime_error_does_not_kill_backend(self):
        runtime = FakeAudioRuntime(AudioPlan(exception=RuntimeError("boom")), plan())
        harness = self.harness(runtime)
        status, _, payload = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(payload)["error"]["code"], "runtime_error")
        self.assert_served(harness)

    def test_unexpected_handler_error_cancels_and_is_structured(self):
        harness = self.harness(FakeAudioRuntime(plan(block=True)))
        stderr = io.StringIO()
        with (
            mock.patch.object(
                harness.backend, "cancel", wraps=harness.backend.cancel
            ) as cancel,
            mock.patch.object(
                audio_server.AudioHandler,
                "_speech_complete",
                side_effect=RuntimeError("boom"),
            ),
            mock.patch.object(http_server.sys, "stderr", stderr),
        ):
            status, _, payload = harness.request(
                "POST", "/v1/audio/speech", speech_body()
            )
        self.assertEqual(status, 500, payload)
        self.assertEqual(json.loads(payload)["error"]["code"], "internal_server_error")
        cancel.assert_called_once()
        # Where the server last had the error, and not its message.
        self.assertRegex(
            stderr.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · internal_server_error · RuntimeError · "
            r"server/audio_server\.py:\d+\n$",
        )
        self.assertNotIn("boom", stderr.getvalue())

    def test_value_errors_past_the_body_are_internal_errors(self):
        # Only a body the server cannot read is invalid JSON. A ValueError or
        # RecursionError raised while a request is prepared or answered is
        # the server's own: logged, answered with 500, and the job it
        # submitted is cancelled.
        for error in (ValueError("late"), RecursionError()):
            for target, name in (
                ("app", "prepare_speech"),
                ("handler", "_speech_complete"),
            ):
                with self.subTest(error=type(error).__name__, name=name):
                    harness = self.harness(FakeAudioRuntime(plan(block=True)))
                    owner = (
                        harness.app if target == "app" else audio_server.AudioHandler
                    )
                    with (
                        mock.patch.object(owner, name, side_effect=error),
                        mock.patch.object(
                            harness.backend, "cancel", wraps=harness.backend.cancel
                        ) as cancel,
                        mock.patch.object(http_server, "log_unexpected") as logged,
                    ):
                        status, _, payload = harness.request(
                            "POST", "/v1/audio/speech", speech_body()
                        )
                    self.assertEqual(status, 500, payload)
                    self.assertEqual(
                        json.loads(payload)["error"]["code"], "internal_server_error"
                    )
                    logged.assert_called_once_with(error)
                    if target == "handler":
                        cancel.assert_called_once()
                    else:
                        cancel.assert_not_called()

    def test_unexpected_stream_error_cuts_the_stream_and_cancels(self):
        # The stream fails while the engine still decodes.
        streaming = AudioPlan(chunks=[CHUNK, CHUNK], delay=1)
        harness = self.harness(FakeAudioRuntime(streaming))
        stderr = io.StringIO()
        with (
            mock.patch.object(
                harness.backend, "cancel", wraps=harness.backend.cancel
            ) as cancel,
            mock.patch.object(audio_io, "wav_header", side_effect=RuntimeError("boom")),
            mock.patch.object(http_server.sys, "stderr", stderr),
        ):
            connection, response = harness.open_stream(
                "/v1/audio/speech", speech_body(stream=True)
            )
            self.addCleanup(connection.close)
            # The response began before the failure: it ends short, without
            # its final chunk, and no error can be told inside it.
            self.assertEqual(response.status, 200)
            self.assertEqual(read_cut_short(response), b"")
        cancel.assert_called_once()
        self.assertTrue(streaming.cancelled.is_set())
        self.assertRegex(
            stderr.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · internal_server_error · RuntimeError · "
            r"server/audio_server\.py:\d+\n$",
        )


if __name__ == "__main__":
    unittest.main()
