import errno
import http.client
import json
import select
import socket
import struct
import sys
import time
import unittest
from unittest import mock

from dev.tests.audio_fixtures import (
    AudioHarness,
    AudioHarnessTestCase,
    FakeAudioRuntime,
    speech_body,
)
from dev.tests.server_fixtures import FakeFactory
from server import audio_protocol as audio_wire
from server import audio_runtime, http_server, json_codec
from server.errors import APIError


class RequestContractTests(AudioHarnessTestCase):
    def test_trailing_bytes_do_not_hide_connection_close(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        handler = object.__new__(http_server.FrontendHandler)
        handler.connection = local
        self.assertFalse(handler._client_disconnected())
        peer.sendall(b"trailing")
        peer.close()
        self.assertFalse(handler._client_disconnected())
        self.assertTrue(handler._client_disconnected())

    def test_descriptor_exhaustion_does_not_read_as_a_disconnect(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        handler = object.__new__(http_server.FrontendHandler)
        handler.connection = local
        exhausted = OSError(errno.EMFILE, "Too many open files")
        with mock.patch("select.kqueue", side_effect=exhausted):
            self.assertFalse(handler._client_disconnected())

    def test_reset_client_reads_as_disconnected(self):
        listener = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(listener.close)
        client = socket.create_connection(listener.getsockname())
        accepted, _ = listener.accept()
        self.addCleanup(accepted.close)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        self.assertTrue(select.select([accepted], [], [], 2)[0])
        handler = object.__new__(http_server.FrontendHandler)
        handler.connection = accepted
        self.assertTrue(handler._client_disconnected())

    def test_unsupported_http_version_is_rejected_before_header_validation(self):
        handler = object.__new__(http_server.FrontendHandler)
        handler.server = mock.Mock()
        handler.connection = mock.Mock()
        handler.request_version = "HTTP/0.9"
        handler.headers = {}
        handler.send_error = mock.Mock()
        with mock.patch.object(
            http_server.BaseHTTPRequestHandler, "parse_request", return_value=True
        ):
            self.assertFalse(handler.parse_request())
        handler.send_error.assert_called_once_with(505, "HTTP version not supported")
        self.assertTrue(handler.close_connection)

    def test_unparsable_requests_get_an_http_1_1_json_error(self):
        harness = AudioHarness(FakeAudioRuntime())
        self.addCleanup(harness.close)
        # Each request is all the server reads: bytes it left unread would
        # reset the connection under its response.
        for data, status, code in (
            (b"GARBAGE\r\n", 400, "invalid_request_error"),
            (b"GET / FOO/1.1\r\n", 400, "invalid_request_error"),
            # HTTP/0.9's request line, which has no version. Python before
            # 3.13 reads headers after it all the same.
            (
                b"GET /\r\n" + b"\r\n" * (sys.version_info < (3, 13)),
                505,
                "http_version_not_supported",
            ),
            # One byte over the stdlib's request and header line limits.
            (b"x" * 65537, 414, "invalid_request_error"),
            (
                b"POST /v1/audio/speech HTTP/1.1\r\n" + b"x" * 65537,
                431,
                "invalid_request_error",
            ),
        ):
            with self.subTest(data=data[:24]):
                client = socket.create_connection(
                    harness.server.server_address, timeout=2
                )
                self.addCleanup(client.close)
                client.sendall(data)
                response = http.client.HTTPResponse(client)
                response.begin()
                self.assertEqual((response.version, response.status), (11, status))
                self.assertEqual(response.getheader("Connection"), "close")
                self.assertEqual(response.getheader("Content-Type"), "application/json")
                payload = json.loads(response.read())
                self.assertNotIn("type", payload)
                self.assertEqual(payload["error"]["code"], code)
                self.assertTrue(payload["error"]["message"])

    def test_every_error_on_a_path_answers_as_openai(self):
        harness = AudioHarness(FakeAudioRuntime())
        self.addCleanup(harness.close)
        not_found = {
            "error": {
                "message": "not found",
                "type": "invalid_request_error",
                "code": "not_found",
            }
        }
        for method, path, status, payload in (
            # A method the server does not implement is a server error.
            (
                "PUT",
                "/v1/audio/speech",
                501,
                {
                    "error": {
                        "message": "Unsupported method ('PUT')",
                        "type": "server_error",
                        "code": "not_implemented",
                    }
                },
            ),
            ("GET", "/v1/audio/speech", 404, not_found),
            ("DELETE", "/v1/models", 404, not_found),
            ("POST", "/v1/audio/speechX", 404, not_found),
            ("POST", "/v1/unknown", 404, not_found),
            ("GET", "/v1/chat/completions", 404, not_found),
        ):
            with self.subTest(method=method, path=path):
                response = harness.request(
                    method, path, {} if method == "POST" else None
                )
                self.assertEqual(
                    (response[0], json.loads(response[2])), (status, payload)
                )

    def test_only_a_retryable_refusal_invites_a_retry(self):
        harness = AudioHarness(FakeAudioRuntime())
        self.addCleanup(harness.close)

        def retry_after(path):
            connection = http.client.HTTPConnection(
                *harness.server.server_address, timeout=3
            )
            self.addCleanup(connection.close)
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            return response.status, response.getheader("Retry-After")

        self.assertEqual(retry_after("/ready"), (200, None))
        self.assertEqual(retry_after("/v1/unknown"), (404, None))
        with mock.patch.object(harness.backend, "is_ready", return_value=False):
            self.assertEqual(retry_after("/ready"), (503, "1"))

    def test_missing_native_executable_has_upgrade_guidance(self):
        with self.assertRaisesRegex(
            audio_runtime.EngineUnhealthy,
            "restart Splash from the current installation",
        ):
            audio_runtime.AudioRuntime(
                process_factory=mock.Mock(side_effect=FileNotFoundError()),
                eager_start=True,
            )

    def test_head_and_options(self):
        harness = AudioHarness(FakeAudioRuntime())
        self.addCleanup(harness.close)
        status, headers, payload = harness.request("HEAD", "/health")
        self.assertEqual((status, payload), (200, b""))
        self.assertNotIn("Python", str(headers))
        self.assertEqual(harness.request("OPTIONS", "/v1/audio/speech")[0], 204)

    def test_stream_failure_before_admission_preserves_retryable_http_status(self):
        class RejectBeforeStart(FakeAudioRuntime):
            def _run(self, call):
                call.complete(error=APIError(503, "retry later", "resource_timeout"))

        harness = AudioHarness(RejectBeforeStart())
        self.addCleanup(harness.close)
        status, headers, payload = harness.request(
            "POST", "/v1/audio/speech", speech_body(stream=True)
        )
        self.assertEqual(status, 503, payload)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(json.loads(payload)["error"]["code"], "resource_timeout")

    def test_json_nested_deeper_than_it_is_read_is_a_request_error(self):
        harness = AudioHarness(FakeAudioRuntime())
        self.addCleanup(harness.close)
        limit = json_codec.MAX_DEPTH

        def post(text):
            connection = http.client.HTTPConnection(
                *harness.server.server_address, timeout=3
            )
            self.addCleanup(connection.close)
            connection.request(
                "POST",
                "/v1/audio/speech",
                text.encode(),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            return response.status, payload.get("error") and payload["error"]["message"]

        def arrays(depth):
            return "[" * depth + "]" * depth

        def with_field(body, value):
            return json.dumps(body)[:-1] + f',"extra":{value}}}'

        invalid = (400, "invalid JSON request body")
        for text, expected in (
            (with_field(speech_body(), arrays(limit)), invalid),
            (with_field(speech_body(), arrays(10_000)), invalid),
        ):
            with self.subTest(size=len(text)):
                self.assertEqual(post(text), expected)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_unacknowledged_cancel_fails_generation_and_releases_calls(self):
        factory = FakeFactory(audio=True)
        engine = audio_runtime.AudioRuntime(process_factory=factory, eager_start=True)
        self.addCleanup(engine.close)
        engine._cancel_grace_seconds = 0.02
        request = audio_runtime.AudioRequest(
            audio_wire.SpeechRequest(0, (11, 12, 13)), time.monotonic() + 10
        )
        call = engine.submit(request)
        call.cancel()
        with self.assertRaises(audio_runtime.EngineUnhealthy):
            call.result(1)
        self.assertTrue(call.done)


if __name__ == "__main__":
    unittest.main()
