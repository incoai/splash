import argparse
import concurrent.futures
import http.client
import io
import json
import math
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.tests import audio_fixtures as fixtures
from server import audio_protocol as audio_wire
from server import connections, http_server, json_codec, serve_options
from server import runtime as engine_runtime
from server.errors import APIError, RequestValidationError

SPEECH = "/v1/audio/speech"


def padded_body(size):
    """A speech request body of exactly `size` bytes."""
    return json.dumps(fixtures.speech_body()).encode().ljust(size)


class HttpBodyBudgetTests(unittest.TestCase):
    def harness(self, **kwargs):
        runtime = kwargs.pop("runtime", fixtures.FakeAudioRuntime())
        harness = fixtures.AudioHarness(runtime, **kwargs)
        self.addCleanup(harness.close)
        return harness

    def wait_bytes(self, harness, amount):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if harness.server.request_bodies.stats()["active"] == amount:
                return
            time.sleep(0.005)
        self.assertEqual(harness.server.request_bodies.stats()["active"], amount)

    def wait_connections(self, harness, count, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if harness.server.connections.stats()["active"] == count:
                return
            time.sleep(0.005)
        self.assertEqual(harness.server.connections.stats()["active"], count)

    def headers(self, harness, length, path=SPEECH):
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=2
        )
        self.addCleanup(connection.close)
        connection.putrequest("POST", path)
        connection.putheader(
            "Content-Type",
            "multipart/form-data; boundary=x"
            if path.startswith("/v1/audio/transcriptions")
            else "application/json",
        )
        connection.putheader("Content-Length", str(length))
        connection.endheaders()
        return connection

    def test_config_and_exact_body_boundary(self):
        for value in ("128M", "128MB", "128MiB", "134217728"):
            self.assertEqual(serve_options.parse_request_size(value), 128 * 1024**2)
        for value in ("auto", "0", "-1", "bad", str(2**64)):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                serve_options.parse_request_size(value)
        harness = self.harness(max_request_bytes=128)
        body = padded_body(128)
        connection = self.headers(harness, len(body))
        connection.send(body)
        response = connection.getresponse()
        self.assertEqual(response.status, 200, response.read())
        response.read()
        connection.close()
        self.wait_bytes(harness, 0)
        for path in (SPEECH, "/v1/audio/transcriptions?beta=true"):
            with self.subTest(path=path):
                response = self.headers(harness, 129, path).getresponse()
                self.assertEqual(response.status, 413)
                self.assertIn("129 bytes; limit is 128 bytes", response.read().decode())
        self.wait_bytes(harness, 0)
        status = harness.server.status()["http"]
        self.assertEqual(status["max_request_bytes"], 128)
        self.assertEqual(status["request_body_bytes"]["active"], 0)

    def test_body_decode_preserves_json_encodings_and_strict_validation(self):
        def parse(payload):
            handler = object.__new__(http_server.FrontendHandler)
            handler.headers = http.client.HTTPMessage()
            handler.headers["Content-Type"] = "application/json"
            handler.headers["Content-Length"] = str(len(payload))
            handler.server = SimpleNamespace(
                max_request_bytes=4096,
                request_bodies=connections.HttpAdmission(4096),
            )
            handler.connection = SimpleNamespace(settimeout=lambda _: None)
            handler.rfile = io.BytesIO(payload)
            try:
                return handler._read_json_body(time.monotonic() + 1)
            finally:
                handler._body_reservation.release()
                self.assertEqual(handler.server.request_bodies.active, 0)

        values = ({"content": "你好 🦆"}, {"content": "\ud800"})
        for value in values:
            for encoding in (
                "utf-8",
                "utf-8-sig",
                "utf-16",
                "utf-16-le",
                "utf-16-be",
                "utf-32",
                "utf-32-le",
                "utf-32-be",
            ):
                with self.subTest(value=value, encoding=encoding):
                    payload = json.dumps(value, ensure_ascii=False).encode(
                        encoding, "surrogatepass"
                    )
                    self.assertEqual(parse(payload), json_codec.loads(payload))
        for payload in (
            b"{",
            b"\xff",
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":-Infinity}',
            b'"\x00"',
            b"\xef\xbb\xbf\xef\xbb\xbf{}",
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    json_codec.loads(payload)
                # The body reader answers each with its own 400.
                with self.assertRaises(RequestValidationError) as raised:
                    parse(payload)
                self.assertEqual(
                    (raised.exception.status, raised.exception.message),
                    (400, "invalid JSON request body"),
                )

    def test_shared_budget_rejects_before_reading_and_recovers_on_disconnect(self):
        with mock.patch.object(http_server, "DEFAULT_REQUEST_BODY_BUDGET", 256):
            harness = self.harness(max_request_bytes=128, queue_size=16)
        uploads = [self.headers(harness, 128) for _ in range(2)]
        self.wait_bytes(harness, 256)
        for path in (SPEECH, "/v1/audio/transcriptions"):
            with self.subTest(path=path):
                response = self.headers(harness, 1, path).getresponse()
                self.assertEqual(response.status, 503)
                self.assertIn("request body capacity", response.read().decode())
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        self.assertEqual(harness.request("GET", "/status")[0], 200)
        for upload in uploads:
            upload.close()
        self.wait_bytes(harness, 0)
        self.assertEqual(
            harness.request("POST", SPEECH, fixtures.speech_body())[0], 200
        )
        self.wait_bytes(harness, 0)

    def test_rejections_before_the_body_reach_a_client_still_uploading(self):
        # Closing with the upload unread would reset the connection under a
        # client that sends the whole body before reading the response.
        harness = self.harness(max_request_bytes=1024, api_key="test-key")
        body = b" " * (8 * 1024 * 1024)
        authorized = {"Authorization": "Bearer test-key"}
        for path, headers, status in (
            (SPEECH, {}, 401),
            (SPEECH, authorized, 413),
            ("/v1/unknown", authorized, 404),
        ):
            with self.subTest(status=status):
                connection = http.client.HTTPConnection(
                    *harness.server.server_address, timeout=5
                )
                self.addCleanup(connection.close)
                connection.request(
                    "POST",
                    path,
                    body,
                    {"Content-Type": "application/json", **headers},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, status)
                response.read()

    def _refused_upload(self, harness, sent, length):
        upload = socket.create_connection(harness.server.server_address, timeout=5)
        self.addCleanup(upload.close)
        upload.sendall(
            b"POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {length}\r\n\r\n".encode()
            + b" " * sent
        )
        return upload

    def test_a_refusal_reaches_a_client_that_uploads_in_chunks(self):
        size = 256 * 1024
        with mock.patch.object(http_server, "DEFAULT_REQUEST_BODY_BUDGET", 2 * size):
            harness = self.harness(max_request_bytes=size, queue_size=4, timeout=15)
        held = [self.headers(harness, size) for _ in range(2)]
        self.wait_bytes(harness, 2 * size)
        # The shared budget is full, so the server refuses this body unread;
        # the client sends half, then the rest in chunks.
        refused = self._refused_upload(harness, size // 2, size)
        for _ in range(8):
            refused.sendall(b" " * (size // 16))
            time.sleep(0.01)
        response = http.client.HTTPResponse(refused)
        response.begin()
        self.assertEqual(response.status, 503)
        self.assertIn("request body capacity", response.read().decode())
        # The server is done with the connection once all of the upload has
        # arrived, though the client keeps it open.
        self.wait_connections(harness, len(held), 0.5)

    def test_a_refused_upload_that_stalls_is_dropped(self):
        harness = self.harness(max_request_bytes=128, io_timeout=1)
        refused = self._refused_upload(harness, 50, 200)
        response = http.client.HTTPResponse(refused)
        response.begin()
        self.assertEqual(response.status, 413)
        self.wait_connections(harness, 1, 0.5)
        # Like any upload, it may pause for the inactivity timeout, no longer.
        self.wait_connections(harness, 0, 2)

    def test_parsed_bodies_stay_charged_while_preparation_is_pending(self):
        with mock.patch.object(http_server, "DEFAULT_REQUEST_BODY_BUDGET", 4096):
            harness = self.harness(max_request_bytes=1024, queue_size=8)
        body = fixtures.speech_body(extra="x" * 850)
        size = len(json.dumps(body).encode())
        release = threading.Event()
        prepare_lock = threading.Lock()
        prepare = harness.app.prepare_speech

        def paused(body, **kwargs):
            # The large requests wait; a small one is prepared at once.
            if "extra" in body and not release.wait(3):
                raise TimeoutError("preparation test timed out")
            with prepare_lock:
                return prepare(body, **kwargs)

        with mock.patch.object(harness.app, "prepare_speech", side_effect=paused):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                calls = [
                    pool.submit(harness.request, "POST", SPEECH, body) for _ in range(4)
                ]
                try:
                    self.wait_bytes(harness, 4 * size)
                    self.assertEqual(
                        harness.request("POST", SPEECH, fixtures.speech_body())[0],
                        200,
                    )
                    response = self.headers(harness, size).getresponse()
                    self.assertEqual(response.status, 503)
                    response.read()
                finally:
                    release.set()
                self.assertTrue(all(call.result()[0] == 200 for call in calls))
        self.wait_bytes(harness, 0)

    def test_partial_upload_timeout_parse_and_preparation_failures_release(self):
        harness = self.harness(io_timeout=0.1)
        upload = self.headers(harness, 100)
        upload.send(b"{")
        response = upload.getresponse()
        self.assertEqual(response.status, 408)
        response.read()
        self.wait_bytes(harness, 0)
        for data in (b"{", b"[]", b"\xff", b"[" * 2000):
            self.assertEqual(harness.raw_post(data, len(data))[0], 400)
            self.wait_bytes(harness, 0)
        body = fixtures.speech_body()
        for error, status in (
            (APIError(400, "bad input"), 400),
            (RuntimeError("test"), 500),
        ):
            with (
                mock.patch.object(harness.app, "prepare_speech", side_effect=error),
                mock.patch.object(http_server, "log_unexpected"),
            ):
                self.assertEqual(harness.request("POST", SPEECH, body)[0], status)
            self.wait_bytes(harness, 0)
        with mock.patch.object(
            harness.backend.runtime,
            "submit",
            side_effect=engine_runtime.PendingLimitExceeded("full"),
        ):
            self.assertEqual(harness.request("POST", SPEECH, body)[0], 503)
        self.wait_bytes(harness, 0)

    def test_an_upload_stays_charged_until_its_request_ends(self):
        plan = fixtures.AudioPlan(
            chunks=[fixtures.pcm([0] * audio_wire.SAMPLES_PER_FRAME)], block=True
        )
        harness = self.harness(runtime=fixtures.FakeAudioRuntime(plan))
        # The response starts with the first chunk, which the engine holds
        # back: the request is in flight, its body charged until it ends.
        body = json.dumps(fixtures.speech_body(stream=True)).encode()
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=5
        )
        self.addCleanup(connection.close)
        connection.request("POST", SPEECH, body, {"Content-Type": "application/json"})
        self.assertTrue(plan.started.wait(1))
        self.assertEqual(harness.server.request_bodies.stats()["active"], len(body))
        self.assertEqual(harness.server.requests.stats()["active"], 1)
        plan.release.set()
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        response.read()
        self.wait_bytes(harness, 0)

    def test_upload_deadline_scales_with_body_size_and_caps_at_request_deadline(self):
        for length, request_seconds, expected in (
            (1024, 1800, 30 + 1 / 512),
            (128 * 1024**2, 1800, 286),
            (256 * 1024**2, 1800, 542),
            (256 * 1024**2, math.inf, 542),
            (128 * 1024**2, 5, 5),
        ):
            with self.subTest(length=length, request_seconds=request_seconds):
                now = 0.0
                socket_timeout = 30.0

                def settimeout(value):
                    nonlocal socket_timeout
                    socket_timeout = value

                def drip(size):
                    nonlocal now
                    now += min(10, socket_timeout)
                    if socket_timeout < 10:
                        raise TimeoutError
                    return b" "

                handler = object.__new__(http_server.FrontendHandler)
                handler.headers = http.client.HTTPMessage()
                handler.headers["Content-Type"] = "application/json"
                handler.headers["Content-Length"] = str(length)
                handler.server = SimpleNamespace(
                    max_request_bytes=length,
                    request_bodies=connections.HttpAdmission(length),
                )
                handler.connection = SimpleNamespace(settimeout=settimeout)
                handler.rfile = SimpleNamespace(read1=drip)
                try:
                    with (
                        mock.patch.object(
                            http_server.time, "monotonic", side_effect=lambda: now
                        ),
                        self.assertRaises(TimeoutError),
                    ):
                        handler._read_json_body(request_seconds)
                    self.assertAlmostEqual(now, expected)
                finally:
                    handler._body_reservation.release()
                self.assertEqual(handler.server.request_bodies.active, 0)

    @mock.patch.object(http_server, "HTTP_UPLOAD_BYTES_PER_SECOND", 20)
    def test_active_upload_can_outlast_idle_timeout(self):
        harness = self.harness(io_timeout=2, timeout=15)
        payload = padded_body(100)
        connection = self.headers(harness, len(payload))
        for offset in range(0, len(payload), 10):
            connection.send(payload[offset : offset + 10])
            time.sleep(0.25)
        response = connection.getresponse()
        self.assertEqual(response.status, 200, response.read())
        response.read()
        self.wait_bytes(harness, 0)

    @mock.patch.object(http_server, "HTTP_UPLOAD_BYTES_PER_SECOND", 256)
    def test_drip_uploads_expire_and_restore_shared_capacity(self):
        with mock.patch.object(http_server, "DEFAULT_REQUEST_BODY_BUDGET", 256):
            harness = self.harness(max_request_bytes=128, io_timeout=0.4, timeout=4)
        uploads = [self.headers(harness, 128) for _ in range(2)]
        self.wait_bytes(harness, 256)
        response = self.headers(harness, 1).getresponse()
        self.assertEqual(response.status, 503, response.read())
        response.read()
        for _ in range(14):
            for connection in uploads:
                try:
                    connection.send(b" ")
                except OSError:
                    pass
            time.sleep(0.08)
        for connection in uploads:
            response = connection.getresponse()
            self.assertEqual(response.status, 408, response.read())
            response.read()
        self.wait_bytes(harness, 0)
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        self.assertEqual(
            harness.request("POST", SPEECH, fixtures.speech_body())[0], 200
        )
        self.wait_bytes(harness, 0)

    def test_active_upload_still_obeys_request_deadline(self):
        harness = self.harness(io_timeout=1, timeout=0.3)
        connection = self.headers(harness, 100)
        for _ in range(5):
            try:
                connection.send(b" ")
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(0.05)
        response = connection.getresponse()
        self.assertEqual(response.status, 408, response.read())
        response.read()
        self.wait_bytes(harness, 0)

    def test_parallel_reservations_do_not_overcommit(self):
        admission = connections.HttpAdmission(1000)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            admitted = list(pool.map(lambda _: admission.acquire(100), range(100)))
        self.assertEqual(sum(admitted), 10)
        self.assertEqual(admission.active, 1000)
        for _ in range(10):
            admission.release(100)
        self.assertEqual(admission.active, 0)
