import argparse
import base64
import concurrent.futures
import gc
import http.client
import io
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from dev.tests import test_server as fixtures
from install import launcher
from server import frontend
from server import server as api


class HttpBodyBudgetTests(unittest.TestCase):
    def harness(self, **kwargs):
        runtime = kwargs.pop("runtime", fixtures.FakeRuntime())
        harness = fixtures.Harness(runtime, **kwargs)
        self.addCleanup(harness.close)
        return harness

    def wait_bytes(self, harness, amount):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if harness.server.request_bodies.stats()["active"] == amount:
                return
            time.sleep(0.005)
        self.assertEqual(harness.server.request_bodies.stats()["active"], amount)

    def headers(self, harness, length, path="/v1/chat/completions"):
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=2
        )
        self.addCleanup(connection.close)
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(length))
        connection.endheaders()
        return connection

    def test_config_and_exact_body_boundary(self):
        for parse in (launcher._parse_request_size, api._parse_request_size):
            for value in ("128M", "128MB", "128MiB", "134217728"):
                self.assertEqual(parse(value), 128 * 1024**2)
            for value in ("auto", "0", "-1", "bad", str(2**64)):
                with (
                    self.subTest(value=value),
                    self.assertRaises(argparse.ArgumentTypeError),
                ):
                    parse(value)
        harness = self.harness(max_request_bytes=128)
        body = json.dumps({"content": "hello"}).encode().ljust(128)
        connection = self.headers(harness, len(body), "/tokenize")
        connection.send(body)
        response = connection.getresponse()
        self.assertEqual(response.status, 200, response.read())
        response.read()
        connection.close()
        self.wait_bytes(harness, 0)
        for path in (
            "/v1/chat/completions",
            "/v1/messages?beta=true",
            "/v1/responses",
            "/tokenize",
            "/apply-template",
            "/v1/messages/count_tokens?beta=true",
        ):
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
            handler = object.__new__(api.FrontendHandler)
            handler.headers = http.client.HTTPMessage()
            handler.headers["Content-Type"] = "application/json"
            handler.headers["Content-Length"] = str(len(payload))
            handler.server = SimpleNamespace(
                max_request_bytes=4096,
                request_bodies=api.HttpAdmission(4096),
                io_timeout=1,
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
                    self.assertEqual(parse(payload), api.strict_json_loads(payload))
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
                    api.strict_json_loads(payload)
                with self.assertRaises(ValueError):
                    parse(payload)

    def test_shared_budget_rejects_before_reading_and_recovers_on_disconnect(self):
        with mock.patch.object(api, "DEFAULT_REQUEST_BODY_BUDGET", 256):
            harness = self.harness(max_request_bytes=128, queue_size=16)
        uploads = [self.headers(harness, 128) for _ in range(2)]
        self.wait_bytes(harness, 256)
        for path in (
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/responses",
            "/tokenize",
            "/apply-template",
            "/v1/messages/count_tokens",
        ):
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
            harness.request("POST", "/tokenize", {"content": "hello"})[0], 200
        )
        self.wait_bytes(harness, 0)

    def test_parsed_bodies_stay_charged_while_preparation_is_pending(self):
        with mock.patch.object(api, "DEFAULT_REQUEST_BODY_BUDGET", 4096):
            harness = self.harness(max_request_bytes=1024, queue_size=8)
        body = {"messages": [{"role": "user", "content": "hello"}], "extra": "x" * 850}
        size = len(json.dumps(body).encode())
        release = threading.Event()
        prepare_lock = threading.Lock()
        prepare = harness.app.prepare

        def paused(*args, **kwargs):
            if not release.wait(3):
                raise TimeoutError("preparation test timed out")
            with prepare_lock:
                return prepare(*args, **kwargs)

        with mock.patch.object(harness.app, "prepare", side_effect=paused):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                calls = [
                    pool.submit(harness.request, "POST", "/v1/chat/completions", body)
                    for _ in range(4)
                ]
                try:
                    self.wait_bytes(harness, 4 * size)
                    self.assertEqual(
                        harness.request("POST", "/tokenize", {"content": "small"})[0],
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
        body = {"messages": [{"role": "user", "content": "hello"}]}
        for error, status in (
            (api.APIError(400, "bad input"), 400),
            (RuntimeError("test"), 500),
        ):
            with (
                mock.patch.object(harness.app, "prepare", side_effect=error),
                mock.patch.object(api, "log_unexpected"),
            ):
                self.assertEqual(
                    harness.request("POST", "/v1/chat/completions", body)[0], status
                )
            self.wait_bytes(harness, 0)
        with mock.patch.object(harness.backend, "submit", return_value=False):
            self.assertEqual(
                harness.request("POST", "/v1/chat/completions", body)[0], 429
            )
        self.wait_bytes(harness, 0)

    def test_text_upload_releases_before_decode_finishes(self):
        plan = fixtures.Plan([[4]], block=True)
        harness = self.harness(runtime=fixtures.FakeRuntime(plan))
        body = {"messages": [{"role": "user", "content": "hello"}], "stream": True}
        connection, response = harness.open_stream("/v1/chat/completions", body)
        self.addCleanup(connection.close)
        self.addCleanup(response.close)
        self.assertTrue(plan.started.wait(1))
        self.wait_bytes(harness, 0)
        self.assertEqual(harness.server.requests.stats()["active"], 1)
        plan.release.set()
        response.read()

    def test_retained_inputs_stay_charged_until_job_is_released(self):
        admission = api.HttpAdmission(1024)
        job = SimpleJob()
        job.response_history_items = [{"type": "message", "content": "x" * 200}]
        reservation = api.RequestBodyReservation(admission, 900)
        reservation.retain_for(job)
        retained_bytes = admission.active
        self.assertGreater(retained_bytes, 200)
        self.assertLess(retained_bytes, 900)
        del reservation
        self.assertEqual(admission.active, retained_bytes)
        del job
        self.assertEqual(admission.active, 0)

    def test_responses_input_is_charged_through_stream_and_cancel(self):
        plan = fixtures.Plan([[4]], block=True)
        harness = self.harness(runtime=fixtures.FakeRuntime(plan))
        body = {
            "input": "retained history " * 1000,
            "stream": True,
            "reasoning": {"effort": "none"},
        }
        enabled = gc.isenabled()
        gc.disable()
        try:
            connection, response = harness.open_stream("/v1/responses", body)
            self.addCleanup(connection.close)
            self.addCleanup(response.close)
            self.assertTrue(plan.started.wait(1))
            self.assertGreater(harness.server.request_bodies.stats()["active"], 16000)
            response.close()
            connection.close()
            self.assertTrue(plan.cancelled.wait(1))
            for thread in harness.backend.runtime.threads:
                thread.join(1)
            # This fixture archives callbacks; the real transport drops them
            # when a call completes. Until then their input stays charged.
            self.assertGreater(harness.server.request_bodies.stats()["active"], 0)
            harness.backend.runtime.calls.clear()
            self.wait_bytes(harness, 0)
        finally:
            if enabled:
                gc.enable()

    def test_tool_and_response_schemas_remain_charged(self):
        admission = api.HttpAdmission(4096)
        job = SimpleJob()
        schema = {"type": "string", "description": "x" * 300}
        job.tool_policy = SimpleNamespace(
            schemas={"tool": schema}, namespaces={"tool": ("ns", "tool")}
        )
        job.response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
        reservation = api.RequestBodyReservation(admission, 2048)
        reservation.retain_for(job)
        self.assertGreater(admission.active, 600)
        self.assertLess(admission.active, 2048)
        del job
        self.assertEqual(admission.active, 0)

    def test_expanded_history_needs_additional_capacity(self):
        admission = api.HttpAdmission(256)
        job = SimpleJob()
        job.response_history_items = ["x" * 200]
        reservation = api.RequestBodyReservation(admission, 10)
        reservation.retain_for(job)
        self.assertGreater(admission.active, 200)
        blocked = api.RequestBodyReservation(admission, 1)
        other = SimpleJob()
        other.response_history_items = ["x" * 200]
        with self.assertRaises(api.APIError) as error:
            blocked.retain_for(other)
        self.assertEqual(error.exception.status, 503)
        blocked.release()
        del job
        self.assertEqual(admission.active, 0)

    def test_history_admission_precedes_decode_and_recovers(self):
        harness = self.harness()
        harness.server.request_bodies = api.HttpAdmission(1024)
        history = [{"role": "user", "content": "x" * 2048}]
        harness.app.response_store.put({"id": "resp_previous"}, history)
        record = harness.app.response_store.get("resp_previous")
        body = {"input": "continue", "previous_response_id": "resp_previous"}
        with mock.patch.object(frontend.json, "loads", wraps=json.loads) as decode:
            status, _, payload = harness.request("POST", "/v1/responses", body)
            self.assertEqual(status, 503, payload)
            self.assertFalse(
                any(
                    call.args[0] is record.history_json
                    for call in decode.call_args_list
                )
            )
        self.assertFalse(harness.backend.runtime.requests)
        self.wait_bytes(harness, 0)
        harness.server.request_bodies = api.HttpAdmission(4096)
        status, _, payload = harness.request("POST", "/v1/responses", body)
        self.assertEqual(status, 200, payload)
        self.assertIn("x" * 2048, str(harness.tokenizer.templates))
        harness.backend.runtime.calls.clear()
        self.wait_bytes(harness, 0)

    def test_store_false_uses_history_without_retaining_it(self):
        harness = self.harness()
        harness.app.response_store.put(
            {"id": "resp_previous"}, [{"role": "user", "content": "old history"}]
        )
        reservation = api.RequestBodyReservation(harness.server.request_bodies, 100)
        job, *_ = harness.app.prepare_responses(
            {
                "input": "continue",
                "previous_response_id": "resp_previous",
                "store": False,
            },
            reserve_input=reservation.grow,
        )
        self.assertIn("old history", str(harness.tokenizer.templates))
        self.assertIsNone(job.response_history_items)
        reservation.retain_for(job)
        self.wait_bytes(harness, 0)

    def test_history_reservation_releases_after_preparation_error(self):
        harness = self.harness()
        harness.app.response_store.put(
            {"id": "resp_previous"}, [{"role": "user", "content": "old history"}]
        )
        body = {"input": "continue", "previous_response_id": "resp_previous"}
        with mock.patch.object(
            harness.app, "_prepare", side_effect=api.APIError(400, "invalid input")
        ):
            self.assertEqual(harness.request("POST", "/v1/responses", body)[0], 400)
        self.wait_bytes(harness, 0)
        self.assertEqual(harness.app.preparation_active, 0)

    def test_upload_deadline_scales_with_body_size_and_caps_at_request_deadline(self):
        for length, request_seconds, expected in (
            (1024, 1800, 30 + 1 / 512),
            (128 * 1024**2, 1800, 286),
            (256 * 1024**2, 1800, 542),
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

                handler = object.__new__(api.FrontendHandler)
                handler.headers = http.client.HTTPMessage()
                handler.headers["Content-Type"] = "application/json"
                handler.headers["Content-Length"] = str(length)
                handler.server = SimpleNamespace(
                    max_request_bytes=length,
                    request_bodies=api.HttpAdmission(length),
                    io_timeout=30,
                )
                handler.connection = SimpleNamespace(settimeout=settimeout)
                handler.rfile = SimpleNamespace(read1=drip)
                try:
                    with (
                        mock.patch.object(
                            api.time, "monotonic", side_effect=lambda: now
                        ),
                        self.assertRaises(TimeoutError),
                    ):
                        handler._read_json_body(request_seconds)
                    self.assertAlmostEqual(now, expected)
                finally:
                    handler._body_reservation.release()
                self.assertEqual(handler.server.request_bodies.active, 0)

    @mock.patch.object(api, "HTTP_UPLOAD_BYTES_PER_SECOND", 20)
    def test_active_upload_can_outlast_idle_timeout(self):
        harness = self.harness(io_timeout=2, timeout=15)
        payload = json.dumps({"content": "hello"}).encode().ljust(100)
        connection = self.headers(harness, len(payload), "/tokenize")
        for offset in range(0, len(payload), 10):
            connection.send(payload[offset : offset + 10])
            time.sleep(0.25)
        response = connection.getresponse()
        self.assertEqual(response.status, 200, response.read())
        response.read()
        self.wait_bytes(harness, 0)

    @mock.patch.object(api, "HTTP_UPLOAD_BYTES_PER_SECOND", 256)
    def test_drip_uploads_expire_and_restore_shared_capacity(self):
        with mock.patch.object(api, "DEFAULT_REQUEST_BODY_BUDGET", 256):
            harness = self.harness(max_request_bytes=128, io_timeout=0.4, timeout=4)
        uploads = [self.headers(harness, 128, "/tokenize") for _ in range(2)]
        self.wait_bytes(harness, 256)
        response = self.headers(harness, 1, "/tokenize").getresponse()
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
            harness.request("POST", "/tokenize", {"content": "hello"})[0], 200
        )
        self.wait_bytes(harness, 0)

    def test_active_upload_still_obeys_request_deadline(self):
        harness = self.harness(io_timeout=1, timeout=0.3)
        connection = self.headers(harness, 100, "/tokenize")
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
        admission = api.HttpAdmission(1000)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            admitted = list(pool.map(lambda _: admission.acquire(100), range(100)))
        self.assertEqual(sum(admitted), 10)
        self.assertEqual(admission.active, 1000)
        for _ in range(10):
            admission.release(100)
        self.assertEqual(admission.active, 0)

    def test_large_multimodal_body_reaches_image_preparation_and_native_submission(
        self,
    ):
        harness = self.harness(
            tokenizer=fixtures.ServerTest.ImagePadTokenizer(),
            max_context=32768,
            timeout=30,
        )
        images = []
        for index in range(5):
            payload = io.BytesIO()
            Image.new("RGB", (1024, 1024), (index, 30, 90)).save(
                payload, format="PNG", compress_level=0
            )
            images.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(payload.getvalue()).decode()
                    },
                }
            )
        body = {"messages": [{"role": "user", "content": images}], "max_tokens": 8}
        self.assertGreater(len(json.dumps(body)), 16 * 1024**2)
        status, _, payload = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200, payload)
        request = harness.backend.runtime.requests[-1]
        self.assertEqual(len(request.image_spans), 5)
        self.wait_bytes(harness, 0)


class SimpleJob:
    tool_policy = None
    response_history_items = None
    response_format = None
    stop_sequences = ()
