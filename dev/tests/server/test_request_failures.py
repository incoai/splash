import http.client
import io
import json
import socket
import struct
import threading
import time
import unittest
from unittest import mock

from dev.tests.server_fixtures import (
    FOREVER,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    Plan,
    anthropic_body,
    byte_backend,
    chat_body,
    response_events,
    responses_body,
)
from server import server as api


class EndFailTokenizer(FakeTokenizer):
    def __init__(self):
        super().__init__()
        self.fragments[6] = "tail"
        self.backend_tokenizer = byte_backend(self.fragments)
        self.tail_decodes = 0

    def decode(self, token_ids, **kwargs):
        if token_ids == [6]:
            self.tail_decodes += 1
            if self.tail_decodes == 1:
                raise RuntimeError("decode failed during end")
        return super().decode(token_ids, **kwargs)


class RenderFailTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if any(
            isinstance(message.get("content"), str) and "\ud800" in message["content"]
            for message in messages
        ):
            raise TypeError("invalid surrogate")
        return super().apply_chat_template(messages, **kwargs)


class RequestFailureTests(HarnessTestCase):
    def test_sse_write_timeout_cancels_the_submitted_job(self):
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking)
        harness = self.harness(runtime)
        with mock.patch.object(api.FrontendHandler, "_sse", side_effect=TimeoutError):
            status, _, _ = harness.request(
                "POST",
                "/v1/chat/completions",
                chat_body(stream=True, reasoning_effort="none"),
            )
        self.assertEqual(status, 200)
        deadline = time.monotonic() + 1
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)

    def test_template_render_error_is_structured_and_server_recovers(self):
        harness = self.harness(FakeRuntime(), tokenizer=RenderFailTokenizer())
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(messages=[{"role": "user", "content": "\ud800"}]),
        )
        self.assertEqual(status, 400)
        self.assertEqual(
            json.loads(payload)["error"]["message"], "messages could not be rendered"
        )
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)

    def test_very_large_timeout_does_not_overflow_wait(self):
        blocking = Plan([[4]], block=True)
        harness = self.harness(FakeRuntime(blocking))
        release = threading.Timer(0.03, blocking.release.set)
        release.start()
        self.addCleanup(release.cancel)
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(timeout=1e308)
        )
        self.assertEqual(status, 200)

    def test_pending_limit_returns_retryable_http_overload(self):
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking)
        harness = self.harness(runtime, queue_size=1)
        head = harness.app.prepare(chat_body(timeout=2), deadline=FOREVER)
        harness.backend.submit(head)
        self.assertTrue(blocking.started.wait(1))

        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        self.addCleanup(connection.close)
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(chat_body()),
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
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking, Plan([[4]]))
        harness = self.harness(runtime)
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(timeout=0.03)
        )
        self.assertEqual(status, 504)
        self.assertEqual(runtime.cancel_count, 1)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )

    def test_http_stream_graceful_disconnect_cancels_and_next_request_runs(self):
        streaming = Plan([[4]] * 100, delay=0.02)
        runtime = FakeRuntime(streaming, Plan([[4]]))
        harness = self.harness(runtime, timeout=10)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=2
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(chat_body(stream=True, reasoning_effort="none")),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        response.readline()
        response.close()
        connection.close()
        deadline = time.monotonic() + 2
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )

    def test_http_nonstream_disconnect_cancels_and_next_request_runs(self):
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking, Plan([[4]]))
        harness = self.harness(runtime, timeout=10)
        connection = http.client.HTTPConnection(*harness.server.server_address)
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(chat_body(reasoning_effort="none")),
            {"Content-Type": "application/json"},
        )
        self.assertTrue(blocking.started.wait(1))
        connection.close()
        deadline = time.monotonic() + 2
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )

    def test_stream_disconnect_before_headers_cancels(self):
        waiting = Plan([[4]], block=True, before_start=True)
        runtime = FakeRuntime(waiting)
        harness = self.harness(runtime, timeout=10)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=2
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(chat_body(stream=True, reasoning_effort="none")),
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

    def test_runtime_error_does_not_poison_next_request(self):
        runtime = FakeRuntime(Plan(error="broken"), Plan([[4]]))
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body()
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["code"], "constraint_error")
        status, _, _ = harness.request("POST", "/v1/chat/completions", chat_body())
        self.assertEqual(status, 200)

    def test_capacity_failure_is_a_bad_request_naming_the_limits(self):
        native = (
            "could not allocate KV target: engine memory budget exceeded "
            "(additional_pages=12, free_pages=0)"
        )
        capacity = [
            Plan(
                exception=api.engine_runtime.RequestFailed(
                    1, b"capacity_exhausted", native.encode(), retryable=False
                )
            )
            for _ in range(2)
        ]
        harness = self.harness(FakeRuntime(*capacity, Plan([[4]])))
        # Retrying fails the same way, so no response invites a retry.
        for path, body in (
            ("/v1/chat/completions", chat_body()),
            ("/v1/messages", anthropic_body()),
        ):
            connection = http.client.HTTPConnection(
                *harness.server.server_address, timeout=3
            )
            self.addCleanup(connection.close)
            connection.request(
                "POST", path, json.dumps(body), {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            error = json.loads(response.read())["error"]
            self.assertEqual(response.status, 400)
            self.assertIsNone(response.getheader("Retry-After"))
            self.assertEqual(error["type"], "invalid_request_error")
            for text in ("--max-memory", "--max-context", native):
                self.assertIn(text, error["message"])
            if path == "/v1/chat/completions":
                self.assertEqual(error["code"], "capacity_exhausted")
        status, _, _ = harness.request("POST", "/v1/chat/completions", chat_body())
        self.assertEqual(status, 200)

    def test_unexpected_runtime_error_does_not_kill_backend(self):
        runtime = FakeRuntime(
            Plan(exception=RuntimeError("boom")),
            Plan([[4]]),
        )
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body()
        )
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(payload)["error"]["code"], "runtime_error")
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(reasoning_effort="none"),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )

    def test_unexpected_handler_error_cancels_and_is_structured(self):
        harness = self.harness(FakeRuntime(Plan([[4]], block=True)))
        stderr = io.StringIO()
        with (
            mock.patch.object(
                harness.backend, "cancel", wraps=harness.backend.cancel
            ) as cancel,
            mock.patch.object(
                api.FrontendHandler, "_complete", side_effect=RuntimeError("boom")
            ),
            mock.patch.object(api.sys, "stderr", stderr),
        ):
            status, _, payload = harness.request(
                "POST", "/v1/chat/completions", chat_body()
            )
        self.assertEqual(status, 500, payload)
        self.assertEqual(json.loads(payload)["error"]["code"], "internal_server_error")
        cancel.assert_called_once()
        # Where the server last had the error, and not its message.
        self.assertRegex(
            stderr.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · internal_server_error · RuntimeError · "
            r"server/server\.py:\d+\n$",
        )
        self.assertNotIn("boom", stderr.getvalue())

    def test_value_errors_past_the_body_are_internal_errors(self):
        # Only a body the server cannot read is invalid JSON. A ValueError or
        # RecursionError raised while a request is prepared or answered is
        # the server's own: logged, answered with 500, and the job it
        # submitted is cancelled.
        systemone = {
            "model": "test-model",
            "state": "evidence",
            "questions": {"supported": {"type": "noul"}},
        }
        for error in (ValueError("late"), RecursionError()):
            for path, body, target, name in (
                ("/v1/chat/completions", chat_body(), "app", "prepare"),
                ("/v1/systemone", systemone, "app", "prepare_systemone"),
                ("/v1/chat/completions", chat_body(), "handler", "_complete"),
            ):
                with self.subTest(error=type(error).__name__, path=path, name=name):
                    harness = self.harness(FakeRuntime(Plan([[4]], block=True)))
                    owner = harness.app if target == "app" else api.FrontendHandler
                    with (
                        mock.patch.object(owner, name, side_effect=error),
                        mock.patch.object(
                            harness.backend, "cancel", wraps=harness.backend.cancel
                        ) as cancel,
                        mock.patch.object(api, "log_unexpected") as logged,
                    ):
                        status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 500, payload)
                    self.assertEqual(
                        json.loads(payload)["error"]["code"], "internal_server_error"
                    )
                    logged.assert_called_once_with(error)
                    if target == "handler":
                        cancel.assert_called_once()
                    else:
                        cancel.assert_not_called()

    def test_unexpected_responses_stream_error_is_failed_and_cancels(self):
        # The first output item fails to render while the model still writes.
        plan = Plan([[14], [15]], delay=1)
        harness = self.harness(FakeRuntime(plan))
        stderr = io.StringIO()
        with (
            mock.patch.object(
                harness.backend, "cancel", wraps=harness.backend.cancel
            ) as cancel,
            mock.patch.object(api, "responses_item", side_effect=RuntimeError("boom")),
            mock.patch.object(api.sys, "stderr", stderr),
        ):
            status, _, payload = harness.request(
                "POST",
                "/v1/responses",
                responses_body(stream=True, reasoning={"effort": "none"}),
            )
        self.assertEqual(status, 200)
        events = response_events(payload)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(
            events[-1]["response"]["error"]["code"], "internal_server_error"
        )
        cancel.assert_called_once()
        self.assertTrue(plan.cancelled.is_set())
        self.assertRegex(
            stderr.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · internal_server_error · RuntimeError · "
            r"server/server\.py:\d+\n$",
        )

    def test_streamer_end_error_does_not_kill_backend(self):
        runtime = FakeRuntime(Plan([[6]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=EndFailTokenizer())
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body()
        )
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(payload)["error"]["code"], "runtime_error")
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(reasoning_effort="none"),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )


if __name__ == "__main__":
    unittest.main()
