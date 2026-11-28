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
    console_written,
    response_events,
    responses_body,
)
from server import protocol as wire
from server import server as api
from server.dialects import responses as responses_dialect
from server.dialects import scoring


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
        with mock.patch.object(api.FrontendHandler, "sse", side_effect=TimeoutError):
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

    def test_a_disconnect_while_a_stream_writes_nothing_cancels(self):
        # The output projector holds an integer argument until its close, so
        # the stream has nothing to write that could fail; the request still
        # finds the client gone and cancels.
        plan = Plan([[13]] + [[14]] * 400, delay=0.01)
        runtime = FakeRuntime(plan)
        harness = self.harness(runtime, timeout=30)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "integer"}},
                    },
                },
            }
        ]
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=5
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(chat_body(stream=True, reasoning_effort="none", tools=tools)),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        response.readline()
        self.assertTrue(plan.started.wait(2))
        closed_at = time.monotonic()
        response.close()
        connection.close()
        while runtime.cancel_count == 0 and time.monotonic() < closed_at + 3:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)

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

    def test_capacity_failure_is_a_context_overflow_in_each_api(self):
        # A request that does not fit even alone is answered as each API
        # answers a prompt over the context, so that agents compact and
        # retry: its tokens, the prompt tokens memory holds it now where the
        # engine says so, and what helps; never --max-context, which sizes no
        # memory a lone request lacks. The console keeps the engine's
        # sentence, beside the same advice.
        native = (
            "could not allocate KV target: engine memory budget exceeded "
            "(additional_pages=12, free_pages=0); the request, a Resident lane "
            "of 2 tokens in prefill, does not fit in the memory this server may "
            "use beside device_untracked_bytes=0 of Metal memory outside the "
            "server's buffers, even after every cached prefix was evicted and "
            "any store of streamed experts shrank to its floor"
        )
        records = []
        capacity = [
            Plan(
                exception=api.engine_runtime.RequestFailed(
                    1,
                    b"capacity_exhausted",
                    native.encode(),
                    capacity_limit=wire.CapacityLimit.MEMORY,
                    fits_tokens=1,
                )
            )
            for _ in range(2)
        ]
        harness = self.harness(
            FakeRuntime(*capacity, Plan([[4]])), request_logger=records.append
        )
        bodies = {}
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
            # Retrying the same request fails the same way: no Retry-After.
            self.assertEqual(response.status, 400)
            self.assertIsNone(response.getheader("Retry-After"))
            bodies[path] = json.loads(response.read())
        tokens = [r["prompt_tokens"] for r in records if r["outcome"] == "error"]
        remedy = (
            "shorten the conversation, or restart the server with a larger "
            "--max-memory if one was set below this Mac's memory"
        )
        chat = bodies["/v1/chat/completions"]["error"]
        self.assertEqual(chat["type"], "invalid_request_error")
        self.assertEqual(chat["code"], "context_length_exceeded")
        self.assertEqual(
            chat["message"],
            f"prompt does not fit in this server's memory: {tokens[0]} tokens > "
            f"1 tokens it holds for this request now; {remedy}",
        )
        messages = bodies["/v1/messages"]
        self.assertEqual(messages["type"], "error")
        self.assertEqual(messages["error"]["type"], "invalid_request_error")
        self.assertEqual(
            messages["error"]["message"],
            f"prompt is too long: {tokens[1]} tokens > 1 maximum input tokens "
            f"this server's memory holds for it now; {remedy}",
        )
        for body in (chat["message"], messages["error"]["message"]):
            self.assertNotIn("--max-context", body)
            self.assertNotIn("additional_pages", body)
        status, _, _ = harness.request("POST", "/v1/chat/completions", chat_body())
        self.assertEqual(status, 200)
        # The console line names the engine's code, its reason and the advice
        # its client got.
        failed = [record for record in records if record["outcome"] == "error"]
        self.assertEqual(
            [record["error_code"] for record in failed], ["capacity_exhausted"] * 2
        )
        self.assertEqual([record["engine_reason"] for record in failed], [native] * 2)
        self.assertEqual([record["remedy"] for record in failed], [remedy] * 2)

    def test_capacity_failure_names_counts_only_where_they_say_why(self):
        # Where the engine cannot say what memory holds the request, or its
        # prompt fit and its output outgrew what it holds, the message names
        # its tokens alone: each limit with what helps it.
        cases = (
            (wire.CapacityLimit.MEMORY, 0, "this server's memory", "--max-memory"),
            (wire.CapacityLimit.MEMORY, 4080, "this server's memory", "--max-memory"),
            (
                wire.CapacityLimit.CLOSED_DISK,
                0,
                "this server's memory",
                "reopens the SSD cache that stopped taking writes",
            ),
        )
        for limit, fits, where, remedy in cases:
            with self.subTest(limit=limit.name, fits=fits):
                failure = api.engine_runtime.RequestFailed(
                    1,
                    b"capacity_exhausted",
                    b"does not fit",
                    capacity_limit=limit,
                    fits_tokens=fits,
                )
                harness = self.harness(
                    FakeRuntime(Plan(exception=failure), Plan(exception=failure))
                )
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body()
                )
                self.assertEqual(status, 400)
                self.assertRegex(
                    json.loads(payload)["error"]["message"],
                    rf"^prompt does not fit in {where}: 2 tokens; .*{remedy}",
                )
                status, _, payload = harness.request(
                    "POST", "/v1/messages", anthropic_body()
                )
                self.assertEqual(status, 400)
                self.assertRegex(
                    json.loads(payload)["error"]["message"],
                    rf"^prompt is too long: 2 tokens do not fit in {where} now; "
                    rf".*{remedy}",
                )

    def test_disk_quota_failure_is_a_context_overflow_naming_the_ssd_cache(self):
        native = (
            "this request's KV needs needed_bytes=1703936 of the SSD cache, which may "
            "hold active_limit_bytes=425984 of active KV now (--max-cache-disk and the "
            "volume's free space)"
        )
        runtime = FakeRuntime(
            Plan(
                exception=api.engine_runtime.RequestFailed(
                    1,
                    b"kv_disk_quota",
                    native.encode(),
                    capacity_limit=wire.CapacityLimit.DISK_QUOTA,
                    fits_tokens=1,
                )
            ),
            Plan([[4]]),
        )
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body()
        )
        error = json.loads(payload)["error"]
        # Retrying fails the same way while memory and the quota stay as they are.
        self.assertEqual(status, 400)
        self.assertEqual(error["code"], "context_length_exceeded")
        self.assertEqual(
            error["message"],
            "prompt does not fit in the SSD cache: 2 tokens > 1 tokens it holds "
            "for this request now; shorten the conversation, or restart the "
            "server with a larger --max-cache-disk",
        )
        status, _, _ = harness.request("POST", "/v1/chat/completions", chat_body())
        self.assertEqual(status, 200)

    def test_write_budget_refusal_says_when_to_retry(self):
        # The SSD cache's daily write budget covers what spilling requests
        # were promised again in 6 hours, not in a second.
        runtime = FakeRuntime(
            Plan(
                exception=api.engine_runtime.RequestFailed(
                    1,
                    b"kv_write_budget",
                    b"the SSD cache's daily write budget is spent",
                    retryable=True,
                    retry_after_seconds=21_600,
                )
            ),
            Plan([[4]]),
        )
        harness = self.harness(runtime)
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
        self.assertEqual(response.getheader("Retry-After"), "21600")
        self.assertEqual(
            json.loads(response.read())["error"]["code"], "kv_write_budget"
        )
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

    def test_a_failure_the_backend_logged_has_one_console_line(self):
        """A request the engine took ends in one line, the backend's, which
        names its id; the HTTP handler does not add an id-less one."""
        for stream in (False, True):
            with self.subTest(stream=stream):
                records = []
                harness = self.harness(
                    FakeRuntime(Plan(exception=RuntimeError("boom"))),
                    request_logger=records.append,
                )
                with mock.patch.object(api, "print_status") as console:
                    status, _, _ = harness.request(
                        "POST", "/v1/chat/completions", chat_body(stream=stream)
                    )
                self.assertEqual(status, 200 if stream else 500)
                (end,) = [r for r in records if r["outcome"] != "started"]
                self.assertEqual(
                    (end["outcome"], end["error_code"], end["request_id"]),
                    ("error", "runtime_error", 1),
                )
                console.assert_not_called()

    def test_a_deadline_after_the_engine_took_the_request_ends_its_record(self):
        """The handler's deadline of a request the engine took ends the
        request's record, the one line that names it by the engine's id;
        the handler prints none of its own."""
        records = []
        blocking = Plan([[4]], block=True)
        harness = self.harness(FakeRuntime(blocking), request_logger=records.append)
        with mock.patch.object(api, "print_status") as console:
            status, _, _ = harness.request(
                "POST", "/v1/chat/completions", chat_body(timeout=0.03)
            )
        blocking.release.set()
        self.assertEqual(status, 504)
        console.assert_not_called()
        deadline = time.monotonic() + 5
        while not records or records[-1]["outcome"] == "started":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertEqual(
            (
                records[-1]["outcome"],
                records[-1]["error_code"],
                records[-1]["request_id"],
            ),
            ("error", "request_timeout", 1),
        )

    def test_unexpected_handler_error_cancels_and_is_structured(self):
        harness = self.harness(FakeRuntime(Plan([[4]], block=True)))
        stderr = io.StringIO()
        with (
            mock.patch.object(
                harness.backend, "cancel", wraps=harness.backend.cancel
            ) as cancel,
            mock.patch.object(
                api.FrontendHandler, "collect", side_effect=RuntimeError("boom")
            ),
            mock.patch.object(api.sys, "stderr", stderr),
        ):
            status, _, payload = harness.request(
                "POST", "/v1/chat/completions", chat_body()
            )
        self.assertEqual(status, 500, payload)
        self.assertEqual(json.loads(payload)["error"]["code"], "internal_server_error")
        cancel.assert_called_once()
        console_written()
        # Where the server, in a subpackage too, last had the error, and not
        # its message.
        self.assertRegex(
            stderr.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · internal_server_error · RuntimeError · "
            r"server/dialects/openai\.py:\d+\n$",
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
                ("/v1/systemone", systemone, "scoring", "prepare_systemone"),
                ("/v1/chat/completions", chat_body(), "handler", "collect"),
            ):
                with self.subTest(error=type(error).__name__, path=path, name=name):
                    harness = self.harness(FakeRuntime(Plan([[4]], block=True)))
                    owner = {
                        "app": harness.app,
                        "scoring": scoring,
                        "handler": api.FrontendHandler,
                    }[target]
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
            mock.patch.object(
                responses_dialect, "output_item", side_effect=RuntimeError("boom")
            ),
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
        self.assertTrue(plan.ended.is_set())
        console_written()
        self.assertRegex(
            stderr.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · internal_server_error · RuntimeError · "
            r"server/dialects/responses\.py:\d+\n$",
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
