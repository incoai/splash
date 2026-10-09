import json
import unittest
from unittest import mock

from dev.tests.server_fixtures import (
    FOREVER,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    Plan,
    anthropic_body,
    chat_body,
    make_frontend,
    response_events,
    responses_body,
)
from server import backend as backend_api
from server import server as api


class OutputBudgetTests(HarnessTestCase):
    def test_no_server_option_bounds_a_request_without_a_limit(self):
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(
            FakeRuntime(), tokenizer, lambda _record: None
        )
        self.addCleanup(backend.close)
        # No server option bounds the output of a request that names no
        # limit: it may use what the two-token prompt leaves of the window.
        app = make_frontend(tokenizer, backend, "test-model", 40000, 1, 2, vision=True)
        self.assertEqual(
            app.prepare(chat_body(), deadline=FOREVER).max_new_tokens, 39998
        )

    def test_context_window_rejects_output_budget_without_truncating(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, max_context=10)
        # Each error names the field its client sent and gives the counts.
        cases = (
            (
                "/v1/chat/completions",
                chat_body(max_completion_tokens=9),
                "max_completion_tokens",
            ),
            ("/v1/chat/completions", chat_body(max_tokens=9), "max_tokens"),
            (
                "/v1/responses",
                responses_body(max_output_tokens=9),
                "max_output_tokens",
            ),
        )
        for path, body, field in cases:
            with self.subTest(path=path, body=body):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 400, payload)
                self.assertRegex(
                    json.loads(payload)["error"]["message"],
                    rf"^prompt and {field} exceed the context window: "
                    r"[1-9]\d* \+ 9 > 10 tokens$",
                )
                error = json.loads(payload)["error"]
                self.assertEqual(error["type"], "invalid_request_error")
                if path != "/v1/messages":
                    self.assertEqual(error["code"], "context_length_exceeded")
        self.assertEqual(runtime.requests, [])

    def test_output_budget_errors_name_the_field_the_client_sent(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for path, body, field in (
            ("/v1/chat/completions", chat_body(max_tokens=0), "max_tokens"),
            (
                "/v1/chat/completions",
                chat_body(max_completion_tokens=0, max_tokens=5),
                "max_completion_tokens",
            ),
            (
                "/v1/responses",
                responses_body(max_output_tokens=0),
                "max_output_tokens",
            ),
        ):
            with self.subTest(path=path, field=field):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 400, payload)
                self.assertEqual(
                    json.loads(payload)["error"]["message"],
                    f"{field} must be a positive integer",
                )
        self.assertEqual(runtime.requests, [])

    def test_anthropic_output_budget_is_clamped_to_the_remaining_window(self):
        # Claude Code sends max_tokens 32K on every turn and does not compact
        # for it; the request must proceed with what the window allows.
        runtime = FakeRuntime()
        harness = self.harness(runtime, max_context=10)
        status, _, payload = harness.request(
            "POST", "/v1/messages", anthropic_body(max_tokens=9)
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(runtime.requests), 1)
        request = runtime.requests[0].frame
        self.assertEqual(
            len(request.prompt_tokens) + request.logical_max_output_tokens, 10
        )
        self.assertGreaterEqual(request.logical_max_output_tokens, 1)

    def test_anthropic_reports_the_context_window_when_it_ends_the_response(self):
        # The prompt is 2 tokens of a 10-token window: max_tokens 9 is
        # lowered to 8, 8 fits exactly and 7 leaves room.
        cases = [
            (max_tokens, stream, expected)
            for max_tokens, expected in (
                (9, "model_context_window_exceeded"),
                (8, "max_tokens"),
                (7, "max_tokens"),
            )
            for stream in (False, True)
        ]
        harness = self.harness(
            FakeRuntime(*(Plan([[4]], reason="length") for _ in cases)),
            max_context=10,
        )
        for max_tokens, stream, expected in cases:
            with self.subTest(max_tokens=max_tokens, stream=stream):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/messages",
                    anthropic_body(max_tokens=max_tokens, stream=stream),
                )
                self.assertEqual(status, 200, payload)
                if stream:
                    events = response_events(payload)
                    self.assertEqual(events[-2]["type"], "message_delta")
                    stop_reason = events[-2]["delta"]["stop_reason"]
                else:
                    stop_reason = json.loads(payload)["stop_reason"]
                self.assertEqual(stop_reason, expected)

    def test_default_output_budget_uses_remaining_context(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, max_context=10)
        for path, body in (
            ("/v1/chat/completions", chat_body()),
            ("/v1/responses", responses_body()),
        ):
            with self.subTest(path=path):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 200, payload)
                self.assertEqual(
                    runtime.requests[-1].frame.logical_max_output_tokens, 8
                )
                self.assertEqual(runtime.requests[-1].frame.prompt_tokens, (101, 102))

        # The window, not a server default, bounds a request that names no
        # limit; one that names a limit inside the window gets it as is.
        harness.app.max_context = 262144
        for path, body, expected in (
            ("/v1/chat/completions", chat_body(), 262142),
            ("/v1/responses", responses_body(), 262142),
            ("/v1/chat/completions", chat_body(max_tokens=131072), 131072),
        ):
            with self.subTest(path=path, body=body):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 200, payload)
                self.assertEqual(
                    runtime.requests[-1].frame.logical_max_output_tokens, expected
                )

        harness.app.max_context = 100000
        for length, expected in ((90000, 10000), (99999, 1)):
            with mock.patch.object(
                FakeTokenizer, "__call__", return_value={"input_ids": [101] * length}
            ):
                job = harness.app.prepare(chat_body(), deadline=FOREVER)
            self.assertEqual(job.max_new_tokens, expected)
            self.assertEqual(len(job.prompt_tokens), length)
        with mock.patch.object(
            FakeTokenizer, "__call__", return_value={"input_ids": [101] * 100000}
        ):
            with self.assertRaisesRegex(api.APIError, "prompt exceeds") as caught:
                harness.app.prepare(chat_body(), deadline=FOREVER)
            self.assertEqual(caught.exception.code, "context_length_exceeded")


if __name__ == "__main__":
    unittest.main()
