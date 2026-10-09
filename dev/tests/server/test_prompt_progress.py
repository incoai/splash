import json
import unittest

from dev.tests.server_fixtures import FakeRuntime, Harness, Plan, request
from server import protocol as wire
from server.runtime import RuntimeCall


class ProgressRuntime(FakeRuntime):
    def submit(self, request, *, on_event, on_complete):
        def emit(call, event):
            on_event(call, event)
            if request.frame.flags & wire.RequestFlag.RETURN_PROGRESS and isinstance(
                event, wire.StartEvent
            ):
                counts = sorted(
                    {event.matched_prompt_tokens, len(request.frame.prompt_tokens)}
                )
                for index, count in enumerate(counts):
                    on_event(
                        call,
                        wire.PromptProgressEvent(call.request_id, count, index * 1000),
                    )

        return super().submit(request, on_event=emit, on_complete=on_complete)


def events(payload):
    return [
        json.loads(line[6:])
        for line in payload.splitlines()
        if line.startswith(b"data: ") and line != b"data: [DONE]"
    ]


class PromptProgressTests(unittest.TestCase):
    def test_all_streams_are_opt_in_and_preserve_completion(self):
        paths = (
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages?beta=true",
        )
        for path in paths:
            for enabled in (False, True):
                for cached in (0, 1):
                    with self.subTest(path=path, enabled=enabled, cached=cached):
                        runtime = ProgressRuntime(Plan([[4]], matched_tokens=cached))
                        harness = Harness(runtime)
                        try:
                            body = {
                                "model": "test-model",
                                "stream": True,
                                "return_progress": enabled,
                                "max_tokens": 8,
                                "reasoning_effort": "none",
                            }
                            if path == "/v1/responses":
                                body["input"] = "Hi"
                                body.pop("max_tokens")
                                body["max_output_tokens"] = 8
                            elif path == "/v1/completions":
                                body["prompt"] = "Hi"
                            else:
                                body["messages"] = [{"role": "user", "content": "Hi"}]
                            status, _, payload = harness.request("POST", path, body)
                            self.assertEqual(status, 200, payload)
                            stream = events(payload)
                            progress = [
                                event["prompt_progress"]
                                for event in stream
                                if "prompt_progress" in event
                            ]
                            self.assertEqual(
                                bool(
                                    runtime.requests[0].frame.flags
                                    & wire.RequestFlag.RETURN_PROGRESS
                                ),
                                enabled,
                            )
                            self.assertIn(b"plain answer", payload)
                            if not enabled:
                                self.assertEqual(progress, [])
                                continue
                            self.assertEqual(
                                progress,
                                [
                                    {
                                        "total": 2,
                                        "cache": cached,
                                        "processed": cached,
                                        "time_ms": 0.0,
                                    },
                                    {
                                        "total": 2,
                                        "cache": cached,
                                        "processed": 2,
                                        "time_ms": 1.0,
                                    },
                                ],
                            )
                            progress_events = [
                                event for event in stream if "prompt_progress" in event
                            ]
                            if path == "/v1/chat/completions":
                                self.assertTrue(
                                    all(
                                        event["choices"][0]["delta"] == {}
                                        for event in progress_events
                                    )
                                )
                            elif path == "/v1/completions":
                                self.assertTrue(
                                    all(
                                        event["choices"][0]["text"] == ""
                                        for event in progress_events
                                    )
                                )
                            else:
                                expected = (
                                    "response.in_progress"
                                    if path == "/v1/responses"
                                    else "ping"
                                )
                                self.assertTrue(
                                    all(
                                        event["type"] == expected
                                        for event in progress_events
                                    )
                                )
                            self.assertLess(
                                payload.rfind(b"prompt_progress"),
                                payload.find(b"plain answer"),
                            )
                        finally:
                            harness.close()

    def test_progress_arrives_before_generation_finishes(self):
        plan = Plan([[4]], block=True, matched_tokens=0)
        harness = Harness(ProgressRuntime(plan))
        self.addCleanup(harness.close)
        connection, response = harness.open_stream(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "return_progress": True,
                "reasoning_effort": "none",
            },
        )
        self.addCleanup(connection.close)
        try:
            self.assertEqual(response.status, 200)
            while True:
                line = response.readline()
                self.assertTrue(line)
                if b"prompt_progress" in line:
                    break
            self.assertFalse(plan.terminal.is_set())
        finally:
            plan.release.set()
        self.assertIn(b"[DONE]", response.read())

    def test_invalid_option_is_rejected_before_native_submission(self):
        runtime = ProgressRuntime()
        harness = Harness(runtime)
        self.addCleanup(harness.close)
        for path in (
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
        ):
            for extra in (
                {"return_progress": True},
                {"return_progress": True, "stream": False},
                {"return_progress": 1, "stream": True},
                {"return_progress": "true", "stream": True},
            ):
                with self.subTest(path=path, extra=extra):
                    body = {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "prompt": "Hi",
                        "input": "Hi",
                        "max_tokens": 8,
                        **extra,
                    }
                    status, _, _ = harness.request("POST", path, body)
                    self.assertEqual(status, 400)
        self.assertEqual(runtime.requests, [])

    def test_runtime_rejects_unsolicited_or_regressing_progress(self):
        def call(enabled=True):
            result = RuntimeCall(
                None, 1, 1, request(10, return_progress=enabled), None, None
            )
            result._record_start(wire.StartEvent(1, 0, 1))
            return result

        self.assertIsNotNone(
            call(False)._record_progress(wire.PromptProgressEvent(1, 1, 0))
        )
        for invalid in (
            wire.PromptProgressEvent(1, 0, 100),
            wire.PromptProgressEvent(1, 3, 100),
        ):
            self.assertIsNotNone(call()._record_progress(invalid))
        current = call()
        self.assertIsNone(current._record_progress(wire.PromptProgressEvent(1, 1, 100)))
        self.assertIsNotNone(
            current._record_progress(wire.PromptProgressEvent(1, 1, 101))
        )
        self.assertIsNotNone(
            current._record_progress(wire.PromptProgressEvent(1, 2, 99))
        )
        self.assertIsNone(current._record_progress(wire.PromptProgressEvent(1, 2, 101)))
        self.assertIsNone(current._record_tokens(wire.TokensEvent(1, 0, (42,))))
        self.assertIsNotNone(
            current._record_progress(wire.PromptProgressEvent(1, 2, 102))
        )
