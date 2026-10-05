import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.tests.server_fixtures import (
    FakeRuntime,
    HarnessTestCase,
    Plan,
    anthropic_body,
    chat_body,
    next_sse_data,
    responses_body,
)
from server import backend as backend_api
from server import protocol as native_wire
from server import server as api
from server import tool_schema


class StreamKeepaliveTests(HarnessTestCase):
    def test_stream_sends_keepalive_while_native_is_idle(self):
        plan = Plan([[4]], block=True)
        harness = self.harness(FakeRuntime(plan))
        with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
            connection, response = harness.open_stream(
                "/v1/chat/completions",
                chat_body(stream=True, reasoning_effort="none"),
            )
            try:
                role = json.loads(next_sse_data(response))
                self.assertEqual(role["choices"][0]["delta"]["role"], "assistant")
                heartbeat = json.loads(next_sse_data(response))
                self.assertEqual(heartbeat["choices"][0]["delta"], {})
                self.assertIsNone(heartbeat["choices"][0]["finish_reason"])
            finally:
                plan.release.set()
                response.read()
                connection.close()

    def test_stream_keepalive_is_a_comment_until_the_role_chunk(self):
        plan = Plan([[4]], before_start=True)
        harness = self.harness(FakeRuntime(plan))
        with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
            connection, response = harness.open_stream(
                "/v1/chat/completions",
                chat_body(stream=True, reasoning_effort="none"),
            )
            try:
                self.assertEqual(response.readline(), b": splash-keepalive\n")
                self.assertFalse(plan.started.is_set())
                plan.start_release.set()
                role = json.loads(next_sse_data(response))
                self.assertEqual(role["choices"][0]["delta"]["role"], "assistant")
            finally:
                plan.start_release.set()
                response.read()
                connection.close()

    def test_chat_and_text_streams_share_keepalive_and_usage_chunks(self):
        usage = {"stream": True, "stream_options": {"include_usage": True}}
        for path, body, field, empty in (
            (
                "/v1/chat/completions",
                chat_body(reasoning_effort="none", **usage),
                "delta",
                {},
            ),
            (
                "/v1/completions",
                {"model": "test-model", "prompt": "hello", **usage},
                "text",
                "",
            ),
        ):
            with self.subTest(path=path):
                plan = Plan([[4]], before_start=True, block=True)
                harness = self.harness(FakeRuntime(plan))
                with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
                    connection, response = harness.open_stream(path, body)
                    try:
                        # A comment until the request starts, then a chunk
                        # that adds nothing.
                        self.assertEqual(response.readline(), b": splash-keepalive\n")
                        plan.start_release.set()
                        chunk = json.loads(next_sse_data(response))
                        if path == "/v1/chat/completions":
                            self.assertEqual(
                                chunk["choices"][0]["delta"],
                                {"role": "assistant", "content": ""},
                            )
                            chunk = json.loads(next_sse_data(response))
                        self.assertEqual(chunk["choices"][0][field], empty)
                        self.assertIsNone(chunk["choices"][0]["finish_reason"])
                    finally:
                        plan.release.set()
                        payload = response.read()
                        connection.close()
                chunks = [
                    json.loads(line[6:])
                    for line in payload.decode().splitlines()
                    if line.startswith("data: {")
                ]
                finish, final = chunks[-2:]
                self.assertEqual(finish["choices"][0]["finish_reason"], "stop")
                self.assertIn("timings", finish)
                self.assertEqual(final["choices"], [])
                self.assertEqual(final["usage"]["completion_tokens"], 1)
                self.assertIn("metrics", final)
                self.assertTrue(payload.endswith(b"data: [DONE]\n\n"))

    def test_responses_stream_heartbeats_before_native_start(self):
        plan = Plan([[4]], before_start=True)
        harness = self.harness(FakeRuntime(plan))

        def next_event(response):
            payload = json.loads(next_sse_data(response))
            return payload["type"], payload

        with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
            connection, response = harness.open_stream(
                "/v1/responses",
                responses_body(stream=True, reasoning={"effort": "none"}),
            )
            try:
                created, _ = next_event(response)
                initial, _ = next_event(response)
                heartbeat, payload = next_event(response)
                self.assertEqual(created, "response.created")
                self.assertEqual(initial, "response.in_progress")
                self.assertEqual(heartbeat, "response.in_progress")
                self.assertEqual(payload["response"]["status"], "in_progress")
                self.assertFalse(plan.started.is_set())
            finally:
                plan.start_release.set()
                response.read()
                connection.close()

    def test_responses_stream_heartbeats_after_start_until_first_output(self):
        handler = object.__new__(api.FrontendHandler)
        handler.wfile = io.BytesIO()
        handler._last_sse_write = 0.0
        handler._response_started = False
        handler._client_disconnected = lambda: False
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        app = SimpleNamespace(
            response_model="test-model",
            backend=SimpleNamespace(cancel=mock.Mock()),
            persist_response=mock.Mock(),
        )
        handler.server = SimpleNamespace(app=app)
        job = backend_api.Job(
            request_id=1,
            prompt_tokens=[101, 102],
            max_new_tokens=16,
            seed=0,
            sampling=native_wire.SamplingParameters(),
            deadline=100,
            public_id="prefill-heartbeat",
        )
        clock = [0.0]
        events = iter(
            [
                ("start", None),
                None,
                None,
                ("text", "answer"),
                None,
                None,
                ("done", backend_api.NativeResult("stop", 2, 1, 1, 1, 1)),
            ]
        )

        def next_event(**_kwargs):
            event = next(events)
            if event is None:
                clock[0] += api.SSE_KEEPALIVE_SECONDS + 0.1
                raise api.queue.Empty
            return event

        job.events = SimpleNamespace(get=next_event)
        with mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]):
            handler._responses_stream(job)

        raw = handler.wfile.getvalue()
        stream = io.BytesIO(raw)
        parsed = []
        while data := next_sse_data(stream):
            parsed.append(json.loads(data))
        kinds = [event["type"] for event in parsed]
        first_output = kinds.index("response.output_item.added")
        self.assertEqual(
            kinds[:first_output],
            ["response.created", *(["response.in_progress"] * 3)],
        )
        for event in parsed[:first_output]:
            self.assertEqual(event["response"]["id"], "resp_prefill-heartbeat")
            self.assertEqual(event["response"]["status"], "in_progress")
            self.assertEqual(event["response"]["output"], [])
        self.assertNotIn("response.in_progress", kinds[first_output:])
        self.assertEqual(
            [event["sequence_number"] for event in parsed], list(range(len(parsed)))
        )
        before_output, _, after_output = raw.partition(
            b"event: response.output_item.added\n"
        )
        self.assertNotIn(b": splash-keepalive", before_output)
        self.assertEqual(after_output.count(b": splash-keepalive"), 2)
        self.assertEqual(kinds[-1], "response.completed")
        response = parsed[-1]["response"]
        self.assertEqual(response["id"], "resp_prefill-heartbeat")
        self.assertEqual(response["output"][0]["content"][0]["text"], "answer")
        app.backend.cancel.assert_not_called()
        app.persist_response.assert_called_once()

    def test_buffered_tool_generation_keeps_stream_alive(self):
        schema = {
            "type": "object",
            "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
            "required": ["questions"],
        }
        _, policy = tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "ask", "parameters": schema}}],
            "required",
            False,
        )
        handler = object.__new__(api.FrontendHandler)
        handler.wfile = io.BytesIO()
        handler._last_sse_write = 0.0
        handler._response_started = True
        handler._client_disconnected = lambda: False
        clock = [0.0]
        snapshots = []
        fragments = [
            '<tool_call>\n<function=ask>\n<parameter=questions>\n["',
            *("a" for _ in range(24)),
            '"]\n</parameter>\n</function>\n</tool_call>',
        ]
        events = iter(
            [
                *(("text", fragment) for fragment in fragments),
                ("done", backend_api.NativeResult("stop", 1, 26, 1, 1, 1)),
            ]
        )

        def next_event(**_kwargs):
            snapshots.append(handler.wfile.getvalue())
            clock[0] += 0.5
            return next(events)

        job = SimpleNamespace(
            public_id="request",
            thinking=False,
            tool_policy=policy,
            may_call_tools=True,
            response_validator=None,
            deadline=100,
            events=SimpleNamespace(get=next_event),
        )
        with mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]):
            calls = handler._collect(
                job,
                on_text=lambda field, text: handler._sse({field: text}),
                on_tool_delta=handler._sse,
                on_idle=handler._sse_keepalive,
            ).tool_calls
        # All native events were immediately available, but the JSON array
        # remained buffered across several heartbeat periods.
        self.assertGreaterEqual(snapshots[-3].count(b": splash-keepalive"), 3)
        self.assertNotIn(b"questions", snapshots[-3])
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"questions": ["a" * 24]}
        )

    def test_visible_stream_output_resets_keepalive_clock(self):
        handler = object.__new__(api.FrontendHandler)
        handler.wfile = io.BytesIO()
        handler._last_sse_write = 0.0
        handler._client_disconnected = lambda: False
        clock = [0.0]
        job = SimpleNamespace(
            deadline=100, events=SimpleNamespace(get=lambda **_: ("text", "x"))
        )
        heartbeat = mock.Mock(wraps=handler._sse_keepalive)
        with mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]):
            for index in range(12):
                clock[0] += 0.5
                handler._next_event(job, heartbeat)
                if index % 2:
                    handler._event_sse("event", {"text": "x"})
                else:
                    handler._sse({"text": "x"})
        heartbeat.assert_not_called()

    def test_anthropic_stream_pings_during_prefill_and_decode_waits(self):
        plan = Plan([[4]], before_start=True, block=True)
        harness = self.harness(FakeRuntime(plan))
        with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
            connection, response = harness.open_stream(
                "/v1/messages",
                anthropic_body(stream=True),
            )
            try:
                self.assertEqual(json.loads(next_sse_data(response)), {"type": "ping"})
                self.assertFalse(plan.started.is_set())
                plan.start_release.set()
                self.assertTrue(plan.started.wait(1))
                start = json.loads(next_sse_data(response))
                while start["type"] == "ping":
                    start = json.loads(next_sse_data(response))
                self.assertEqual(start["type"], "message_start")
                self.assertEqual(
                    start["message"]["usage"],
                    {
                        "input_tokens": 1,
                        "cache_read_input_tokens": 1,
                        "output_tokens": 0,
                    },
                )
                for _ in range(2):
                    self.assertEqual(
                        json.loads(next_sse_data(response)), {"type": "ping"}
                    )
                self.assertFalse(plan.release.is_set())
            finally:
                plan.start_release.set()
                plan.release.set()
                response.read()
                connection.close()


if __name__ == "__main__":
    unittest.main()
