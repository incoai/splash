import json
import unittest
from dataclasses import replace

from dev.tests.server_fixtures import (
    FakeRuntime,
    HarnessTestCase,
    Plan,
    chat_body,
    openai_client,
    responses_body,
)
from server.chat_format import QWEN


class ChatCompletionTests(HarnessTestCase):
    def test_nonstream_reasoning_usage_metrics_and_seed(self):
        runtime = FakeRuntime(Plan([[1], [2], [3]]))
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(seed=7)
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        message = response["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "because ")
        self.assertEqual(message["content"], "answer\n")
        self.assertEqual(
            response["usage"],
            {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 1},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        )
        self.assertEqual(response["metrics"]["prefill"], {"tokens": 1})
        self.assertEqual(response["metrics"]["decode"], {"tokens": 2})
        self.assertEqual(
            response["metrics"]["request_latency"],
            {
                "start_to_first_token_ms": 1.0,
                "first_token_to_done_ms": 2.0,
                "wall_ms": 3.0,
                "ttft_ms": 1.0,
                "queue_to_start_ms": 0.0,
                "stream_tokens_per_second": 1000.0,
            },
        )
        self.assertNotIn("tokens_per_second", response["metrics"])
        self.assertNotIn("speculative", response["metrics"])
        self.assertNotIn("batch", response["metrics"])
        self.assertEqual(
            response["metrics"]["cache"],
            {
                "status": "hit",
                "matched_tokens": 1,
                "lane": 0,
                "deduplicated_tokens": 0,
            },
        )
        self.assertNotIn("queue_ms", response["metrics"])
        self.assertEqual(runtime.requests[0].frame.seed, 7)

    def test_every_generation_api_sends_the_cache_salt(self):
        # Responses and Messages reach the engine through chat bodies of
        # their own, which carry the salt too.
        requests = (
            ("/v1/chat/completions", chat_body(cache_salt="tenant-a")),
            (
                "/v1/completions",
                {"model": "test-model", "prompt": "hi", "max_tokens": 1}
                | {"cache_salt": "tenant-a"},
            ),
            ("/v1/responses", responses_body(cache_salt="tenant-a")),
            (
                "/v1/messages",
                {
                    "model": "test-model",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                    "cache_salt": "tenant-a",
                },
            ),
            ("/v1/chat/completions", chat_body()),
        )
        runtime = FakeRuntime(
            *(Plan([[1]], reason="length") for _ in range(len(requests)))
        )
        harness = self.harness(runtime)
        for path, body in requests:
            status, _, payload = harness.request("POST", path, body)
            self.assertEqual(status, 200, (path, payload))
        *salted, unsalted = (request.frame.cache_salt for request in runtime.requests)
        # Persisted caches keep salted prompts under this value: changing the
        # hash orphans them.
        self.assertEqual(salted, [0xC9D55DE1D7B39AF63B3B58B43F8BDEBE] * 4)
        self.assertEqual(unsalted, 0)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(cache_salt="")
        )
        self.assertEqual(status, 400, payload)

    def test_streaming_sse_and_usage(self):
        harness = self.harness(FakeRuntime(Plan([[1], [2], [3]], reason="length")))
        status, content_type, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(stream=True, stream_options={"include_usage": True}),
        )
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/event-stream")
        events = [
            line[6:]
            for line in payload.decode().splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
        self.assertEqual(deltas[0], {"role": "assistant", "content": ""})
        self.assertIn({"reasoning_content": "because "}, deltas)
        self.assertIn({"content": "answer\n"}, deltas)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 3)
        self.assertAlmostEqual(
            chunks[-1]["metrics"]["request_latency"]["stream_tokens_per_second"],
            1000.0,
        )

    def test_stop_sequence_cancels_native_without_emitting_the_marker(self):
        runtime = FakeRuntime(Plan([[14], [15], [4]], reason="length", delay=0.01))
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(stop="second", reasoning_effort="none"),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], "first ")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"]["completion_tokens"], 2)
        self.assertEqual(runtime.cancel_count, 1)

    def test_usage_counts_only_tokens_before_the_thinking_delimiter(self):
        # The fake vocabulary's </think> is token 26.
        harness = self.harness(
            FakeRuntime(Plan([[1], [26], [27]])),
            chat_format=replace(QWEN, think_end_id=26),
        )
        with openai_client(harness) as client:
            response = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
            )
        self.assertEqual(response.choices[0].message.content, "answer\n")
        self.assertEqual(
            response.usage.completion_tokens_details.reasoning_tokens,
            1,
        )
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 1)

    def test_usage_counts_reasoning_tokens_before_a_call_from_the_reasoning(self):
        # Where a call may follow, its tag ends the reasoning, and a later
        # </think> is dropped from the text.
        # The fake vocabulary's </think> is token 26, and token 5 is a call,
        # which begins with <tool_call>.
        harness = self.harness(
            FakeRuntime(Plan([[1], [5], [26], [27]])),
            chat_format=replace(QWEN, think_end_id=26, call_open_id=5),
        )
        with openai_client(harness) as client:
            response = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"type": "function", "function": {"name": "weather"}}],
                temperature=0,
            )
        message = response.choices[0].message
        self.assertEqual(message.reasoning_content, "because ")
        self.assertEqual(
            [
                (call.function.name, call.function.arguments)
                for call in message.tool_calls
            ],
            [("weather", '{"city":"Paris"}')],
        )
        self.assertEqual(message.content, "\nanswer\n")
        self.assertEqual(response.usage.completion_tokens_details.reasoning_tokens, 1)

    def test_stream_options_null_uses_defaults(self):
        harness = self.harness(FakeRuntime(Plan([[3]])))
        status, content_type, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(
                stream=True,
                stream_options=None,
                reasoning_effort="none",
            ),
        )
        self.assertEqual((status, content_type), (200, "text/event-stream"))
        events = [
            line[6:]
            for line in payload.decode().splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        self.assertEqual(
            chunks[0]["choices"][0]["delta"],
            {"role": "assistant", "content": ""},
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_validation(self):
        harness = self.harness(FakeRuntime())
        invalid = [
            chat_body(stop=""),
            chat_body(stop=3),
            chat_body(stop=["x"] * 5),
            chat_body(stop=["x", ""]),
            chat_body(reasoning_effort="bad"),
            chat_body(seed=2**64),
            chat_body(n=True),
            chat_body(n=1.0),
            chat_body(logprobs=0),
            chat_body(temperature=float("nan")),
            chat_body(stream="true"),
            chat_body(messages=[{"role": "system", "content": "instructions"}]),
            chat_body(messages=[{"role": "assistant", "content": "answer"}]),
            chat_body(messages=[{"role": "user", "content": [{"type": "image_url"}]}]),
            chat_body(
                messages=[{"role": "user", "content": [{"type": "text", "text": 3}]}]
            ),
            chat_body(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": None, "tool_calls": 3},
                ]
            ),
            chat_body(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "tool", "content": "result", "tool_call_id": {}},
                ]
            ),
            chat_body(tools=[{"type": "function", "function": "bad"}]),
            chat_body(tools=[{"type": "function", "function": {"name": ""}}]),
            chat_body(
                tools=[
                    {"type": "function", "function": {"name": "same"}},
                    {"type": "function", "function": {"name": "same"}},
                ]
            ),
            chat_body(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": []},
                    }
                ]
            ),
            chat_body(parallel_tool_calls=1),
            chat_body(
                tools=[{"type": "function", "function": {"name": "weather"}}],
                tool_choice={"type": "function", "function": {"name": []}},
            ),
            chat_body(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "tool_calls": [{"function": "bad"}]},
                ]
            ),
        ]
        for body in invalid:
            status, _, _ = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 400)
        for stop in (None, [], "x", ["x", "y"]):
            body = chat_body()
            if stop is not None:
                body["stop"] = stop
            status, _, _ = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
