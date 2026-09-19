import copy
import json
import unittest
from itertools import product
from unittest import mock

from dev.tests.test_server import FakeRuntime, Harness, Plan
from server import errors as api_errors
from server import output as model_output
from server import server as api
from server import tool_schema
from server.api_shapes import (
    anthropic_to_chat_prompt,
    normalize_messages,
    template_messages,
)
from server.thinking import ThinkingCodec


def request_body(**fields):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        **fields,
    }


class ProtocolRecoveryTests(unittest.TestCase):
    def harness(self, runtime, **kwargs):
        harness = Harness(runtime, **kwargs)
        self.addCleanup(harness.close)
        return harness

    def test_hidden_thinking_stream_and_complete_roundtrip(self):
        for display, thinking_type, stream in product(
            ("omitted", "updates"), ("enabled", "adaptive"), (False, True)
        ):
            with self.subTest(
                display=display, thinking_type=thinking_type, stream=stream
            ):
                harness = self.harness(FakeRuntime(Plan([[1], [2], [3]])))
                status, _, payload = harness.request(
                    "POST",
                    "/v1/messages",
                    request_body(
                        stream=stream,
                        thinking={"type": thinking_type, "display": display},
                    ),
                )
                self.assertEqual(status, 200, payload)
                self.assertNotIn(b"because ", payload)
                if stream:
                    events = [
                        json.loads(line[6:])
                        for line in payload.decode().splitlines()
                        if line.startswith("data: ")
                    ]
                    deltas = [
                        event["delta"]
                        for event in events
                        if event["type"] == "content_block_delta"
                    ]
                    reasoning = [
                        delta for delta in deltas if delta["type"] == "thinking_delta"
                    ]
                    self.assertEqual(
                        reasoning, [{"type": "thinking_delta", "thinking": ""}]
                    )
                    signature = next(
                        delta["signature"]
                        for delta in deltas
                        if delta["type"] == "signature_delta"
                    )
                    block = {"type": "thinking", "thinking": "", "signature": signature}
                    usage = events[-2]["usage"]
                else:
                    response = json.loads(payload)
                    block = response["content"][0]
                    self.assertEqual(block["type"], "thinking")
                    self.assertEqual(block["thinking"], "")
                    usage = response["usage"]
                hidden = harness.app.thinking_codec.decode(block["signature"])
                self.assertEqual(hidden, "because ")
                self.assertEqual(usage["output_tokens"], 3)
                history = [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": [block, {"type": "text", "text": "answer"}],
                    },
                    {"role": "user", "content": "continue"},
                ]
                continuation = request_body(messages=history)
                count_status, _, count_payload = harness.request(
                    "POST", "/v1/messages/count_tokens", continuation
                )
                self.assertEqual(count_status, 200, count_payload)
                self.assertEqual(
                    harness.tokenizer.templates[-1][0][1]["reasoning_content"], hidden
                )
                next_status, _, next_payload = harness.request(
                    "POST", "/v1/messages", continuation
                )
                self.assertEqual(next_status, 200, next_payload)
                self.assertEqual(
                    harness.tokenizer.templates[-1][0][1]["reasoning_content"], hidden
                )

    def test_null_thinking_display_keeps_default_reasoning(self):
        for thinking_type, stream in product(("enabled", "adaptive"), (False, True)):
            with self.subTest(thinking_type=thinking_type, stream=stream):
                harness = self.harness(FakeRuntime(Plan([[1], [2], [3]])))
                status, _, payload = harness.request(
                    "POST",
                    "/v1/messages",
                    request_body(
                        stream=stream,
                        thinking={"type": thinking_type, "display": None},
                    ),
                )
                self.assertEqual(status, 200, payload)
                self.assertIn(b"because ", payload)

    def test_omitted_tool_roundtrip_and_summarized_compatibility(self):
        codec = ThinkingCodec()
        signature = codec.encode("inspect before acting")
        for text, signed in (("ignored replacement", signature), ("plain history", "")):
            history = request_body(
                messages=[
                    {"role": "user", "content": "read"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": text, "signature": signed},
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "read",
                                "input": {"path": "a"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "done",
                            }
                        ],
                    },
                ]
            )
            chat = anthropic_to_chat_prompt(history, thinking_resolver=codec.decode)
            self.assertEqual(
                chat["messages"][1]["reasoning_content"],
                "inspect before acting" if signed else text,
            )
            self.assertEqual(chat["messages"][2]["tool_call_id"], "t1")

    def test_visible_foreign_thinking_can_continue_but_hidden_content_needs_its_key(
        self,
    ):
        foreign_signature = ThinkingCodec().encode("foreign reasoning")
        for visible in ("visible reasoning", ""):
            harness = self.harness(FakeRuntime(Plan([[4]])))
            body = request_body(
                messages=[
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": visible,
                                "signature": foreign_signature,
                            },
                            {"type": "text", "text": "hello"},
                        ],
                    },
                    {"role": "user", "content": "continue"},
                ]
            )
            for path in ("/v1/messages/count_tokens", "/v1/messages"):
                with self.subTest(visible=bool(visible), path=path):
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200 if visible else 400, payload)
                    if not visible:
                        self.assertIn(b"invalid signature in thinking block", payload)
            if visible:
                converted = anthropic_to_chat_prompt(
                    body, thinking_resolver=harness.app.thinking_codec.decode
                )
                self.assertEqual(converted["messages"][1]["reasoning_content"], visible)

    def test_hidden_thinking_survives_display_changes_and_tool_roundtrips(self):
        harness = self.harness(
            FakeRuntime(Plan([[1], [26], [5], [16]]), Plan([[1], [26], [5]]))
        )
        tools = [
            {
                "name": "weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
            {"name": "time", "input_schema": {"type": "object"}},
        ]
        messages = [{"role": "user", "content": "weather and time"}]
        for count, display in ((2, "updates"), (1, "omitted")):
            status, _, payload = harness.request(
                "POST",
                "/v1/messages",
                request_body(
                    messages=messages,
                    tools=tools,
                    thinking={"type": "adaptive", "display": display},
                ),
            )
            self.assertEqual(status, 200, payload)
            response = json.loads(payload)
            self.assertEqual(response["stop_reason"], "tool_use")
            self.assertEqual(response["content"][0]["thinking"], "")
            self.assertNotIn(b"because ", payload)
            calls = [b for b in response["content"] if b["type"] == "tool_use"]
            self.assertEqual(len(calls), count)
            messages.extend(
                [
                    {"role": "assistant", "content": response["content"]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call["id"],
                                "content": "done",
                            }
                            for call in calls
                        ],
                    },
                ]
            )
        status, _, payload = harness.request(
            "POST", "/v1/messages", request_body(messages=messages, tools=tools)
        )
        self.assertEqual(status, 200, payload)
        rendered_history = harness.tokenizer.templates[-1][0]
        reasoning = [
            message["reasoning_content"]
            for message in rendered_history
            if "reasoning_content" in message
        ]
        self.assertEqual(reasoning, ["because ", "because "])
        self.assertEqual(
            sum(message["role"] == "tool" for message in rendered_history), 3
        )

    def test_hidden_history_rejects_invalid_or_other_instance_signatures_before_submit(
        self,
    ):
        harness = self.harness(FakeRuntime())
        for signature in (
            "malformed-client-token",
            ThinkingCodec().encode("private reasoning"),
        ):
            for path in ("/v1/messages", "/v1/messages/count_tokens"):
                body = request_body(
                    messages=[
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": signature,
                                }
                            ],
                        },
                        {"role": "user", "content": "continue"},
                    ]
                )
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 400, payload)
                error = json.loads(payload)["error"]
                self.assertEqual(error["type"], "invalid_request_error")
                self.assertNotIn("private reasoning", error["message"])
                self.assertNotIn(signature, error["message"])
        self.assertFalse(harness.backend.runtime.requests)

    def test_thinking_signature_authentication_and_size_limits(self):
        codec = ThinkingCodec()
        signature = codec.encode("private reasoning")
        self.assertEqual(codec.decode(signature), "private reasoning")
        with self.assertRaises(api.APIError):
            ThinkingCodec().decode(signature)
        changed = (
            signature[:25] + ("A" if signature[25] != "A" else "B") + signature[26:]
        )
        for invalid in (changed, "not-a-signature", "☃", None):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(api.APIError) as caught,
            ):
                codec.decode(invalid)
            self.assertEqual(caught.exception.status, 400)
            self.assertNotIn("private reasoning", str(caught.exception))
        with mock.patch.object(codec, "MAX_SIGNATURE_BYTES", 8):
            with self.assertRaises(api.APIError):
                codec.decode(signature)
            with self.assertRaises(api.APIError):
                codec.encode("too long")
        for _ in range(300):
            codec.encode("later response")
        self.assertEqual(codec.decode(signature), "private reasoning")

    def test_truncated_calls_preserve_text_and_order_without_fabricated_arguments(self):
        def call(arguments):
            return {
                "type": "function",
                "function": {"name": "write", "arguments": arguments},
            }

        source = [
            {"role": "user", "content": "write"},
            {
                "role": "assistant",
                "content": "before",
                "reasoning_content": "reason",
                "tool_calls": [
                    call('{"a":1}'),
                    call('{"a":"unfinished'),
                    call('{"a":2}'),
                ],
            },
            {"role": "tool", "tool_call_id": "interrupted", "content": "interrupted"},
        ]
        original = copy.deepcopy(source)
        normalized = normalize_messages(source)
        projected = template_messages(normalized)
        self.assertEqual(source, original)
        self.assertEqual(
            normalized[1]["tool_calls"][1]["function"]["arguments"], '{"a":"unfinished'
        )
        self.assertEqual(
            projected[1]["tool_calls"][0]["function"]["arguments"], {"a": 1}
        )
        self.assertEqual(projected[1]["reasoning_content"], "reason")
        self.assertEqual(
            json.loads(projected[2]["content"].split("\n", 1)[1]),
            {"name": "write", "arguments": '{"a":"unfinished'},
        )
        self.assertNotIn("tool_calls", projected[2])
        self.assertEqual(
            projected[3]["tool_calls"][0]["function"]["arguments"], {"a": 2}
        )
        self.assertEqual(projected[4], source[2])

    def test_incomplete_responses_history_can_continue(self):
        harness = self.harness(FakeRuntime(Plan([[13]], reason="length")))
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
        status, _, raw = harness.request(
            "POST",
            "/v1/responses",
            {
                "model": "test-model",
                "input": "weather",
                "tools": [tool],
                "stream": True,
                "reasoning": {"effort": "none"},
            },
        )
        self.assertEqual(status, 200, raw)
        events = [
            json.loads(line[6:])
            for line in raw.decode().splitlines()
            if line.startswith("data: ")
        ]
        response = events[-1]["response"]
        self.assertEqual(response["output"][0]["arguments"], '{"city":"Par')
        status, _, raw = harness.request(
            "POST",
            "/v1/responses",
            {
                "model": "test-model",
                "input": "continue",
                "previous_response_id": response["id"],
            },
        )
        self.assertEqual(status, 200, raw)
        history = harness.tokenizer.templates[-1][0]
        self.assertNotIn("tool_calls", history[1])
        self.assertIn("Incomplete tool call", history[1]["content"])
        self.assertEqual(
            harness.app.response_store.get(response["id"]).response["output"],
            response["output"],
        )

    def test_mixed_string_unions_keep_schema_and_value_types(self):
        for union, valid, invalid in (
            (
                {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                ["abc", ["abc"]],
                3,
            ),
            ({"type": ["string", "integer"]}, ["abc", 7], False),
            (
                {"anyOf": [{"type": "string"}, {"type": "integer"}], "minimum": 3},
                ["abc", 7],
                2,
            ),
            ({"enum": ["abc", 7]}, ["abc", 7], False),
        ):
            schema = {
                "type": "object",
                "properties": {"value": union},
                "required": ["value"],
            }
            grammar = tool_schema._tool_arguments_grammar(schema)
            generated, _ = json.JSONDecoder().raw_decode(grammar.split("%json ", 1)[1])
            generated.pop("x-guidance", None)
            self.assertEqual(generated, union)
            _, policy = tool_schema.normalize_tools(
                [
                    {
                        "type": "function",
                        "function": {"name": "echo", "parameters": schema},
                    }
                ],
                "auto",
                True,
            )
            for value in [*valid, invalid]:
                text = (
                    "<tool_call>\n<function=echo>\n<parameter=value>\n"
                    + json.dumps(value)
                    + "\n</parameter>\n</function>\n</tool_call>"
                )
                _, calls = model_output.parse_tool_calls(text, 1, policy)
                self.assertEqual(
                    json.loads(calls[0]["function"]["arguments"]), {"value": value}
                )
                if value is invalid:
                    with self.assertRaises(api.APIError):
                        model_output.validate_tool_calls(calls, policy)
                else:
                    model_output.validate_tool_calls(calls, policy)

    def test_anthropic_overflow_has_actual_counts_for_text_and_image_precheck(self):
        harness = self.harness(FakeRuntime(), max_context=2)
        status, _, raw = harness.request("POST", "/v1/messages", request_body())
        self.assertEqual(status, 400)
        self.assertEqual(
            json.loads(raw)["error"]["message"],
            "prompt is too long: 2 tokens > 1 maximum input tokens",
        )
        with mock.patch.object(
            harness.app,
            "_prepare_images",
            side_effect=api_errors.ContextLengthError(100, 1, image_tokens_only=True),
        ):
            status, _, raw = harness.request("POST", "/v1/messages", request_body())
        self.assertEqual(status, 400)
        self.assertIn("100 tokens > 1", json.loads(raw)["error"]["message"])
        self.assertIn("image tokens alone", json.loads(raw)["error"]["message"])
        with self.assertRaises(api_errors.ContextLengthError) as caught:
            harness.app._check_image_request_size(100, 1, 4)
        self.assertEqual(caught.exception.input_tokens, 100)
        status, _, raw = harness.request("POST", "/v1/chat/completions", request_body())
        self.assertEqual(status, 400)
        self.assertEqual(
            json.loads(raw)["error"]["message"], "prompt exceeds the context window"
        )


if __name__ == "__main__":
    unittest.main()
