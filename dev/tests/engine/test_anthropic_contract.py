import copy
import json
import unittest

from dev.tests import test_server
from dev.tests.engine.test_documents import document_block
from dev.tests.test_server import (
    FOREVER,
    FakeConstraintFactory,
    FakeRuntime,
    Harness,
    Plan,
    no_signed_thinking,
)
from server import documents
from server.api_shapes import (
    anthropic_to_chat_body,
    anthropic_to_chat_prompt,
    normalize_messages,
)
from server.errors import APIError

SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
    "additionalProperties": False,
}
FORMAT = {"type": "json_schema", "schema": SCHEMA}


def request_body(**fields):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        **fields,
    }


def stream_events(payload):
    return [
        json.loads(line.removeprefix("data: "))
        for line in payload.decode().splitlines()
        if line.startswith("data: ")
    ]


class AnthropicAdapterTest(unittest.TestCase):
    def test_redacted_history_preserves_visible_text_and_tool_calls(self):
        body = request_body(
            messages=[
                {"role": "user", "content": "Check Paris."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque-provider-data"},
                        {"type": "text", "text": "Checking."},
                        {
                            "type": "tool_use",
                            "id": "call1",
                            "name": "weather",
                            "input": {"city": "Paris"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call1",
                            "content": "Sunny.",
                        }
                    ],
                },
            ]
        )
        original = copy.deepcopy(body)
        translated = anthropic_to_chat_prompt(
            body, thinking_resolver=no_signed_thinking
        )
        messages = translated["messages"]
        self.assertEqual(messages[1]["content"], "Checking.")
        self.assertNotIn("reasoning_content", messages[1])
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["arguments"], {"city": "Paris"}
        )
        self.assertEqual(messages[2]["tool_call_id"], "call1")
        self.assertNotIn("opaque-provider-data", json.dumps(messages))
        self.assertEqual(body, original)

    def test_redacted_history_requires_assistant_and_nonempty_data(self):
        for role, data in (
            ("user", "opaque"),
            ("assistant", None),
            ("assistant", 7),
            ("assistant", ""),
        ):
            with (
                self.subTest(role=role, data=data),
                self.assertRaises(APIError) as caught,
            ):
                anthropic_to_chat_prompt(
                    request_body(
                        messages=[
                            {
                                "role": role,
                                "content": [
                                    {"type": "redacted_thinking", "data": data}
                                ],
                            }
                        ]
                    ),
                    thinking_resolver=no_signed_thinking,
                )
            self.assertEqual(caught.exception.status, 400)

    def test_empty_context_management_preserves_template_defaults(self):
        for fields in (
            {},
            {"context_management": None},
            {"context_management": {}},
            {"context_management": {"edits": []}},
        ):
            with self.subTest(fields=fields):
                self.assertNotIn(
                    "preserve_thinking",
                    anthropic_to_chat_prompt(
                        request_body(**fields), thinking_resolver=no_signed_thinking
                    ),
                )

    def test_format_aliases_preserve_schema_and_input(self):
        for fields in (
            {"output_config": {"format": FORMAT}},
            {"output_format": FORMAT},
        ):
            with self.subTest(fields=fields):
                body = request_body(**fields)
                original = copy.deepcopy(body)
                translated, _ = anthropic_to_chat_body(
                    body, thinking_resolver=no_signed_thinking
                )
                self.assertEqual(body, original)
                self.assertEqual(
                    translated["response_format"],
                    {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
                )
                self.assertEqual(translated["reasoning_effort"], "none")

    def test_format_and_effort_can_be_combined_with_tools(self):
        body = request_body(
            thinking={"type": "adaptive"},
            output_config={"effort": "xhigh", "format": FORMAT},
            tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
        )
        translated = anthropic_to_chat_prompt(
            body, thinking_resolver=no_signed_thinking
        )
        self.assertEqual(translated["reasoning_effort"], "xhigh")
        self.assertEqual(translated["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertEqual(translated["tools"][0]["function"]["name"], "lookup")

    def test_a_strict_tool_stays_strict(self):
        body = request_body(
            tools=[
                {"name": "lookup", "input_schema": {"type": "object"}, "strict": True}
            ],
        )
        translated = anthropic_to_chat_prompt(
            body, thinking_resolver=no_signed_thinking
        )
        self.assertIs(translated["tools"][0]["function"]["strict"], True)

    def test_format_shape_validation_is_independent_of_thinking(self):
        invalid = [
            {"output_config": value} for value in (None, [], "json_schema", False)
        ]
        invalid.extend(
            {"output_config": {"format": value}}
            for value in (
                None,
                [],
                {},
                {"type": "text"},
                {"type": "json_schema"},
                {"type": "json_schema", "schema": []},
            )
        )
        invalid.extend(
            {"output_format": value}
            for value in (None, [], {}, {"type": "json_schema", "schema": 1})
        )
        invalid.append({"output_config": {"format": FORMAT}, "output_format": FORMAT})
        for fields in invalid:
            for thinking in (None, {"type": "enabled"}, {"type": "adaptive"}):
                with self.subTest(fields=fields, thinking=thinking):
                    with self.assertRaises(APIError):
                        anthropic_to_chat_prompt(
                            request_body(thinking=thinking, **fields),
                            thinking_resolver=no_signed_thinking,
                        )


class AnthropicHTTPContractTest(unittest.TestCase):
    def harness(self, runtime=None, **kwargs):
        harness = Harness(runtime or FakeRuntime(), **kwargs)
        self.addCleanup(harness.close)
        return harness

    def test_trailing_assistant_prefill_is_refused(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        body = request_body(
            messages=[
                {"role": "user", "content": "Answer in JSON."},
                {"role": "assistant", "content": "{"},
            ]
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, payload = harness.request(
                    "POST", "/v1/messages", {**body, "stream": stream}
                )
                self.assertEqual(status, 400, payload)
                error = json.loads(payload)["error"]
                self.assertEqual(error["type"], "invalid_request_error")
                self.assertIn("prefill", error["message"])
        self.assertEqual(runtime.requests, [])
        status, _, payload = harness.request("POST", "/v1/messages/count_tokens", body)
        self.assertEqual(status, 200, payload)
        self.assertGreater(json.loads(payload)["input_tokens"], 0)

    def test_redacted_history_count_and_generation_use_same_visible_prompt(self):
        tokenizer = test_server.FakeTokenizer()
        harness = self.harness(tokenizer=tokenizer)
        body = request_body(
            messages=[
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "Hi"},
                    ],
                },
                {"role": "user", "content": "Continue"},
            ]
        )
        status, _, payload = harness.request(
            "POST", "/v1/messages/count_tokens?beta=true", body
        )
        self.assertEqual(status, 200, payload)
        count = json.loads(payload)["input_tokens"]
        counted = copy.deepcopy(tokenizer.templates[-1])
        self.assertEqual(counted[0][1], {"role": "assistant", "content": "Hi"})
        for stream in (False, True):
            status, _, payload = harness.request(
                "POST", "/v1/messages", {**body, "stream": stream}
            )
            self.assertEqual(status, 200, payload)
            self.assertEqual(tokenizer.templates[-1], counted)
            usage = (
                stream_events(payload)[0]["message"]["usage"]
                if stream
                else json.loads(payload)["usage"]
            )
            self.assertEqual(
                count, usage["input_tokens"] + usage["cache_read_input_tokens"]
            )

    def test_keep_all_forwards_history_and_agrees_with_generation_count(self):
        class InputTokenizer(test_server.FakeTokenizer):
            def apply_chat_template(
                self, messages, add_generation_prompt=False, **kwargs
            ):
                prompt = super().apply_chat_template(
                    messages, add_generation_prompt=add_generation_prompt, **kwargs
                )
                return json.dumps([messages, kwargs], sort_keys=True) + prompt

            def __call__(self, text, **_kwargs):
                return {"input_ids": list(text.encode())}

        context = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
        history = [
            {"role": "user", "content": "First question."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "OLD_REASON_MARKER"},
                    {"type": "text", "text": "First answer."},
                ],
            },
            {"role": "user", "content": "Next question."},
        ]
        runtime = FakeRuntime()
        tokenizer = InputTokenizer()
        harness = self.harness(runtime, tokenizer=tokenizer, max_context=8192)
        default = request_body(messages=history)
        status, _, payload = harness.request(
            "POST", "/v1/messages/count_tokens", default
        )
        self.assertEqual(status, 200, payload)
        self.assertNotIn("preserve_thinking", tokenizer.templates[-1][1])
        body = {**default, "context_management": context}
        before = copy.deepcopy(body)
        status, _, payload = harness.request(
            "POST", "/v1/messages/count_tokens?beta=true", body
        )
        self.assertEqual(status, 200, payload)
        count = json.loads(payload)["input_tokens"]
        counted_template = copy.deepcopy(tokenizer.templates[-1])
        self.assertEqual(
            counted_template[0][1]["reasoning_content"], "OLD_REASON_MARKER"
        )
        self.assertIs(counted_template[1]["preserve_thinking"], True)
        self.assertEqual(runtime.requests, [])
        status, _, payload = harness.request("POST", "/v1/messages", body)
        self.assertEqual(status, 200, payload)
        usage = json.loads(payload)["usage"]
        self.assertEqual(
            count, usage["input_tokens"] + usage["cache_read_input_tokens"]
        )
        self.assertEqual(tokenizer.templates[-1], counted_template)
        self.assertEqual(body, before)

    def test_unsupported_context_edits_reject_before_counting_or_generation(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        invalid = [
            [],
            False,
            "all",
            {"edits": None},
            {"edits": {}},
            {"edits": [None]},
            {"edits": [{"type": "clear_thinking_20251015"}]},
            {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": 1},
                    }
                ]
            },
            {"edits": [{"type": "clear_tool_uses_20250919"}]},
            {"edits": [{"type": "compact_20260112"}]},
        ]
        for context in invalid:
            for path in ("/v1/messages", "/v1/messages/count_tokens?beta=true"):
                with self.subTest(context=context, path=path):
                    status, _, payload = harness.request(
                        "POST", path, request_body(context_management=context)
                    )
                    self.assertEqual(status, 400, payload)
                    self.assertIn(b"context_management", payload)
        self.assertEqual(runtime.requests, [])
        self.assertEqual(harness.tokenizer.templates, [])

    def test_responses_rejects_unimplemented_context_management_and_truncation(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        body = {
            "model": "test-model",
            "input": "hello",
            "reasoning": {"effort": "none"},
        }
        for fields in (
            *({"truncation": value} for value in ("auto", "unknown", True, [], {})),
            *(
                {"context_management": value}
                for value in (
                    [{"type": "compaction", "compact_threshold": 1024}],
                    {},
                    True,
                    "auto",
                )
            ),
        ):
            with self.subTest(fields=fields):
                status, _, payload = harness.request(
                    "POST", "/v1/responses", {**body, **fields}
                )
                self.assertEqual(status, 400, payload)
        self.assertEqual(runtime.requests, [])
        for fields in (
            {},
            {"truncation": None, "context_management": None},
            {"truncation": "disabled", "context_management": []},
        ):
            status, _, payload = harness.request(
                "POST", "/v1/responses", {**body, **fields}
            )
            self.assertEqual(status, 200, payload)
        self.assertEqual(len(runtime.requests), 3)

    def test_pdf_user_and_tool_result_count_the_prepared_image(self):
        runtime = FakeRuntime()
        harness = self.harness(
            runtime,
            tokenizer=test_server.ServerTest.ImagePadTokenizer(),
            max_context=4096,
        )
        document = document_block()
        for messages in (
            [{"role": "user", "content": [document]}],
            [
                {"role": "user", "content": "Read the document."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "pdf", "name": "Read", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "pdf",
                            "content": [document],
                        }
                    ],
                },
            ],
        ):
            with self.subTest(tool_result=len(messages) > 1):
                body = request_body(messages=messages)
                translated, _ = anthropic_to_chat_body(
                    body, thinking_resolver=no_signed_thinking
                )
                # Conversion leaves the PDF to request preparation.
                self.assertEqual(
                    translated["messages"][-1]["content"],
                    [
                        {
                            "type": "file",
                            "file": {
                                "file_data": documents.PDF_DATA_URL_PREFIX
                                + document["source"]["data"]
                            },
                        }
                    ],
                )
                parts = normalize_messages(
                    translated["messages"], vision=True, deadline=FOREVER
                )[-1]["content"]
                self.assertIn("ALPHA 42", parts[0]["text"])
                self.assertEqual(parts[1]["type"], "image_url")
                status, _, payload = harness.request(
                    "POST", "/v1/messages/count_tokens?beta=true", body
                )
                self.assertEqual(status, 200, payload)
                counted = json.loads(payload)["input_tokens"]
                job = harness.app.prepare(translated, deadline=FOREVER)
                self.assertEqual(len(job.prompt_tokens), counted)
                self.assertGreater(counted, 2)
                self.assertEqual(len(job.image_spans), 1)
                del job
        self.assertEqual(runtime.requests, [])

    def test_all_adaptive_efforts_generate_and_count(self):
        harness = self.harness(FakeRuntime(*(Plan([[1, 2, 3]]) for _ in range(5))))
        for effort, expected in (
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
        ):
            with self.subTest(effort=effort):
                body = request_body(
                    thinking={"type": "adaptive"}, output_config={"effort": effort}
                )
                for path in ("/v1/messages", "/v1/messages/count_tokens?beta=true"):
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200, payload)
                    self.assertEqual(
                        harness.tokenizer.templates[-1][1]["reasoning_effort"], expected
                    )

    def test_invalid_efforts_reject_without_native_admission(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for effort in (None, True, 3, [], "invalid"):
            for thinking in (None, {"type": "adaptive"}, {"type": "enabled"}):
                for path in ("/v1/messages", "/v1/messages/count_tokens"):
                    with self.subTest(effort=effort, thinking=thinking, path=path):
                        status, _, payload = harness.request(
                            "POST",
                            path,
                            request_body(
                                thinking=thinking, output_config={"effort": effort}
                            ),
                        )
                        self.assertEqual(status, 400, payload)
                        self.assertIn(
                            "output_config.effort",
                            json.loads(payload)["error"]["message"],
                        )
        self.assertEqual(runtime.requests, [])
        self.assertEqual(harness.tokenizer.templates, [])

    def test_structured_output_aliases_stream_and_count(self):
        runtime = FakeRuntime(*(Plan([[10]]) for _ in range(4)))
        factory = FakeConstraintFactory()
        harness = self.harness(runtime, constraint_factory=factory)
        for fields in (
            {"output_config": {"format": FORMAT}},
            {"output_format": FORMAT},
        ):
            for stream in (False, True):
                with self.subTest(fields=fields, stream=stream):
                    body = request_body(stream=stream, **fields)
                    status, _, payload = harness.request("POST", "/v1/messages", body)
                    self.assertEqual(status, 200, payload)
                    if stream:
                        events = stream_events(payload)
                        text = "".join(
                            event["delta"]["text"]
                            for event in events
                            if event["type"] == "content_block_delta"
                            and event["delta"]["type"] == "text_delta"
                        )
                        self.assertEqual(events[-1]["type"], "message_stop")
                    else:
                        response = json.loads(payload)
                        text = response["content"][0]["text"]
                        self.assertEqual(response["stop_reason"], "end_turn")
                    self.assertEqual(json.loads(text), {"x": 3})
                    status, _, payload = harness.request(
                        "POST", "/v1/messages/count_tokens?beta=true", body
                    )
                    self.assertEqual(status, 200, payload)
                    self.assertEqual(json.loads(payload), {"input_tokens": 2})
        self.assertEqual(len(runtime.requests), 4)
        self.assertTrue(all("%json" in grammar for grammar in factory.grammars))
        self.assertEqual(len(set(factory.grammars)), 1)

    def test_invalid_schema_rejects_before_native_admission(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for schema in (
            {"type": "not-a-json-type"},
            {"$ref": "https://example.com/schema"},
        ):
            with self.subTest(schema=schema):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/messages",
                    request_body(
                        output_config={
                            "format": {"type": "json_schema", "schema": schema}
                        }
                    ),
                )
                self.assertEqual(status, 400, payload)
        self.assertEqual(runtime.requests, [])

    def test_structured_output_rejects_invalid_model_output(self):
        factory = FakeConstraintFactory()
        runtime = FakeRuntime(Plan([[4]]), Plan([[10]]), Plan([[4]]))
        harness = self.harness(runtime, constraint_factory=factory)
        for stream, schema in (
            (False, SCHEMA),
            (False, {"type": "object", "properties": {"x": {"const": 42}}}),
            (True, SCHEMA),
        ):
            with self.subTest(stream=stream, schema=schema):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/messages",
                    request_body(
                        stream=stream,
                        output_format={"type": "json_schema", "schema": schema},
                    ),
                )
                if stream:
                    self.assertEqual(status, 200, payload)
                    events = stream_events(payload)
                    self.assertEqual(events[-1]["type"], "error")
                    self.assertNotIn(
                        "message_stop", [event["type"] for event in events]
                    )
                else:
                    self.assertEqual(status, 500, payload)
                self.assertIn(b"api_error", payload)

    def test_output_limit_retains_partial_json(self):
        harness = self.harness(
            FakeRuntime(Plan([[12]], reason="length")),
            constraint_factory=FakeConstraintFactory(),
        )
        status, _, payload = harness.request(
            "POST", "/v1/messages", request_body(output_format=FORMAT)
        )
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        self.assertEqual(response["stop_reason"], "max_tokens")
        self.assertEqual(response["content"][0]["text"], '{"x":')

    def test_typed_tools_reject_but_custom_web_search_remains_valid(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for tool_type in (
            "web_search_20250305",
            "web_fetch_20250910",
            "code_execution_20250825",
            "computer_20250124",
            "future_tool",
            None,
            [],
        ):
            for path in ("/v1/messages", "/v1/messages/count_tokens"):
                with self.subTest(tool_type=tool_type, path=path):
                    status, _, payload = harness.request(
                        "POST",
                        path,
                        request_body(tools=[{"type": tool_type, "name": "web_search"}]),
                    )
                    self.assertEqual(status, 400, payload)
                    self.assertIn(b"only custom function tools", payload)
        self.assertEqual(runtime.requests, [])
        for extra in ({}, {"type": "custom"}):
            status, _, payload = harness.request(
                "POST",
                "/v1/messages",
                request_body(
                    tools=[
                        {
                            "name": "web_search",
                            "input_schema": {"type": "object"},
                            **extra,
                        }
                    ]
                ),
            )
            self.assertEqual(status, 200, payload)
        self.assertEqual(len(runtime.requests), 2)

    def test_unsupported_thinking_display_rejects_without_reasoning_leak(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for thinking_type in ("enabled", "adaptive", "disabled"):
            for display in ("omitted", "updates", "unknown", None, 0, []):
                if (
                    display in ("omitted", "updates", None)
                    and thinking_type != "disabled"
                ):
                    continue
                for stream in (False, True):
                    with self.subTest(
                        thinking_type=thinking_type, display=display, stream=stream
                    ):
                        status, _, payload = harness.request(
                            "POST",
                            "/v1/messages",
                            request_body(
                                stream=stream,
                                thinking={"type": thinking_type, "display": display},
                            ),
                        )
                        self.assertEqual(status, 400, payload)
                        self.assertNotIn(b"because ", payload)
        self.assertEqual(runtime.requests, [])
        for thinking in (
            {"type": "enabled"},
            {"type": "adaptive"},
            {"type": "disabled"},
            {"type": "enabled", "display": None},
            {"type": "adaptive", "display": None},
        ):
            status, _, payload = harness.request(
                "POST", "/v1/messages", request_body(thinking=thinking)
            )
            self.assertEqual(status, 200, payload)


if __name__ == "__main__":
    unittest.main()
