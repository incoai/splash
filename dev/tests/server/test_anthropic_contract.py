import copy
import json
import unittest
from unittest import mock

from dev.tests.server_fixtures import (
    FOREVER,
    FakeConstraintFactory,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    ImagePadTokenizer,
    Plan,
    TemplateTokenizer,
    anthropic_body,
    document_block,
    make_frontend,
    no_signed_thinking,
    png_data_url,
    reasoning_template,
    response_events,
)
from server import documents
from server import frontend as request_frontend
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

    def test_anthropic_tool_history_and_choice_use_one_chat_pipeline(self):
        translated, _ = anthropic_to_chat_body(
            anthropic_body(
                system=[{"type": "text", "text": "be exact"}],
                messages=[
                    {"role": "user", "content": "read it"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "read_file",
                                "input": {"path": "a.py"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [{"type": "text", "text": "contents"}],
                            },
                            {"type": "text", "text": "summarize"},
                        ],
                    },
                ],
                tools=[
                    {
                        "name": "read_file",
                        "description": "Read a local file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                tool_choice={
                    "type": "tool",
                    "name": "read_file",
                    "disable_parallel_tool_use": True,
                },
            ),
            thinking_resolver=no_signed_thinking,
        )
        self.assertEqual(
            translated["messages"][0], {"role": "system", "content": "be exact"}
        )
        self.assertEqual(translated["messages"][2]["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(translated["messages"][3]["role"], "tool")
        self.assertEqual(translated["messages"][4]["content"], "summarize")
        self.assertEqual(
            translated["tool_choice"],
            {"type": "function", "function": {"name": "read_file"}},
        )
        self.assertFalse(translated["parallel_tool_calls"])

    def test_anthropic_tool_references_name_the_found_tools(self):
        def tool_message(content):
            translated, _ = anthropic_to_chat_body(
                anthropic_body(
                    messages=[
                        {"role": "user", "content": "find the tool"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "ToolSearch",
                                    "input": {"query": "select:recall"},
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": content,
                                }
                            ],
                        },
                    ],
                    tools=[
                        {"name": "ToolSearch", "input_schema": {"type": "object"}},
                        {
                            "name": "recall",
                            "input_schema": {"type": "object"},
                            "defer_loading": True,
                        },
                    ],
                ),
                thinking_resolver=no_signed_thinking,
            )
            self.assertEqual(
                [tool["function"]["name"] for tool in translated["tools"]],
                ["ToolSearch", "recall"],
            )
            return translated["messages"][-1]

        reference = {"type": "tool_reference", "tool_name": "recall"}
        self.assertEqual(
            tool_message([reference]),
            {"role": "tool", "tool_call_id": "toolu_1", "content": "recall"},
        )
        self.assertEqual(
            tool_message(
                [
                    {"type": "text", "text": "Found:"},
                    reference,
                    {"type": "tool_reference", "tool_name": "remember"},
                ]
            )["content"],
            "Found:\nrecall\nremember",
        )
        for invalid in ({"type": "tool_reference"}, {**reference, "tool_name": ""}):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(APIError, "invalid Anthropic tool_reference"),
            ):
                tool_message([invalid])
        # A reference belongs in a tool result, not in a user's own content.
        with self.assertRaisesRegex(APIError, "unsupported Anthropic content block"):
            anthropic_to_chat_body(
                anthropic_body(messages=[{"role": "user", "content": [reference]}]),
                thinking_resolver=no_signed_thinking,
            )

    def test_anthropic_adaptive_thinking_maps_reasoning_effort(self):
        translated, _ = anthropic_to_chat_body(
            anthropic_body(
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
            ),
            thinking_resolver=no_signed_thinking,
        )
        self.assertEqual(translated["reasoning_effort"], "high")
        translated, _ = anthropic_to_chat_body(
            anthropic_body(
                thinking={"type": "adaptive"},
                output_config={"effort": "low"},
            ),
            thinking_resolver=no_signed_thinking,
        )
        self.assertEqual(translated["reasoning_effort"], "low")

    def test_anthropic_strips_nonsemantic_billing_system_block(self):
        def translated(metadata):
            chat, _ = anthropic_to_chat_body(
                anthropic_body(
                    system=[
                        {
                            "type": "text",
                            "text": f"x-anthropic-billing-header: {metadata}",
                        },
                        {
                            "type": "text",
                            "text": "stable agent system",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "dynamic status"},
                    ]
                ),
                thinking_resolver=no_signed_thinking,
            )
            return chat

        first = translated("one")
        self.assertEqual(first, translated("two"))
        self.assertEqual(
            first["messages"][0],
            {"role": "system", "content": "stable agent systemdynamic status"},
        )

    def test_anthropic_preserves_inline_system_messages(self):
        translated, _ = anthropic_to_chat_body(
            anthropic_body(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "system", "content": "dynamic system update"},
                    {"role": "assistant", "content": "acknowledged"},
                    {"role": "user", "content": "continue"},
                ]
            ),
            thinking_resolver=no_signed_thinking,
        )
        self.assertEqual(
            translated["messages"],
            [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "dynamic system update"},
                {"role": "assistant", "content": "acknowledged"},
                {"role": "user", "content": "continue"},
            ],
        )

        body = anthropic_body(messages=[{"content": "missing role"}])
        with self.assertRaisesRegex(APIError, "require user/assistant/system roles"):
            anthropic_to_chat_body(body, thinking_resolver=no_signed_thinking)

    def test_anthropic_thinking_off_is_not_reenabled_by_effort(self):
        tokenizer = TemplateTokenizer(reasoning_template())
        app = make_frontend(tokenizer, None, "test-model", 128, 1, 2, vision=True)
        for thinking in (None, {"type": "disabled"}):
            body = anthropic_body(output_config={"effort": "high"})
            if thinking is not None:
                body["thinking"] = thinking
            chat, _ = anthropic_to_chat_body(body, thinking_resolver=no_signed_thinking)
            job = app.prepare(chat, deadline=FOREVER)
            self.assertFalse(job.thinking)
            self.assertEqual(
                app.count_tokens(
                    anthropic_to_chat_prompt(
                        body, thinking_resolver=no_signed_thinking
                    ),
                    deadline=FOREVER,
                ),
                len(job.prompt_tokens),
            )
            self.assertFalse(tokenizer.templates[-1][1]["enable_thinking"])

    def test_anthropic_effort_is_validated_even_when_thinking_is_off(self):
        for thinking in (
            None,
            {"type": "disabled"},
            {"type": "enabled"},
            {"type": "adaptive"},
        ):
            for effort in ("invalid", "none", "ultra", {}, None):
                body = anthropic_body(output_config={"effort": effort})
                if thinking is not None:
                    body["thinking"] = thinking
                with (
                    self.subTest(thinking=thinking, effort=effort),
                    self.assertRaisesRegex(APIError, "output_config.effort"),
                ):
                    anthropic_to_chat_prompt(body, thinking_resolver=no_signed_thinking)


class AnthropicHTTPContractTest(HarnessTestCase):
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
        tokenizer = FakeTokenizer()
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
        class InputTokenizer(FakeTokenizer):
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
            tokenizer=ImagePadTokenizer(),
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

    def test_anthropic_messages_nonstream_stream_and_errors(self):
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime)
        status, content_type, payload = harness.request(
            "POST", "/v1/messages", anthropic_body()
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(content_type, "application/json")
        message = json.loads(payload)
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(
            message["content"], [{"type": "text", "text": "plain answer\n"}]
        )
        self.assertEqual(message["stop_reason"], "end_turn")
        self.assertEqual(
            message["usage"],
            {"input_tokens": 1, "cache_read_input_tokens": 1, "output_tokens": 1},
        )

        # Standard Anthropic query parameters do not change endpoint routing.
        status, _, payload = harness.request(
            "POST", "/v1/messages?beta=true", anthropic_body()
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["type"], "message")

        status, content_type, payload = harness.request(
            "POST", "/v1/messages", anthropic_body(stream=True)
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(content_type, "text/event-stream")
        events = response_events(payload)
        self.assertEqual(events[0]["type"], "message_start")
        self.assertEqual(events[-2]["type"], "message_delta")
        self.assertEqual(events[-2]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(events[-1], {"type": "message_stop"})
        text = "".join(
            event["delta"]["text"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "text_delta"
        )
        self.assertEqual(text, "plain answer\n")

        body = anthropic_body()
        del body["max_tokens"]
        status, _, payload = harness.request("POST", "/v1/messages", body)
        self.assertEqual(status, 400)
        error = json.loads(payload)
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["error"]["type"], "invalid_request_error")

    def test_anthropic_count_tokens_matches_generation_without_admission(self):
        class InputTokenizer(FakeTokenizer):
            def apply_chat_template(
                self, messages, add_generation_prompt=False, **kwargs
            ):
                prompt = super().apply_chat_template(
                    messages, add_generation_prompt=add_generation_prompt, **kwargs
                )
                return json.dumps([messages, kwargs], sort_keys=True) + prompt

            def __call__(self, text, **kwargs):
                return {"input_ids": list(text.encode())}

        runtime = FakeRuntime()
        factory = FakeConstraintFactory()
        harness = self.harness(
            runtime,
            tokenizer=InputTokenizer(),
            constraint_factory=factory,
            max_context=32768,
        )
        tools = [
            {
                "name": "read_file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ]
        history = [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect the file."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read_file",
                        "input": {"path": "hello.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "hello"},
                    {"type": "text", "text": "Summarize."},
                ],
            },
        ]
        cases = [
            {},
            {
                "system": [{"type": "text", "text": "Be concise."}],
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
            {
                "tools": tools,
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                        },
                    }
                },
            },
            *(
                {"messages": history, "tools": tools, "tool_choice": choice}
                for choice in (
                    {"type": "none"},
                    {"type": "auto"},
                    {"type": "any"},
                    {"type": "tool", "name": "read_file"},
                )
            ),
        ]
        counts = []
        for index, extra in enumerate(cases):
            with self.subTest(extra=extra):
                body = anthropic_body(**extra)
                del body["max_tokens"]
                with (
                    mock.patch.object(
                        request_frontend, "Job", side_effect=AssertionError("Job")
                    ),
                    mock.patch.object(
                        factory, "create", side_effect=AssertionError("grammar")
                    ),
                ):
                    for suffix in ("", "?beta=true"):
                        status, _, payload = harness.request(
                            "POST", "/v1/messages/count_tokens" + suffix, body
                        )
                        self.assertEqual(status, 200, payload)
                        counted = json.loads(payload)
                        self.assertEqual(set(counted), {"input_tokens"})
                chat, _ = anthropic_to_chat_body(
                    {**body, "max_tokens": 8}, thinking_resolver=no_signed_thinking
                )
                job = harness.app.prepare(chat, deadline=FOREVER)
                self.assertEqual(counted["input_tokens"], len(job.prompt_tokens))
                self.assertEqual(job.request_id, index + 1)
                counts.append(counted["input_tokens"])
        self.assertGreater(len(set(counts)), 2)
        self.assertEqual(runtime.requests, [])
        self._wait_for_http_active(harness.server.requests, 0)

    def test_anthropic_count_tokens_can_measure_over_context_images(self):
        harness = self.harness(
            FakeRuntime(), tokenizer=ImagePadTokenizer(), max_context=16
        )
        data = png_data_url().partition(";base64,")[2]
        image = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": data,
            },
        }
        body = anthropic_body(messages=[{"role": "user", "content": [image, image]}])
        del body["max_tokens"]
        with mock.patch.object(
            harness.app, "_expand_image_pads", side_effect=AssertionError("expansion")
        ):
            status, _, payload = harness.request(
                "POST", "/v1/messages/count_tokens", body
            )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload), {"input_tokens": 130})
        self.assertEqual(harness.app.images.stats()["request_bytes"], 0)
        chat, _ = anthropic_to_chat_body(
            {**body, "max_tokens": 1}, thinking_resolver=no_signed_thinking
        )
        with self.assertRaisesRegex(APIError, "context window"):
            harness.app.prepare(chat, deadline=FOREVER)
        harness.app.max_context = 256
        job = harness.app.prepare(chat, deadline=FOREVER)
        self.assertEqual(len(job.prompt_tokens), 130)
        del job
        self.assertEqual(harness.app.images.stats()["request_bytes"], 0)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_anthropic_count_tokens_validates_input_and_releases_capacity(self):
        harness = self.harness(FakeRuntime())
        valid = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        for body, expected in (
            ([], 400),
            ({}, 400),
            ({"messages": valid["messages"]}, 400),
            ({"model": "test-model"}, 400),
            ({**valid, "messages": []}, 400),
            ({**valid, "model": "other-model"}, 404),
            ({**valid, "tools": "bad"}, 400),
            ({**valid, "thinking": {"type": "bad"}}, 400),
            (
                {
                    **valid,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "invalid",
                                    },
                                }
                            ],
                        }
                    ],
                },
                400,
            ),
        ):
            with self.subTest(body=body):
                status, _, payload = harness.request(
                    "POST", "/v1/messages/count_tokens?beta=true", body
                )
                self.assertEqual(status, expected, payload)
                self.assertEqual(json.loads(payload)["type"], "error")
                self._wait_for_http_active(harness.server.requests, 0)
                self.assertEqual(harness.app.preparation_active, 0)
        with mock.patch.object(
            FakeTokenizer, "__call__", return_value={"input_ids": list(range(256))}
        ):
            status, _, payload = harness.request(
                "POST", "/v1/messages/count_tokens", valid
            )
        self.assertEqual((status, json.loads(payload)), (200, {"input_tokens": 256}))
        for _ in range(harness.app.preparation_capacity):
            harness.app.preparation_slots.acquire()
        try:
            status, _, payload = harness.request(
                "POST", "/v1/messages/count_tokens", {**valid, "timeout": 0.02}
            )
            self.assertEqual(status, 504, payload)
        finally:
            for _ in range(harness.app.preparation_capacity):
                harness.app.preparation_slots.release()
        self._wait_for_http_active(harness.server.requests, 0)
        self.assertEqual(harness.app.preparation_waiting, 0)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_anthropic_adaptive_effort_rejects_non_strings_before_admission(self):
        runtime = FakeRuntime(Plan(batches=[(3,)]))
        harness = self.harness(runtime)
        for effort in ([], {}, None, True, 1):
            with self.subTest(effort=effort):
                status, _, raw = harness.request(
                    "POST",
                    "/v1/messages",
                    anthropic_body(
                        thinking={"type": "adaptive"},
                        output_config={"effort": effort},
                    ),
                )
                self.assertEqual(status, 400)
                self.assertEqual(
                    json.loads(raw)["error"]["type"], "invalid_request_error"
                )
        self.assertEqual(runtime.requests, [])
        status, _, _ = harness.request(
            "POST",
            "/v1/messages",
            anthropic_body(
                thinking={"type": "adaptive"}, output_config={"effort": "low"}
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(runtime.requests), 1)

    def test_anthropic_cache_usage_and_matched_stop_sequence(self):
        for stream in (False, True):
            for matched_tokens in (0, 1):
                with self.subTest(stream=stream, matched_tokens=matched_tokens):
                    runtime = FakeRuntime(
                        Plan(
                            [[14], [15], [4]], delay=0.01, matched_tokens=matched_tokens
                        )
                    )
                    harness = self.harness(runtime)
                    status, _, payload = harness.request(
                        "POST",
                        "/v1/messages",
                        anthropic_body(
                            stream=stream, stop_sequences=["unused", "second"]
                        ),
                    )
                    self.assertEqual(status, 200, payload)
                    if stream:
                        events = response_events(payload)
                        usage = {
                            **events[0]["message"]["usage"],
                            **events[-2]["usage"],
                        }
                        finish = events[-2]["delta"]
                        content = "".join(
                            event["delta"]["text"]
                            for event in events
                            if event["type"] == "content_block_delta"
                        )
                    else:
                        finish = json.loads(payload)
                        usage = finish["usage"]
                        content = finish["content"][0]["text"]
                    self.assertEqual(
                        usage,
                        {
                            "input_tokens": 2 - matched_tokens,
                            "cache_read_input_tokens": matched_tokens,
                            "output_tokens": 2,
                        },
                    )
                    self.assertEqual(finish["stop_reason"], "stop_sequence")
                    self.assertEqual(finish["stop_sequence"], "second")
                    self.assertEqual(content, "first ")
                    self.assertEqual(runtime.cancel_count, 1)


if __name__ == "__main__":
    unittest.main()
