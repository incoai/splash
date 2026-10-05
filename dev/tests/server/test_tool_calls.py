import json
import time
import unittest
from unittest import mock

from openai import OpenAI

from dev.tests.server_fixtures import (
    FOREVER,
    FakeConstraintFactory,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    PassthroughStreamer,
    Plan,
    anthropic_body,
    byte_backend,
    chat_body,
    make_frontend,
    next_sse_data,
    response_events,
    responses_body,
    rich_weather_tool,
)
from server import backend as backend_api
from server import output as model_output
from server import server as api


class ToolCallTests(HarnessTestCase):
    def test_tools_history_choice_and_output(self):
        runtime = FakeRuntime(Plan([[5]]))
        harness = self.harness(runtime)
        messages = [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Rome"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old", "content": "sunny"},
            {"role": "user", "content": "again"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            },
            {"type": "function", "function": {"name": "time", "parameters": {}}},
        ]
        choice = {"type": "function", "function": {"name": "weather"}}
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(
                messages=messages,
                tools=tools,
                tool_choice=choice,
                reasoning_effort="none",
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
        call = response["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(
            call["function"], {"name": "weather", "arguments": '{"city":"Paris"}'}
        )
        rendered_messages, kwargs = harness.tokenizer.templates[-1]
        self.assertEqual(
            rendered_messages[0], {"role": "system", "content": "Be concise."}
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in kwargs["tools"]], ["weather", "time"]
        )
        self.assertEqual(
            rendered_messages[2]["tool_calls"][0]["function"]["arguments"],
            {"city": "Rome"},
        )
        self.assertFalse(kwargs["enable_thinking"])

    def test_streaming_tool_call_has_index(self):
        harness = self.harness(FakeRuntime(Plan([[5]])))
        tools = [{"type": "function", "function": {"name": "weather"}}]
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(
                stream=True,
                tools=tools,
                reasoning_effort="none",
            ),
        )
        self.assertEqual(status, 200)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        tool_deltas = [
            call
            for chunk in chunks
            for call in chunk["choices"][0]["delta"].get("tool_calls", [])
        ]
        header = tool_deltas[0]
        self.assertEqual(header["index"], 0)
        self.assertTrue(header["id"].startswith("call_"))
        self.assertEqual(header["type"], "function")
        self.assertEqual(header["function"], {"name": "weather"})
        self.assertEqual(
            "".join(
                delta["function"].get("arguments", "") for delta in tool_deltas[1:]
            ),
            '{"city":"Paris"}',
        )

    def test_streaming_tool_call_arrives_before_native_done(self):
        plan = Plan([[13]], reason="length", after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            connection, response = harness.open_stream(
                "/v1/chat/completions",
                chat_body(
                    stream=True,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                    "additionalProperties": False,
                                },
                            },
                        }
                    ],
                    reasoning_effort="none",
                ),
            )
            try:
                role = json.loads(next_sse_data(response))
                self.assertEqual(role["choices"][0]["delta"]["role"], "assistant")
                header = json.loads(next_sse_data(response))
                call = header["choices"][0]["delta"]["tool_calls"][0]
                self.assertEqual(call["function"]["name"], "weather")
                arguments = ""
                while "Par" not in arguments:
                    chunk = json.loads(next_sse_data(response))
                    delta = chunk["choices"][0]["delta"]["tool_calls"][0]
                    arguments += delta["function"].get("arguments", "")
                self.assertEqual(arguments, '{"city":"Par')
                self.assertTrue(plan.terminal.wait(1))
                self.assertFalse(plan.terminal_release.is_set())
            finally:
                plan.terminal_release.set()
                response.read()
                connection.close()

    def test_responses_tool_arguments_arrive_before_native_done(self):
        plan = Plan([[13]], reason="length", after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            connection, response = harness.open_stream(
                "/v1/responses",
                responses_body(
                    stream=True,
                    tools=[tool],
                    reasoning={"effort": "none"},
                ),
            )
            arguments = ""
            try:
                while "Par" not in arguments:
                    event = json.loads(next_sse_data(response))
                    if event["type"] == "response.function_call_arguments.delta":
                        arguments += event["delta"]
                self.assertEqual(arguments, '{"city":"Par')
                self.assertTrue(plan.terminal.wait(1))
                self.assertFalse(plan.terminal_release.is_set())
            finally:
                plan.terminal_release.set()
                response.read()
                connection.close()

    def test_anthropic_tool_arguments_arrive_before_native_done(self):
        plan = Plan([[13]], reason="length", after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        tool = {
            "name": "weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            connection, response = harness.open_stream(
                "/v1/messages",
                anthropic_body(stream=True, tools=[tool]),
            )
            arguments = ""
            try:
                while "Par" not in arguments:
                    event = json.loads(next_sse_data(response))
                    if (
                        event["type"] == "content_block_delta"
                        and event["delta"]["type"] == "input_json_delta"
                    ):
                        arguments += event["delta"]["partial_json"]
                self.assertEqual(arguments, '{"city":"Par')
                self.assertTrue(plan.terminal.wait(1))
                self.assertFalse(plan.terminal_release.is_set())
            finally:
                plan.terminal_release.set()
                response.read()
                connection.close()

    def test_openai_sdk_tool_streams_match_nonstream_canonical_arguments(self):
        value = "x" * (model_output.TOOL_ARGUMENT_DELTA_CHARS * 2 + 137)
        tokenizer = FakeTokenizer()
        tokenizer.fragments[25] = (
            "<tool_call>\n<function=echo>\n"
            f"<parameter=text>\n{value}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        tokenizer.backend_tokenizer = byte_backend(tokenizer.fragments)
        harness = self.harness(
            FakeRuntime(*(Plan([[25]]) for _ in range(4))), tokenizer=tokenizer
        )
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
        chat_tools = [
            {
                "type": "function",
                "function": {"name": "echo", "parameters": schema},
            }
        ]
        responses_tools = [{"type": "function", "name": "echo", "parameters": schema}]
        host, port = harness.server.server_address
        with OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="test",
            timeout=5,
            max_retries=0,
        ) as client:
            chat = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                tools=chat_tools,
                extra_body={"reasoning_effort": "none"},
            )
            canonical_chat = chat.choices[0].message.tool_calls[0]
            chat_deltas = []
            finish_reasons = []
            for chunk in client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                tools=chat_tools,
                stream=True,
                extra_body={"reasoning_effort": "none"},
            ):
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reasons.append(choice.finish_reason)
                chat_deltas.extend(choice.delta.tool_calls or [])

            header = chat_deltas[0]
            argument_deltas = [
                delta.function.arguments
                for delta in chat_deltas[1:]
                if delta.function and delta.function.arguments is not None
            ]
            self.assertEqual((header.index, header.type), (0, "function"))
            self.assertTrue(header.id.startswith("call_"))
            self.assertEqual(header.function.name, "echo")
            self.assertIsNone(header.function.arguments)
            self.assertGreater(len(argument_deltas), 1)
            self.assertLessEqual(
                max(map(len, argument_deltas)), model_output.TOOL_ARGUMENT_DELTA_CHARS
            )
            self.assertEqual(
                "".join(argument_deltas), canonical_chat.function.arguments
            )
            self.assertEqual(finish_reasons[-1], "tool_calls")

            response = client.responses.create(
                model="test-model",
                input="hello",
                tools=responses_tools,
                reasoning={"effort": "none"},
            )
            canonical_response = next(
                item for item in response.output if item.type == "function_call"
            )
            response_events = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    tools=responses_tools,
                    reasoning={"effort": "none"},
                    stream=True,
                )
            )

        added = next(
            event.item
            for event in response_events
            if event.type == "response.output_item.added"
            and event.item.type == "function_call"
        )
        response_deltas = [
            event.delta
            for event in response_events
            if event.type == "response.function_call_arguments.delta"
        ]
        arguments_done = next(
            event
            for event in response_events
            if event.type == "response.function_call_arguments.done"
        )
        item_done = next(
            event.item
            for event in response_events
            if event.type == "response.output_item.done"
            and event.item.type == "function_call"
        )
        self.assertEqual((added.name, added.arguments), ("echo", ""))
        self.assertEqual(added.id, item_done.id)
        self.assertGreater(len(response_deltas), 1)
        self.assertLessEqual(
            max(map(len, response_deltas)), model_output.TOOL_ARGUMENT_DELTA_CHARS
        )
        self.assertEqual("".join(response_deltas), canonical_response.arguments)
        self.assertEqual(arguments_done.item_id, item_done.id)
        self.assertEqual(arguments_done.name, item_done.name)
        self.assertEqual(arguments_done.arguments, canonical_response.arguments)
        self.assertEqual(item_done.arguments, canonical_response.arguments)
        self.assertEqual(response_events[-1].type, "response.completed")

    def test_tool_capable_chat_streams_text_before_done_without_repeating(self):
        plan = Plan([[14], [15]], delay=0.01, after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        tools = [{"type": "function", "function": {"name": "weather"}}]
        started = time.monotonic()
        connection, response = harness.open_stream(
            "/v1/chat/completions",
            chat_body(
                stream=True,
                tools=tools,
                reasoning_effort="none",
            ),
        )
        parts = []
        try:
            self.assertEqual(response.status, 200)
            while not parts:
                data = next_sse_data(response)
                self.assertIsNotNone(data)
                self.assertNotEqual(data, "[DONE]")
                chunk = json.loads(data)
                if chunk["choices"]:
                    text = chunk["choices"][0]["delta"].get("content")
                    if text:
                        parts.append(text)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(plan.terminal.wait(1))
            self.assertFalse(plan.terminal_release.is_set())
            plan.terminal_release.set()
            while True:
                data = next_sse_data(response)
                self.assertIsNotNone(data)
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk["choices"]:
                    text = chunk["choices"][0]["delta"].get("content")
                    if text:
                        parts.append(text)
        finally:
            plan.terminal_release.set()
            response.close()
            connection.close()
        self.assertEqual(parts, ["first ", "second\n"])

    def test_tool_capable_responses_streams_text_before_done_without_repeating(self):
        plan = Plan([[14], [15]], delay=0.01, after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        tools = [{"type": "function", "name": "weather"}]
        started = time.monotonic()
        connection, response = harness.open_stream(
            "/v1/responses",
            responses_body(
                stream=True,
                tools=tools,
                reasoning={"effort": "none"},
            ),
        )
        parts, completed = [], None
        try:
            self.assertEqual(response.status, 200)
            while not parts:
                data = next_sse_data(response)
                self.assertIsNotNone(data)
                event = json.loads(data)
                if event["type"] == "response.output_text.delta":
                    parts.append(event["delta"])
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(plan.terminal.wait(1))
            self.assertFalse(plan.terminal_release.is_set())
            plan.terminal_release.set()
            while completed is None:
                data = next_sse_data(response)
                self.assertIsNotNone(data)
                event = json.loads(data)
                if event["type"] == "response.output_text.delta":
                    parts.append(event["delta"])
                if event["type"] == "response.completed":
                    completed = event["response"]
        finally:
            plan.terminal_release.set()
            response.close()
            connection.close()
        self.assertEqual(parts, ["first ", "second\n"])
        message = next(
            item for item in completed["output"] if item["type"] == "message"
        )
        self.assertEqual(message["content"][0]["text"], "first second\n")

    def test_streaming_multiple_tools_never_exposes_xml(self):
        harness = self.harness(FakeRuntime(Plan([[5], [16]])))
        tools = [
            {"type": "function", "function": {"name": "weather"}},
            {"type": "function", "function": {"name": "time"}},
        ]
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(
                stream=True,
                tools=tools,
                reasoning_effort="none",
            ),
        )
        self.assertEqual(status, 200, payload)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
        self.assertEqual("".join(delta.get("content", "") for delta in deltas), "")
        calls = [
            call
            for delta in deltas
            for call in delta.get("tool_calls", [])
            if "name" in call["function"]
        ]
        self.assertEqual(
            [call["function"]["name"] for call in calls], ["weather", "time"]
        )
        self.assertNotIn("<tool_call>", payload.decode())

    def test_streaming_tool_tags_out_of_the_template_layout_stay_text(self):
        # Calls read as the chat template lays them out; tags written without
        # its newlines are text, which streams as written.
        harness = self.harness(FakeRuntime(Plan([[14], [8]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(
                stream=True,
                tools=[rich_weather_tool()],
                reasoning_effort="none",
            ),
        )
        self.assertEqual(status, 200, payload)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        deltas = [
            chunk["choices"][0]["delta"] for chunk in chunks if chunk.get("choices")
        ]
        self.assertEqual(
            "".join(delta.get("content") or "" for delta in deltas),
            "first " + FakeTokenizer().fragments[8],
        )
        self.assertFalse(any(delta.get("tool_calls") for delta in deltas))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_prose_beside_a_call_can_name_tool_tags(self):
        # Only <tool_call> opens a call, as in the grammars' text: other tags in
        # prose, <function= among them, stay text.
        tokenizer = FakeTokenizer()
        tokenizer.fragments[40] = "Fix the </parameter> and <function= handling.\n"
        tokenizer.backend_tokenizer = byte_backend(tokenizer.fragments)
        tools = [{"type": "function", "function": {"name": "weather"}}]
        prose = tokenizer.fragments[40]
        for stream in (False, True):
            with self.subTest(stream=stream):
                harness = self.harness(
                    FakeRuntime(Plan([[40], [5]])), tokenizer=tokenizer
                )
                status, _, payload = harness.request(
                    "POST",
                    "/v1/chat/completions",
                    chat_body(tools=tools, stream=stream, reasoning_effort="none"),
                )
                self.assertEqual(status, 200, payload)
                if stream:
                    chunks = [
                        json.loads(line[6:])
                        for line in payload.decode().splitlines()
                        if line.startswith("data: {")
                    ]
                    self.assertFalse(any("error" in chunk for chunk in chunks))
                    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
                    content = "".join(delta.get("content") or "" for delta in deltas)
                    names = [
                        call["function"]["name"]
                        for delta in deltas
                        for call in delta.get("tool_calls", [])
                        if "name" in call["function"]
                    ]
                    reason = chunks[-1]["choices"][0]["finish_reason"]
                else:
                    choice = json.loads(payload)["choices"][0]
                    content = choice["message"]["content"]
                    names = [
                        call["function"]["name"]
                        for call in choice["message"]["tool_calls"]
                    ]
                    reason = choice["finish_reason"]
                self.assertEqual(
                    (content, names, reason), (prose, ["weather"], "tool_calls")
                )

    def test_stop_is_refused_only_while_a_tool_can_be_called(self):
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(
            FakeRuntime(), tokenizer, lambda _record: None
        )
        self.addCleanup(backend.close)
        app = make_frontend(tokenizer, backend, "test-model", 128, 1, 2, vision=True)
        tools = [{"type": "function", "function": {"name": "f"}}]
        job = app.prepare(
            chat_body(tools=tools, tool_choice="none", stop=["x"]), deadline=FOREVER
        )
        self.assertEqual(job.tool_policy.schemas, {})
        for choice in ("auto", "required"):
            with (
                self.subTest(tool_choice=choice),
                self.assertRaisesRegex(api.APIError, "stop cannot be combined"),
            ):
                app.prepare(
                    chat_body(tools=tools, tool_choice=choice, stop=["x"]),
                    deadline=FOREVER,
                )

    def test_tool_names_accept_long_mcp_names_up_to_128_characters(self):
        runtime = FakeRuntime(Plan([[4]]))
        harness = self.harness(runtime)
        name = (
            "mcp__github_enterprise_server__list_pull_request_review_comments_for_repos"
        )
        self.assertEqual(len(name), 74)
        status, _, _ = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(
                tools=[{"type": "function", "function": {"name": name}}],
                reasoning_effort="none",
            ),
        )
        self.assertEqual(status, 200)
        _, kwargs = harness.tokenizer.templates[-1]
        self.assertEqual(kwargs["tools"][0]["function"]["name"], name)
        cases = (
            (
                {"type": "function", "function": {"name": "mcp__" + "a" * 124}},
                "tool name must match [A-Za-z0-9_-]{1,128}",
            ),
            ({"type": "web_search"}, "only function tools are supported"),
        )
        for tool, message in cases:
            with self.subTest(message=message):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(tools=[tool])
                )
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(payload)["error"]["message"], message)
        self.assertEqual(len(runtime.requests), 1)

    def test_pydantic_style_schema_reads_values_through_references(self):
        # The parameter's types come through its local reference, and a value
        # outside its enum still comes back.
        tool = rich_weather_tool()
        harness = self.harness(FakeRuntime(Plan([[28]]), Plan([[5]])))
        for city in ("3", "Paris"):
            status, _, payload = harness.request(
                "POST",
                "/v1/chat/completions",
                chat_body(tools=[tool], reasoning_effort="none"),
            )
            self.assertEqual(status, 200, payload)
            call = json.loads(payload)["choices"][0]["message"]["tool_calls"][0]
            self.assertEqual(json.loads(call["function"]["arguments"]), {"city": city})

    def test_required_named_and_parallel_tool_policies(self):
        # Generation enforces the choice; what the model wrote is read back as
        # it is.
        tools = [
            {"type": "function", "function": {"name": "weather"}},
            {"type": "function", "function": {"name": "time"}},
        ]
        named = {"type": "function", "function": {"name": "weather"}}
        cases = (
            ([4], {"tool_choice": "required"}, []),
            ([16], {"tool_choice": named}, ["time"]),
            ([5, 16], {"parallel_tool_calls": False}, ["weather", "time"]),
        )
        for batch, extra, names in cases:
            with self.subTest(**extra):
                factory = FakeConstraintFactory()
                harness = self.harness(
                    FakeRuntime(Plan([batch])), constraint_factory=factory
                )
                status, _, payload = harness.request(
                    "POST",
                    "/v1/chat/completions",
                    chat_body(tools=tools, reasoning_effort="none", **extra),
                )
                self.assertEqual(status, 200, payload)
                self.assertEqual(len(factory.grammars), 1)
                message = json.loads(payload)["choices"][0]["message"]
                self.assertEqual(
                    [
                        call["function"]["name"]
                        for call in message.get("tool_calls", [])
                    ],
                    names,
                )

    def test_parallel_calls_stream_as_the_model_writes_them(self):
        # No call is checked against its tool's schema: a value its types
        # cannot read keeps its text.
        tokenizer = FakeTokenizer()
        tokenizer.fragments[25] = (
            "<tool_call>\n<function=weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>"
            "<tool_call>\n<function=echo>\n"
            "<parameter=count>\ninvalid\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        tokenizer.backend_tokenizer = byte_backend(tokenizer.fragments)
        schemas = {
            "weather": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
            },
            "echo": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        }
        expected = [("weather", {"city": "Paris"}), ("echo", {"count": "invalid"})]
        harness = self.harness(
            FakeRuntime(Plan([[25]]), Plan([[25]])), tokenizer=tokenizer
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(
                stream=True,
                tools=[
                    {"type": "function", "function": {"name": name, "parameters": s}}
                    for name, s in schemas.items()
                ],
                reasoning_effort="none",
            ),
        )
        self.assertEqual(status, 200, payload)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        calls = {}
        for chunk in chunks:
            for choice in chunk.get("choices", []):
                for call in choice["delta"].get("tool_calls", []):
                    entry = calls.setdefault(call["index"], ["", ""])
                    entry[0] += call["function"].get("name", "")
                    entry[1] += call["function"].get("arguments", "")
        self.assertEqual(
            [(name, json.loads(arguments)) for name, arguments in calls.values()],
            expected,
        )
        self.assertNotIn("<tool_call>", payload.decode())
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                stream=True,
                tools=[
                    {"type": "function", "name": name, "parameters": s}
                    for name, s in schemas.items()
                ],
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200, payload)
        events = response_events(payload)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(
            [
                (item["name"], json.loads(item["arguments"]))
                for item in events[-1]["response"]["output"]
            ],
            expected,
        )

    def test_tool_constraint_requires_object_parameters(self):
        # Where a strict tool's grammar frames its arguments; any tool's name
        # must fit the template either way.
        factory = FakeConstraintFactory()
        harness = self.harness(FakeRuntime(), constraint_factory=factory)
        cases = (
            ({"parameters": {"type": "string"}}, "top-level JSON object"),
            (
                {"parameters": {"type": "object", "$ref": "#/$defs/missing"}},
                "unresolved tool parameter reference",
            ),
            (
                {
                    "parameters": {
                        "type": "object",
                        "properties": {" leading": {"type": "string"}},
                    }
                },
                "invalid tool parameter name",
            ),
            (
                {
                    "parameters": {
                        "type": "object",
                        "properties": {"n" * 257: {"type": "string"}},
                    }
                },
                "invalid tool parameter name",
            ),
        )
        for function, message in cases:
            for strict in (False, True):
                tool = {
                    "type": "function",
                    "function": {"name": "bad", **function, "strict": strict},
                }
                with self.subTest(message=message, strict=strict):
                    status, _, payload = harness.request(
                        "POST", "/v1/chat/completions", chat_body(tools=[tool])
                    )
                    if not strict:
                        self.assertEqual(status, 200, payload)
                        continue
                    self.assertEqual(status, 400)
                    self.assertIn(message, json.loads(payload)["error"]["message"])
        tool = {"type": "function", "function": {"name": "bad>name", "parameters": {}}}
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(tools=[tool])
        )
        self.assertEqual(status, 400)
        self.assertIn("tool name must match", json.loads(payload)["error"]["message"])
        self.assertEqual(factory.grammars, [])


if __name__ == "__main__":
    unittest.main()
