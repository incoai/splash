import json
import unittest

from dev.tests.server_fixtures import (
    FakeConstraintFactory,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    Plan,
    openai_client,
    response_events,
    responses_body,
    rich_weather_tool,
)
from server import api_shapes
from server.api_shapes import _namespace_alias, normalize_responses_input


class ResponsesTests(HarnessTestCase):
    def test_responses_nonstream_contract_and_reasoning(self):
        runtime = FakeRuntime(Plan([[1], [2], [3]]))
        harness = self.harness(runtime)
        body = responses_body(
            instructions="Be concise.",
            tool_choice="auto",
            parallel_tool_calls=True,
            reasoning={"effort": "medium", "summary": "auto", "context": "auto"},
            store=False,
            stream=False,
            stream_options={"reasoning_summary_delivery": "sequential_cutoff"},
            include=["reasoning.encrypted_content"],
            service_tier="auto",
            prompt_cache_key="turn-1",
            text={"verbosity": "low"},
            client_metadata={"origin": "test-client"},
            unknown_option={"future": True},
            max_output_tokens=8,
            seed=7,
        )
        status, content_type, payload = harness.request("POST", "/v1/responses", body)
        self.assertEqual((status, content_type), (200, "application/json"))
        response = json.loads(payload)
        self.assertTrue(response["id"].startswith("resp_"))
        self.assertEqual(
            (response["object"], response["status"]), ("response", "completed")
        )
        self.assertTrue(response["end_turn"])
        self.assertEqual(
            [item["type"] for item in response["output"]],
            ["reasoning", "message"],
        )
        self.assertEqual(response["output"][0]["summary"][0]["text"], "because ")
        self.assertEqual(response["output"][0]["content"][0]["text"], "because ")
        self.assertEqual(response["output"][1]["content"][0]["text"], "answer\n")
        self.assertEqual(
            response["usage"],
            {
                "input_tokens": 2,
                "input_tokens_details": {
                    "cached_tokens": 1,
                    "cache_write_tokens": 1,
                },
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 5,
            },
        )
        rendered, kwargs = harness.tokenizer.templates[-1]
        self.assertEqual(rendered[0], {"role": "system", "content": "Be concise."})
        self.assertEqual(rendered[1]["content"], "hello")
        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertEqual(
            (
                runtime.requests[0].frame.logical_max_output_tokens,
                runtime.requests[0].frame.seed,
            ),
            (8, 7),
        )

    def test_responses_stream_emits_protocol_events_and_usage(self):
        harness = self.harness(FakeRuntime(Plan([[1], [4]], reason="length")))
        status, content_type, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                stream=True,
                reasoning={"effort": "none"},
                store=False,
            ),
        )
        self.assertEqual((status, content_type), (200, "text/event-stream"))
        self.assertNotIn(b"[DONE]", payload)
        events = response_events(payload)
        kinds = [event["type"] for event in events]
        self.assertEqual(
            kinds,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.incomplete",
            ],
        )
        self.assertEqual(
            [
                event["delta"]
                for event in events
                if event["type"] == "response.output_text.delta"
            ],
            ["because ", "plain answer\n"],
        )
        incomplete = events[-1]["response"]
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertEqual(
            incomplete["incomplete_details"], {"reason": "max_output_tokens"}
        )
        self.assertFalse(incomplete["end_turn"])
        self.assertEqual(incomplete["usage"]["output_tokens"], 2)
        message = next(
            item for item in incomplete["output"] if item["type"] == "message"
        )
        self.assertEqual(message["status"], "incomplete")
        self.assertEqual(events[0]["response"]["id"], incomplete["id"])
        added = next(
            event["item"]
            for event in events
            if event["type"] == "response.output_item.added"
        )
        done = next(
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
        )
        self.assertEqual(added["id"], done["id"])
        self.assertEqual(added["content"], [])
        text_done = next(
            event for event in events if event["type"] == "response.output_text.done"
        )
        part_done = next(
            event for event in events if event["type"] == "response.content_part.done"
        )
        self.assertEqual(text_done["text"], "because plain answer\n")
        self.assertEqual(text_done["logprobs"], [])
        self.assertEqual(part_done["part"], done["content"][0])
        self.assertEqual(
            {
                (event.get("item_id"), event.get("output_index"))
                for event in events
                if "item_id" in event
            },
            {(done["id"], 0)},
        )
        self.assertEqual(
            [event["sequence_number"] for event in events], list(range(len(events)))
        )

    def test_responses_nonstream_length_is_incomplete_even_for_partial_json(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        harness = self.harness(
            FakeRuntime(Plan([[12]], reason="length")),
            constraint_factory=FakeConstraintFactory(),
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                reasoning={"effort": "none"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "answer",
                        "schema": schema,
                        "strict": True,
                    }
                },
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(
            response["incomplete_details"], {"reason": "max_output_tokens"}
        )
        self.assertFalse(response["end_turn"])
        self.assertEqual(response["output"][0]["status"], "incomplete")
        self.assertEqual(response["output"][0]["content"][0]["text"], '{"x":')

    def test_responses_length_marks_a_streamed_tool_call_incomplete(self):
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        harness = self.harness(
            FakeRuntime(Plan([[5]], reason="length"), Plan([[5]], reason="length"))
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(
            [item["type"] for item in response["output"]], ["function_call"]
        )
        self.assertEqual(response["output"][0]["status"], "incomplete")
        self.assertNotIn("<tool_call>", payload.decode())

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                stream=True,
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        events = response_events(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertNotIn("<tool_call>", payload.decode())
        calls = [
            event["item"]
            for event in events
            if event.get("item", {}).get("type") == "function_call"
        ]
        self.assertTrue(calls)
        self.assertEqual(calls[-1]["status"], "incomplete")
        self.assertEqual(
            [item["type"] for item in events[-1]["response"]["output"]],
            ["function_call"],
        )

    def test_responses_streams_reasoning_summary_before_answer(self):
        harness = self.harness(FakeRuntime(Plan([[1], [2], [3]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                stream=True,
                reasoning={"effort": "xhigh", "summary": "auto"},
            ),
        )
        self.assertEqual(status, 200)
        events = response_events(payload)
        kinds = [event["type"] for event in events]
        self.assertEqual(
            kinds,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        summary_delta = next(
            event
            for event in events
            if event["type"] == "response.reasoning_summary_text.delta"
        )
        self.assertEqual(summary_delta["delta"], "because ")
        self.assertLess(
            kinds.index("response.reasoning_summary_text.delta"),
            kinds.index("response.output_text.delta"),
        )
        self.assertNotIn("response.reasoning_text.delta", kinds)
        self.assertNotIn("response.reasoning_text.done", kinds)
        summary_done = events[5]
        self.assertEqual(summary_done["text"], "because ")
        self.assertEqual(
            events[6]["part"], {"type": "summary_text", "text": "because "}
        )
        reasoning_id = events[2]["item"]["id"]
        message_id = events[8]["item"]["id"]
        self.assertEqual(
            [(event["output_index"], event.get("item_id")) for event in events[2:8]],
            [
                (0, None),
                (0, reasoning_id),
                (0, reasoning_id),
                (0, reasoning_id),
                (0, reasoning_id),
                (0, None),
            ],
        )
        self.assertEqual(
            [(event["output_index"], event.get("item_id")) for event in events[8:14]],
            [
                (1, None),
                (1, message_id),
                (1, message_id),
                (1, message_id),
                (1, message_id),
                (1, None),
            ],
        )
        completed = events[-1]["response"]
        self.assertEqual(
            [item["type"] for item in completed["output"]],
            ["reasoning", "message"],
        )
        self.assertEqual(completed["output"][0]["summary"][0]["text"], "because ")
        self.assertEqual(completed["output"][1]["content"][0]["text"], "answer\n")
        self.assertEqual(
            [event["sequence_number"] for event in events], list(range(len(events)))
        )
        self.assertEqual(
            {events[0]["response"]["id"], events[1]["response"]["id"], completed["id"]},
            {completed["id"]},
        )

    def test_responses_reasoning_length_marks_item_and_part_incomplete(self):
        harness = self.harness(FakeRuntime(Plan([[1]], reason="length")))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                stream=True,
                reasoning={"effort": "xhigh", "summary": "auto"},
            ),
        )
        self.assertEqual(status, 200, payload)
        events = response_events(payload)
        part_done = next(
            event
            for event in events
            if event["type"] == "response.reasoning_summary_part.done"
        )
        item_done = next(
            event
            for event in events
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "reasoning"
        )
        terminal = events[-1]
        self.assertEqual(part_done["status"], "incomplete")
        self.assertEqual(item_done["item"]["status"], "incomplete")
        self.assertEqual(terminal["type"], "response.incomplete")
        self.assertEqual(terminal["response"]["output"][0]["status"], "incomplete")

    def test_responses_reasoning_status_tracks_the_truncated_phase(self):
        answer_harness = self.harness(
            FakeRuntime(
                Plan([[1], [2], [3]], reason="length"),
                Plan([[1], [2], [3]], reason="length"),
            )
        )
        for stream in (False, True):
            status, _, payload = answer_harness.request(
                "POST",
                "/v1/responses",
                responses_body(
                    stream=stream,
                    reasoning={"effort": "xhigh", "summary": "auto"},
                ),
            )
            self.assertEqual(status, 200, payload)
            response = (
                response_events(payload)[-1]["response"]
                if stream
                else json.loads(payload)
            )
            self.assertEqual(response["status"], "incomplete")
            self.assertEqual(
                [(item["type"], item["status"]) for item in response["output"]],
                [("reasoning", "completed"), ("message", "incomplete")],
            )

        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        tool_harness = self.harness(
            FakeRuntime(
                Plan([[1], [2], [25], [13]], reason="length"),
                Plan([[1], [2], [25], [13]], reason="length"),
            )
        )
        for stream in (False, True):
            status, _, payload = tool_harness.request(
                "POST",
                "/v1/responses",
                responses_body(
                    stream=stream,
                    tools=[tool],
                    reasoning={"effort": "xhigh", "summary": "auto"},
                ),
            )
            self.assertEqual(status, 200, payload)
            response = (
                response_events(payload)[-1]["response"]
                if stream
                else json.loads(payload)
            )
            expected_items = [
                ("reasoning", "completed"),
                ("function_call", "incomplete"),
            ]
            self.assertEqual(
                [(item["type"], item["status"]) for item in response["output"]],
                expected_items,
            )
            self.assertNotIn("<tool_call>", payload.decode())

    def test_openai_sdk_parses_responses_reasoning_lifecycle(self):
        harness = self.harness(FakeRuntime(Plan([[1], [2], [3]])))
        with openai_client(harness) as client:
            events = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "medium", "summary": "auto"},
                )
            )
        self.assertEqual(
            [type(event).__name__ for event in events],
            [
                "ResponseCreatedEvent",
                "ResponseInProgressEvent",
                "ResponseOutputItemAddedEvent",
                "ResponseReasoningSummaryPartAddedEvent",
                "ResponseReasoningSummaryTextDeltaEvent",
                "ResponseReasoningSummaryTextDoneEvent",
                "ResponseReasoningSummaryPartDoneEvent",
                "ResponseOutputItemDoneEvent",
                "ResponseOutputItemAddedEvent",
                "ResponseContentPartAddedEvent",
                "ResponseTextDeltaEvent",
                "ResponseTextDoneEvent",
                "ResponseContentPartDoneEvent",
                "ResponseOutputItemDoneEvent",
                "ResponseCompletedEvent",
            ],
        )
        self.assertEqual(
            [event.sequence_number for event in events], list(range(len(events)))
        )
        self.assertEqual(events[5].text, "because ")
        self.assertEqual(events[11].text, "answer\n")
        self.assertIsNone(events[6].status)
        self.assertEqual(events[7].item.status, "completed")
        self.assertEqual(events[-1].response.status, "completed")
        self.assertEqual(
            events[-1].response.usage.output_tokens_details.reasoning_tokens, 3
        )
        self.assertEqual(
            events[-1].response.usage.input_tokens_details.cached_tokens, 1
        )
        self.assertEqual(
            events[0].response.id,
            events[1].response.id,
        )
        self.assertEqual(events[1].response.id, events[-1].response.id)

        incomplete_harness = self.harness(FakeRuntime(Plan([[4]], reason="length")))
        with openai_client(incomplete_harness) as client:
            incomplete = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "none"},
                )
            )[-1]
        self.assertEqual(type(incomplete).__name__, "ResponseIncompleteEvent")
        self.assertEqual(incomplete.response.status, "incomplete")
        self.assertEqual(
            incomplete.response.incomplete_details.reason, "max_output_tokens"
        )

        reasoning_harness = self.harness(FakeRuntime(Plan([[1]], reason="length")))
        with openai_client(reasoning_harness) as client:
            reasoning_events = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "xhigh", "summary": "auto"},
                )
            )
        part_done = next(
            event
            for event in reasoning_events
            if event.type == "response.reasoning_summary_part.done"
        )
        item_done = next(
            event
            for event in reasoning_events
            if event.type == "response.output_item.done"
            and event.item.type == "reasoning"
        )
        self.assertEqual(part_done.status, "incomplete")
        self.assertEqual(item_done.item.status, "incomplete")
        self.assertEqual(reasoning_events[-1].response.status, "incomplete")

        phase_harness = self.harness(
            FakeRuntime(Plan([[1], [2], [3]], reason="length"))
        )
        with openai_client(phase_harness) as client:
            phase_events = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "xhigh", "summary": "auto"},
                )
            )
        self.assertEqual(
            [(item.type, item.status) for item in phase_events[-1].response.output],
            [("reasoning", "completed"), ("message", "incomplete")],
        )

    def test_openai_sdk_parses_function_call_arguments_done(self):
        harness = self.harness(FakeRuntime(Plan([[5]])))
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        with openai_client(harness) as client:
            events = list(
                client.responses.create(
                    model="test-model",
                    input="weather",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "none"},
                    tools=[tool],
                )
            )
        event_types = [event.type for event in events]
        self.assertEqual(event_types[:2], ["response.created", "response.in_progress"])
        self.assertEqual(event_types[-1], "response.completed")
        self.assertLess(
            event_types.index("response.function_call_arguments.delta"),
            event_types.index("response.function_call_arguments.done"),
        )
        arguments_done = next(
            event
            for event in events
            if event.type == "response.function_call_arguments.done"
        )
        call_done = next(
            event.item
            for event in events
            if event.type == "response.output_item.done"
            and event.item.type == "function_call"
        )
        self.assertEqual(
            type(arguments_done).__name__, "ResponseFunctionCallArgumentsDoneEvent"
        )
        self.assertEqual(
            (
                arguments_done.item_id,
                arguments_done.output_index,
                arguments_done.name,
                arguments_done.arguments,
            ),
            (call_done.id, 0, "weather", '{"city":"Paris"}'),
        )
        self.assertEqual(events[-1].response.status, "completed")

        # A value outside the tool's schema reads back as the model wrote it.
        outside_harness = self.harness(FakeRuntime(Plan([[28]])))
        with openai_client(outside_harness) as client:
            outside = list(
                client.responses.create(
                    model="test-model",
                    input="weather",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "none"},
                    tools=[tool],
                )
            )[-1]
        self.assertEqual(type(outside).__name__, "ResponseCompletedEvent")
        self.assertEqual(outside.response.output[0].arguments, '{"city":"3"}')

    def test_responses_tool_history_named_choice_and_output(self):
        runtime = FakeRuntime(Plan([[5]]))
        harness = self.harness(runtime)
        history = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "weather"}],
                "phase": "final_answer",
            },
            {
                "type": "reasoning",
                "id": "rs_old",
                "summary": [{"type": "summary_text", "text": "Use weather."}],
                "encrypted_content": None,
            },
            {
                "type": "function_call",
                "id": "fc_old",
                "call_id": "call_old",
                "name": "weather",
                "arguments": '{"city":"Rome"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_old",
                "output": [{"type": "input_text", "text": "sunny"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "again"}],
            },
        ]
        tools = [
            {
                "type": "function",
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "strict": True,
            },
            {"type": "function", "name": "time", "parameters": {}},
        ]
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                input=history,
                tools=tools,
                tool_choice={"type": "function", "name": "weather"},
                parallel_tool_calls=False,
                reasoning={"effort": "none"},
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertFalse(response["end_turn"])
        self.assertFalse(response["parallel_tool_calls"])
        call = response["output"][-1]
        self.assertEqual(call["type"], "function_call")
        self.assertTrue(call["id"].startswith("fc_"))
        self.assertTrue(call["call_id"].startswith("call_"))
        self.assertEqual(
            (call["name"], call["arguments"]),
            ("weather", '{"city":"Paris"}'),
        )
        rendered, kwargs = harness.tokenizer.templates[-1]
        self.assertEqual(rendered[0], {"role": "user", "content": "weather"})
        self.assertEqual(
            [tool["function"]["name"] for tool in kwargs["tools"]], ["weather", "time"]
        )
        self.assertEqual(rendered[1]["reasoning_content"], "Use weather.")
        self.assertEqual(rendered[2]["tool_call_id"], "call_old")
        self.assertEqual(rendered[2]["content"], "sunny")
        self.assertEqual(kwargs["tools"][0]["function"]["name"], "weather")
        self.assertTrue(kwargs["tools"][0]["function"]["strict"])

    def test_responses_accepts_shorthand_messages_and_retained_history(self):
        harness = self.harness(FakeRuntime())
        with openai_client(harness) as client:
            first = client.responses.create(
                model="test-model",
                input=[{"role": "user", "content": "hello"}],
                reasoning={"effort": "none"},
            )
            second = client.responses.create(
                model="test-model",
                previous_response_id=first.id,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "again"}],
                    }
                ],
                reasoning={"effort": "none"},
            )
        self.assertEqual((first.status, second.status), ("completed", "completed"))
        messages, _ = harness.tokenizer.templates[-1]
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages],
            [("user", "hello"), ("assistant", "plain answer\n"), ("user", "again")],
        )

    def test_responses_rejects_hosted_tools_without_native_work(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for tool in (
            {"type": "web_search"},
            {"type": "web_search", "external_web_access": False},
            {"type": "web_search", "external_web_access": True},
            {"type": "web_search_preview"},
            {"type": "file_search"},
        ):
            for stream in (False, True):
                with self.subTest(tool=tool, stream=stream):
                    status, _, payload = harness.request(
                        "POST",
                        "/v1/responses",
                        responses_body(tools=[tool], stream=stream),
                    )
                    self.assertEqual(status, 400, payload)
                    self.assertEqual(
                        json.loads(payload)["error"]["code"], "invalid_request_error"
                    )
        self.assertEqual(runtime.requests, [])
        self.assertEqual(harness.tokenizer.templates, [])

    def test_responses_supports_namespaces(self):
        tools = [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                        },
                    }
                ],
            },
        ]
        runtime = FakeRuntime(Plan([[1]]))
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                tools=tools,
                tool_choice="auto",
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200, payload)
        _, template = harness.tokenizer.templates[-1]
        self.assertEqual(
            [tool["function"]["name"] for tool in template["tools"]],
            ["exec_command", "multi_agent_v1__spawn_agent"],
        )

        before = len(runtime.requests)
        named = dict(responses_body())
        named["tools"] = tools
        named["tool_choice"] = {"type": "web_search"}
        status, _, payload = harness.request("POST", "/v1/responses", named)
        self.assertEqual(status, 400, payload)

        namespace_runtime = FakeRuntime(Plan([[11]]), Plan([[4]]))
        namespace_harness = self.harness(namespace_runtime)
        namespace_body = responses_body(
            stream=True,
            tools=tools,
            tool_choice="auto",
            reasoning={"effort": "none"},
        )
        status, _, payload = namespace_harness.request(
            "POST", "/v1/responses", namespace_body
        )
        self.assertEqual(status, 200, payload)
        events = response_events(payload)
        call = next(
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        )
        self.assertEqual(
            (call["namespace"], call["name"], call["arguments"]),
            ("multi_agent_v1", "spawn_agent", '{"message":"inspect"}'),
        )
        followup = dict(namespace_body)
        followup["stream"] = False
        followup["input"] = [
            *namespace_body["input"],
            call,
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": "done",
            },
        ]
        status, _, payload = namespace_harness.request(
            "POST", "/v1/responses", followup
        )
        self.assertEqual(status, 200, payload)
        rendered, _ = namespace_harness.tokenizer.templates[-1]
        self.assertEqual(
            rendered[-2]["tool_calls"][0]["function"]["name"],
            "multi_agent_v1__spawn_agent",
        )
        self.assertEqual(rendered[-1]["content"], "done")
        self.assertEqual(len(runtime.requests), before)

        bad = dict(responses_body())
        bad["tools"] = [{"type": "computer_use_preview"}]
        status, _, payload = harness.request("POST", "/v1/responses", bad)
        self.assertEqual(status, 400, payload)

    def test_responses_shortens_long_namespace_aliases_without_collisions(self):
        namespace = "n" * 64
        first_name = "same_prefix_" + "a" * 51
        second_name = "same_prefix_" + "a" * 50 + "b"
        tools = [
            {
                "type": "namespace",
                "name": namespace,
                "tools": [
                    {"type": "function", "name": first_name, "parameters": {}},
                    {"type": "function", "name": second_name, "parameters": {}},
                ],
            }
        ]
        body = responses_body(
            tools=tools,
            tool_choice={
                "type": "function",
                "namespace": namespace,
                "name": first_name,
            },
        )
        translated, namespaces = api_shapes.responses_to_chat_body(body, body["input"])
        aliases = [tool["function"]["name"] for tool in translated["tools"]]
        self.assertEqual([len(alias) for alias in aliases], [64, 64])
        self.assertNotEqual(aliases[0], aliases[1])
        self.assertEqual(
            translated["tool_choice"],
            {"type": "function", "function": {"name": aliases[0]}},
        )
        self.assertEqual(namespaces[aliases[0]], (namespace, first_name))
        self.assertEqual(_namespace_alias(namespace, first_name), aliases[0])

    def test_responses_streaming_tool_calls(self):
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
            },
        }
        harness = self.harness(FakeRuntime(Plan([[5]]), Plan([[28]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                stream=True,
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200)
        events = response_events(payload)
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[:2], ["response.created", "response.in_progress"])
        self.assertEqual(kinds[-1], "response.completed")
        added_index, added_output_index, added = next(
            (index, event["output_index"], event["item"])
            for index, event in enumerate(events)
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "function_call"
        )
        delta_indexes = [
            index
            for index, event in enumerate(events)
            if event["type"] == "response.function_call_arguments.delta"
        ]
        arguments_done_index, arguments_done = next(
            (index, event)
            for index, event in enumerate(events)
            if event["type"] == "response.function_call_arguments.done"
        )
        item_done_index, done = next(
            (index, event["item"])
            for index, event in enumerate(events)
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        )
        self.assertEqual(added["arguments"], "")
        self.assertEqual(
            (done["name"], done["arguments"]),
            ("weather", '{"city":"Paris"}'),
        )
        self.assertEqual(
            "".join(events[index]["delta"] for index in delta_indexes),
            done["arguments"],
        )
        self.assertEqual(
            (
                arguments_done["item_id"],
                arguments_done["output_index"],
                arguments_done["name"],
                arguments_done["arguments"],
            ),
            (
                done["id"],
                added_output_index,
                done["name"],
                done["arguments"],
            ),
        )
        self.assertEqual(added["id"], done["id"])
        self.assertTrue(added_index < delta_indexes[0] < arguments_done_index)
        self.assertLess(arguments_done_index, item_done_index)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertNotIn("<tool_call>", payload.decode())
        self.assertEqual(
            [item["type"] for item in events[-1]["response"]["output"]],
            ["function_call"],
        )
        self.assertEqual(
            [event["sequence_number"] for event in events], list(range(len(events)))
        )

        # A value outside the tool's enum comes back as written.
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                stream=True,
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200)
        events = response_events(payload)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(
            [item["arguments"] for item in events[-1]["response"]["output"]],
            ['{"city":"3"}'],
        )
        self.assertNotIn("<tool_call>", payload.decode())

    def test_responses_output_items_carry_distinct_ids(self):
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
            },
        }
        # Text, a tool call, then more text.
        harness = self.harness(FakeRuntime(Plan([[4, 5, 4]]), Plan([[1, 2, 3, 5]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                stream=True,
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200)
        events = response_events(payload)
        response = events[-1]["response"]
        public_id = response["id"].removeprefix("resp_")
        ids = [item["id"] for item in response["output"]]
        self.assertEqual(
            ids, [f"msg_{public_id}_0", f"fc_{public_id}_1", f"msg_{public_id}_2"]
        )
        ids_by_index = {}
        for event in events:
            if "output_index" in event:
                item_id = event["item"]["id"] if "item" in event else event["item_id"]
                ids_by_index.setdefault(event["output_index"], set()).add(item_id)
        self.assertEqual(
            ids_by_index, {index: {item_id} for index, item_id in enumerate(ids)}
        )
        status, _, stored = harness.request("GET", f"/v1/responses/{response['id']}")
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in json.loads(stored)["output"]], ids)

        status, _, payload = harness.request(
            "POST", "/v1/responses", responses_body(tools=[tool])
        )
        self.assertEqual(status, 200)
        response = json.loads(payload)
        public_id = response["id"].removeprefix("resp_")
        self.assertEqual(
            [item["id"] for item in response["output"]],
            [f"rs_{public_id}_0", f"msg_{public_id}_1", f"fc_{public_id}_2"],
        )

    def test_responses_structured_text_uses_decode_constraint(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        factory = FakeConstraintFactory()
        harness = self.harness(FakeRuntime(Plan([[10]])), constraint_factory=factory)
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                reasoning={"effort": "none"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "answer",
                        "schema": schema,
                        "strict": True,
                    }
                },
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(response["output"][0]["content"][0]["text"], '{"x":3}')
        self.assertEqual(response["text"]["format"]["schema"], schema)
        self.assertIn("%json", factory.grammars[0])

    def test_responses_tools_and_schema_work_in_stream_and_nonstream(self):
        harness = self.harness(FakeRuntime(Plan([[10]]), Plan([[10]]), Plan([[5]])))
        tool = rich_weather_tool()["function"]
        body = responses_body(
            tools=[{"type": "function", **tool}],
            reasoning={"effort": "none"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"const": 3}},
                        "required": ["x"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        status, _, payload = harness.request("POST", "/v1/responses", body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            json.loads(payload)["output"][-1]["content"][0]["text"], '{"x":3}'
        )
        status, _, payload = harness.request(
            "POST", "/v1/responses", {**body, "stream": True}
        )
        self.assertEqual(status, 200, payload)
        self.assertIn(b"response.completed", payload)
        self.assertNotIn(b"invalid_model_output", payload)
        status, _, payload = harness.request("POST", "/v1/responses", body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["output"][-1]["type"], "function_call")

    def test_responses_rejects_unsupported_features_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        cases = (
            ({"store": "yes"}, "store must be a boolean"),
            ({"previous_response_id": ""}, "previous_response_id"),
            ({"conversation": "conv_old"}, "conversation"),
            ({"background": True}, "background"),
            (
                {"text": {"format": {"type": "json_schema"}}},
                "response_format.json_schema.schema is required",
            ),
            ({"tools": [{"type": "custom", "name": "shell"}]}, "only function"),
            (
                {
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": "x"}],
                        }
                    ]
                },
                "only data: image URLs",
            ),
        )
        for extra, message in cases:
            with self.subTest(extra=extra):
                status, _, payload = harness.request(
                    "POST", "/v1/responses", responses_body(**extra)
                )
                self.assertEqual(status, 400)
                self.assertIn(message, json.loads(payload)["error"]["message"])
        self.assertEqual(runtime.requests, [])

    def test_responses_system_items_keep_their_input_position(self):
        messages = normalize_responses_input(
            "Be concise.",
            [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": "Answer in French.",
                },
                {"type": "message", "role": "user", "content": "again"},
            ],
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "system", "content": "Answer in French."},
                {"role": "user", "content": "again"},
            ],
        )

    def test_responses_store_retrieve_delete_and_store_false(self):
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(previous_response_id="resp_missing"),
        )
        missing_parent = {
            "message": "previous response not found",
            "type": "invalid_request_error",
            "code": "previous_response_not_found",
        }
        self.assertEqual((status, json.loads(payload)["error"]), (404, missing_parent))
        self.assertEqual(runtime.requests, [])

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(reasoning={"effort": "none"}),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertIs(response["store"], True)
        response_id = response["id"]

        status, _, payload = harness.request(
            "GET", f"/v1/responses/{response_id}?include=reasoning.encrypted_content"
        )
        self.assertEqual((status, json.loads(payload)), (200, response))
        status, _, payload = harness.request("GET", "/status")
        store_status = json.loads(payload)["response_store"]
        self.assertEqual(store_status["entries"], 1)
        self.assertLessEqual(store_status["bytes"], store_status["budget_bytes"])

        status, _, payload = harness.request("DELETE", f"/v1/responses/{response_id}")
        self.assertEqual(
            (status, json.loads(payload)),
            (200, {"id": response_id, "object": "response", "deleted": True}),
        )
        status, _, payload = harness.request("GET", f"/v1/responses/{response_id}")
        self.assertEqual(
            (status, json.loads(payload)["error"]["code"]), (404, "not_found_error")
        )
        status, _, payload = harness.request("DELETE", f"/v1/responses/{response_id}")
        self.assertEqual(
            (status, json.loads(payload)["error"]["code"]), (404, "not_found_error")
        )
        submitted = len(runtime.requests)
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(previous_response_id=response_id),
        )
        self.assertEqual((status, json.loads(payload)["error"]), (404, missing_parent))
        self.assertEqual(len(runtime.requests), submitted)

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(store=False, reasoning={"effort": "none"}),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertIs(response["store"], False)
        status, _, _ = harness.request("GET", f"/v1/responses/{response['id']}")
        self.assertEqual(status, 404)

    def test_previous_response_replays_history_but_not_old_instructions(self):
        tokenizer = FakeTokenizer()
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=tokenizer)
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                instructions="old instruction",
                reasoning={"effort": "none"},
            ),
        )
        first = json.loads(payload)
        self.assertEqual(status, 200)

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                input="follow up",
                instructions="new instruction",
                previous_response_id=first["id"],
                store=False,
                reasoning={"effort": "none"},
            ),
        )
        second = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(second["previous_response_id"], first["id"])
        rendered, _ = tokenizer.templates[-1]
        self.assertEqual(
            rendered,
            [
                {"role": "system", "content": "new instruction"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "plain answer\n"},
                {"role": "user", "content": "follow up"},
            ],
        )

    def test_streaming_response_is_stored_only_at_terminal_event(self):
        harness = self.harness(FakeRuntime(Plan([[4]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(stream=True, reasoning={"effort": "none"}),
        )
        self.assertEqual(status, 200)
        events = response_events(payload)
        terminal = events[-1]["response"]
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertIs(terminal["store"], True)
        status, _, payload = harness.request("GET", f"/v1/responses/{terminal['id']}")
        self.assertEqual((status, json.loads(payload)), (200, terminal))

    def test_previous_response_preserves_function_call_and_output(self):
        tokenizer = FakeTokenizer()
        runtime = FakeRuntime(Plan([[5]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=tokenizer)
        tools = [
            {
                "type": "function",
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                tools=tools,
                reasoning={"effort": "none"},
            ),
        )
        first = json.loads(payload)
        self.assertEqual(status, 200)
        call = next(item for item in first["output"] if item["type"] == "function_call")
        status, _, _ = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                previous_response_id=first["id"],
                tools=tools,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "sunny",
                    }
                ],
                store=False,
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200)
        rendered, _ = tokenizer.templates[-1]
        self.assertEqual(rendered[-2]["role"], "assistant")
        self.assertEqual(
            rendered[-2]["tool_calls"][0]["function"],
            {"name": "weather", "arguments": {"city": "Paris"}},
        )
        self.assertEqual(
            rendered[-1],
            {"role": "tool", "tool_call_id": call["call_id"], "content": "sunny"},
        )


if __name__ == "__main__":
    unittest.main()
