"""Custom (freeform) Responses tools, such as Codex's apply_patch: the model
calls a function of one string, and the API sees custom_tool_call items."""

import json

from dev.tests.server_fixtures import (
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    Plan,
    response_events,
    responses_body,
)
from server import api_shapes

PATCH = "*** Begin Patch\n*** Update File: notes.txt\n-a\n+b\n*** End Patch"
GRAMMAR = 'start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch" LF\n'
APPLY_PATCH = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Edit files. This is a FREEFORM tool.",
    "format": {"type": "grammar", "syntax": "lark", "definition": GRAMMAR},
}
EXEC = {
    "type": "function",
    "name": "exec_command",
    "parameters": {
        "type": "object",
        "properties": {"cmd": {"type": "string"}},
        "required": ["cmd"],
    },
}
CALL_TOKEN = 40


class PatchTokenizer(FakeTokenizer):
    """Its token CALL_TOKEN calls apply_patch with PATCH."""

    def __init__(self):
        super().__init__()
        self.fragments[CALL_TOKEN] = (
            "<tool_call>\n<function=apply_patch>\n<parameter=input>\n"
            f"{PATCH}\n</parameter>\n</function>\n</tool_call>\n"
        )


class CustomToolTests(HarnessTestCase):
    def patching(self, *plans):
        return self.harness(
            FakeRuntime(*plans) if plans else FakeRuntime(Plan([[CALL_TOKEN]])),
            tokenizer=PatchTokenizer(),
        )

    def test_the_model_sees_a_function_of_one_string_described_by_its_grammar(self):
        body = responses_body(tools=[EXEC, APPLY_PATCH])
        chat, names = api_shapes.responses_to_chat_body(body, body["input"])
        function = chat["tools"][1]["function"]
        self.assertEqual(function["name"], "apply_patch")
        self.assertEqual(function["description"], APPLY_PATCH["description"])
        self.assertEqual(function["parameters"]["required"], ["input"])
        self.assertFalse(function["parameters"]["additionalProperties"])
        described = function["parameters"]["properties"]["input"]
        self.assertEqual(described["type"], "string")
        self.assertIn("lark grammar", described["description"])
        self.assertIn(GRAMMAR, described["description"])
        self.assertEqual(
            names, {"apply_patch": (api_shapes.CUSTOM_TOOL, "apply_patch")}
        )
        # A text format, or none, is described as raw text.
        body = responses_body(tools=[{"type": "custom", "name": "note"}])
        chat, _ = api_shapes.responses_to_chat_body(body, body["input"])
        self.assertEqual(
            chat["tools"][0]["function"]["parameters"]["properties"]["input"][
                "description"
            ],
            "The tool's input, as raw text.",
        )

    def test_a_streamed_call_is_a_custom_tool_call_item(self):
        harness = self.patching()
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                stream=True, tools=[EXEC, APPLY_PATCH], reasoning={"effort": "none"}
            ),
        )
        self.assertEqual(status, 200, payload)
        events = response_events(payload)
        kinds = [event["type"] for event in events]
        self.assertNotIn("response.function_call_arguments.delta", kinds)
        self.assertNotIn("response.function_call_arguments.done", kinds)
        added = next(
            event
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "custom_tool_call"
        )
        self.assertEqual(added["item"]["input"], "")
        self.assertEqual(added["item"]["status"], "in_progress")
        delta = kinds.index("response.custom_tool_call_input.delta")
        done = kinds.index("response.custom_tool_call_input.done")
        item_done = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "custom_tool_call"
        )
        self.assertLess(events.index(added), delta)
        self.assertLess(delta, done)
        self.assertLess(done, item_done)
        item = events[item_done]["item"]
        self.assertEqual(
            {key: item[key] for key in ("type", "name", "input", "status")},
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "input": PATCH,
                "status": "completed",
            },
        )
        self.assertTrue(item["id"].startswith("ctc_"))
        self.assertTrue(item["call_id"])
        for index in (delta, done):
            self.assertEqual(events[index]["item_id"], item["id"])
            self.assertEqual(events[index]["output_index"], added["output_index"])
        self.assertEqual(events[delta]["delta"], PATCH)
        self.assertEqual(events[done]["input"], PATCH)
        completed = events[-1]["response"]
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(completed["output"], [item])
        self.assertFalse(completed["end_turn"])
        self.assertEqual(
            [event["sequence_number"] for event in events], list(range(len(events)))
        )

    def test_a_complete_response_holds_the_custom_tool_call(self):
        harness = self.patching()
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(tools=[APPLY_PATCH], reasoning={"effort": "none"}),
        )
        self.assertEqual(status, 200, payload)
        [item] = json.loads(payload)["output"]
        self.assertEqual(
            (item["type"], item["name"], item["input"]),
            ("custom_tool_call", "apply_patch", PATCH),
        )

    def test_custom_calls_and_their_outputs_return_as_history(self):
        harness = self.patching(Plan([[4]]))
        body = responses_body(tools=[APPLY_PATCH])
        body["input"] += [
            {
                "type": "custom_tool_call",
                "id": "ctc_1",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": PATCH,
                "status": "completed",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_patch",
                "output": "Success. Updated the following files:\nM notes.txt",
            },
        ]
        status, _, payload = harness.request("POST", "/v1/responses", body)
        self.assertEqual(status, 200, payload)
        messages = harness.tokenizer.templates[-1][0]
        call = messages[-2]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "apply_patch")
        arguments = call["function"]["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        self.assertEqual(arguments, {"input": PATCH})
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertIn("M notes.txt", json.dumps(messages[-1]["content"]))

    def test_a_stored_custom_call_continues_by_previous_response_id(self):
        harness = self.patching(Plan([[CALL_TOKEN]]), Plan([[4]]))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(tools=[APPLY_PATCH], reasoning={"effort": "none"}),
        )
        first = json.loads(payload)
        call = first["output"][-1]
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            {
                "model": "test-model",
                "previous_response_id": first["id"],
                "tools": [APPLY_PATCH],
                "input": [
                    {
                        "type": "custom_tool_call_output",
                        "call_id": call["call_id"],
                        "output": "Done",
                    }
                ],
            },
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["status"], "completed")

    def test_a_named_custom_tool_choice_names_its_function(self):
        body = responses_body(
            tools=[EXEC, APPLY_PATCH],
            tool_choice={"type": "custom", "name": "apply_patch"},
        )
        chat, _ = api_shapes.responses_to_chat_body(body, body["input"])
        self.assertEqual(
            chat["tool_choice"],
            {"type": "function", "function": {"name": "apply_patch"}},
        )
        harness = self.patching()
        for choice in (
            {"type": "custom", "name": "exec_command"},
            {"type": "custom", "name": "missing"},
            {"type": "function", "name": "apply_patch", "namespace": "x"},
        ):
            with self.subTest(choice=choice):
                status, _, _ = harness.request(
                    "POST",
                    "/v1/responses",
                    responses_body(tools=[EXEC, APPLY_PATCH], tool_choice=choice),
                )
                self.assertEqual(status, 400)

    def test_invalid_custom_tools_are_refused(self):
        harness = self.patching()
        for tool in (
            {"type": "custom"},
            {"type": "custom", "name": "a<b"},
            {"type": "custom", "name": " padded"},
            {"type": "custom", "name": "x", "description": 3},
            {"type": "custom", "name": "x", "format": {"type": "regex"}},
            {"type": "custom", "name": "x", "format": {"type": "grammar"}},
            {"type": "custom", "name": "exec_command"},
        ):
            with self.subTest(tool=tool):
                status, _, payload = harness.request(
                    "POST", "/v1/responses", responses_body(tools=[EXEC, tool])
                )
                self.assertEqual(status, 400, payload)
        for item in (
            {"type": "custom_tool_call", "call_id": "c", "name": "x"},
            {"type": "custom_tool_call", "call_id": "c", "input": "x"},
            {"type": "custom_tool_call_output", "output": "x"},
        ):
            with self.subTest(item=item):
                body = responses_body()
                body["input"].append(item)
                status, _, _ = harness.request("POST", "/v1/responses", body)
                self.assertEqual(status, 400)

    def test_custom_input_reads_the_one_string_of_the_arguments(self):
        for arguments, expected in (
            ('{"input": "patch"}', "patch"),
            ('{"patch": "text"}', "text"),
            ('{"input": "a", "other": 1}', "a"),
            ('{"a": "x", "b": "y"}', '{"a": "x", "b": "y"}'),
            ('{"input": "cut', '{"input": "cut'),
            ("[1]", "[1]"),
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(api_shapes.custom_input(arguments), expected)
