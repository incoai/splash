import json
import re
import tracemalloc
import unittest

from jsonschema import Draft202012Validator

from dev.tests.test_server import (
    FOREVER,
    FakeRuntime,
    FakeTokenizer,
    Harness,
    Plan,
    _byte_backend,
)
from dev.tests.tool_output import project, streamed_arguments, streamed_text
from server import api_shapes
from server import output as model_output
from server import server as api
from server.tool_schema import (
    PARAMETER_CLOSE,
    TOOL_CALL_OPEN,
    ToolPolicy,
    normalize_tools,
)

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
TOOL = {"type": "function", "function": {"name": "weather", "parameters": SCHEMA}}


def events(payload):
    return [
        json.loads(line[6:])
        for line in payload.decode().splitlines()
        if line.startswith("data: {")
    ]


def weather_call(city):
    return (
        "<tool_call>\n<function=weather>\n<parameter=city>\n"
        f"{city}\n</parameter>\n</function>\n</tool_call>"
    )


PARIS, ROME = weather_call("Paris"), weather_call("Rome")


def weather_policy():
    return ToolPolicy(
        {"weather": Draft202012Validator(SCHEMA)}, {"weather": SCHEMA}, False, True
    )


def character_tokenizer(text):
    """A tokenizer with one token per character of `text`, and the token of
    each character."""
    tokenizer = FakeTokenizer()
    tokenizer.fragments = dict(enumerate(dict.fromkeys(text), 1))
    tokenizer.backend_tokenizer = _byte_backend(tokenizer.fragments)
    return tokenizer, {value: key for key, value in tokenizer.fragments.items()}


def character_harness(text, reason="stop"):
    """A server whose model writes `text`, one character per token."""
    tokenizer, token_ids = character_tokenizer(text)
    return Harness(
        FakeRuntime(Plan([[token_ids[char]] for char in text], reason=reason)),
        tokenizer=tokenizer,
        max_context=8192,
    )


def weather_request(path, stream, thinking):
    """A request to `path` that offers the weather tool."""
    effort = "high" if thinking else "none"
    body = {"model": "test-model", "stream": stream}
    if path == "/v1/responses":
        return body | {
            "input": "Paris",
            "reasoning": {"effort": effort},
            "tools": [{"type": "function", **TOOL["function"]}],
        }
    body |= {"messages": [{"role": "user", "content": "Paris"}], "max_tokens": 4096}
    if path == "/v1/messages":
        return body | {
            "thinking": {"type": "enabled", "budget_tokens": 2048}
            if thinking
            else {"type": "disabled"},
            "tools": [{"name": "weather", "input_schema": SCHEMA}],
        }
    return body | {"reasoning_effort": effort, "tools": [TOOL]}


def respond(path, text, stream, thinking, reason="stop"):
    """The status and body of the response to a weather request whose model
    writes `text`, one character per token."""
    harness = character_harness(text, reason)
    try:
        status, _, payload = harness.request(
            "POST", path, weather_request(path, stream, thinking)
        )
    finally:
        harness.close()
    return status, payload


def append_run(items, kind, value):
    if items and items[-1][0] == kind:
        items[-1] = (kind, items[-1][1] + value)
    else:
        items.append((kind, value))


def chat_output(payload, stream):
    """The reasoning, text and call arguments of a chat completion, in
    stream order when it streams, and its finish reason."""
    if not stream:
        choice = json.loads(payload)["choices"][0]
        message = choice["message"]
        items = [
            (kind, message[field])
            for kind, field in (("reasoning", "reasoning_content"), ("text", "content"))
            if message.get(field)
        ]
        for call in message.get("tool_calls", []):
            items.append(("call", call["function"]["arguments"]))
        return items, choice["finish_reason"]
    items, finish = [], None
    for chunk in events(payload):
        for choice in chunk["choices"]:
            delta = choice["delta"]
            for kind, field in (
                ("reasoning", "reasoning_content"),
                ("text", "content"),
            ):
                if delta.get(field):
                    append_run(items, kind, delta[field])
            for call in delta.get("tool_calls", []):
                if "name" in call["function"]:
                    items.append(("call", ""))
                append_run(items, "call", call["function"].get("arguments", ""))
            finish = choice.get("finish_reason") or finish
    return items, finish


def responses_output(payload, stream):
    """The reasoning, text and call arguments of a response in output
    order, and its status; a stream's are those its last event reports."""
    response = events(payload)[-1]["response"] if stream else json.loads(payload)
    items = []
    for item in response["output"]:
        if item["type"] == "function_call":
            items.append(("call", item["arguments"]))
        else:
            kind = "reasoning" if item["type"] == "reasoning" else "text"
            items.append((kind, item["content"][0]["text"]))
    return items, response["status"]


def messages_output(payload, stream):
    """The thinking, text and tool input of a message in block order, and
    its stop reason."""
    kinds = {"thinking": "reasoning", "text": "text", "tool_use": "call"}
    if not stream:
        message = json.loads(payload)
        items = []
        for block in message["content"]:
            value = (
                json.dumps(block["input"], separators=(",", ":"))
                if block["type"] == "tool_use"
                else block[block["type"]]
            )
            items.append((kinds[block["type"]], value))
        return items, message["stop_reason"]
    fields = {
        "thinking_delta": "thinking",
        "text_delta": "text",
        "input_json_delta": "partial_json",
    }
    items, stop = [], None
    for event in events(payload):
        if event["type"] == "content_block_start":
            items.append((kinds[event["content_block"]["type"]], ""))
        elif event["type"] == "content_block_delta":
            field = fields.get(event["delta"]["type"])
            if field is not None:
                items[-1] = (items[-1][0], items[-1][1] + event["delta"][field])
        elif event["type"] == "message_delta":
            stop = event["delta"]["stop_reason"]
    return items, stop


def text_runs(projected):
    """The runs of text and the names of calls in stream order."""
    runs = []
    for kind, value in projected:
        if kind == "content":
            append_run(runs, "text", value)
        elif "name" in value["function"]:
            runs.append(("call", value["function"]["name"]))
    return runs


def visible_outside_calls(text):
    """The visible characters of `text` outside complete calls, without an
    unfinished call or call marker at its end."""
    parts = re.split(r"<tool_call>.*?</tool_call>", text, flags=re.S)
    unfinished = parts[-1].find(TOOL_CALL_OPEN)
    if unfinished >= 0:
        parts[-1] = parts[-1][:unfinished]
    else:
        parts[-1] = model_output.hold_partial(parts[-1], TOOL_CALL_OPEN)[0]
    return re.sub(r"\s", "", "".join(parts))


class PartialToolOutputTests(unittest.TestCase):
    def test_prefix_scanner_handles_nested_containers_and_scalar_boundaries(self):
        values = [
            [],
            {},
            [[], {}, [True, None]],
            {"a": [{"b": [1, 2]}]},
            -1.25e-19,
            'a\\"\n北京😀',
            True,
            None,
        ]
        for value in values:
            text = json.dumps(value)
            for end in range(len(text) + 1):
                prefix = text[:end]
                try:
                    json.loads(prefix)
                    incomplete = False
                except ValueError:
                    incomplete = True
                with self.subTest(prefix=prefix):
                    self.assertEqual(api_shapes._unfinished_json(prefix), incomplete)
        for invalid in (
            "[1,]",
            '{"a":1,}',
            "{1:",
            "[1 2",
            "1.2.",
            "1e2e",
            "1e2.",
            "01",
            "true false",
            '"\\u0z',
            '"\x01',
            '{"a" 1',
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(api_shapes._unfinished_json(invalid))

    def test_flat_partial_arguments_do_not_allocate_per_element(self):
        # Input construction is outside the measurement. The prefix scanner
        # should retain nesting state, not an array of parsed values/matches.
        arguments = '{"values":[' + ",".join("1" for _ in range(20000)) + ","
        tracemalloc.start()
        try:
            self.assertTrue(api_shapes._unfinished_json(arguments))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 64 * 1024)

    def harness(self, *plans):
        value = Harness(FakeRuntime(*plans))
        self.addCleanup(value.close)
        return value

    def test_tool_only_whitespace_is_absent_in_every_protocol(self):
        call = weather_call("Paris")
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            for stream in (False, True):
                for thinking in (False, True):
                    with self.subTest(path=path, stream=stream, thinking=thinking):
                        text = (
                            ("reason</think>" if thinking else "")
                            + " \n"
                            + call
                            + "\n\t"
                            + call
                            + "\n"
                        )
                        harness = character_harness(text)
                        try:
                            status, _, payload = harness.request(
                                "POST", path, weather_request(path, stream, thinking)
                            )
                            self.assertEqual(status, 200, payload)
                            if stream:
                                rows = events(payload)
                                self.assertFalse(
                                    any("error" in row for row in rows), rows
                                )
                                if path == "/v1/chat/completions":
                                    deltas = [
                                        row["choices"][0]["delta"]
                                        for row in rows
                                        if row.get("choices")
                                    ]
                                    self.assertFalse(
                                        any(delta.get("content") for delta in deltas)
                                    )
                                    self.assertEqual(
                                        sum(
                                            "name" in call["function"]
                                            for delta in deltas
                                            for call in delta.get("tool_calls", [])
                                        ),
                                        2,
                                    )
                                elif path == "/v1/responses":
                                    self.assertFalse(
                                        any(
                                            row["type"] == "response.output_text.delta"
                                            for row in rows
                                        )
                                    )
                                    self.assertEqual(
                                        [
                                            item["type"]
                                            for item in rows[-1]["response"]["output"]
                                            if item["type"] != "reasoning"
                                        ],
                                        ["function_call", "function_call"],
                                    )
                                else:
                                    blocks = [
                                        row["content_block"]
                                        for row in rows
                                        if row["type"] == "content_block_start"
                                    ]
                                    self.assertEqual(
                                        [
                                            block["type"]
                                            for block in blocks
                                            if block["type"] != "thinking"
                                        ],
                                        ["tool_use", "tool_use"],
                                    )
                            else:
                                result = json.loads(payload)
                                if path == "/v1/chat/completions":
                                    message = result["choices"][0]["message"]
                                    self.assertIsNone(message["content"])
                                    self.assertEqual(len(message["tool_calls"]), 2)
                                elif path == "/v1/responses":
                                    self.assertEqual(
                                        [
                                            item["type"]
                                            for item in result["output"]
                                            if item["type"] != "reasoning"
                                        ],
                                        ["function_call", "function_call"],
                                    )
                                else:
                                    self.assertEqual(
                                        [
                                            item["type"]
                                            for item in result["content"]
                                            if item["type"] != "thinking"
                                        ],
                                        ["tool_use", "tool_use"],
                                    )
                        finally:
                            harness.close()

    def test_newlines_after_think_end_do_not_start_the_answer_in_any_protocol(self):
        # Models write "\n</think>\n\n" before the answer, as their chat
        # template lays out a turn. Whether each output thinks, and its text:
        outputs = (
            (True, "why\n</think>\n\nHi\n\nyou"),
            (True, "why\n</think>\n \n\tHi"),
            (True, "why\n</think>\n\n"),
            (True, "why\n</think>"),
            (False, "\n\nHi"),
        )
        protocols = {
            "/v1/chat/completions": chat_output,
            "/v1/responses": responses_output,
            "/v1/messages": messages_output,
        }
        tokenizer, token_ids = character_tokenizer("".join(t for _, t in outputs))
        for path, parse in protocols.items():
            for stream in (False, True):
                for tools in (False, True):
                    with self.subTest(path=path, stream=stream, tools=tools):
                        harness = Harness(
                            FakeRuntime(
                                *(
                                    Plan([[token_ids[char]] for char in text])
                                    for _, text in outputs
                                )
                            ),
                            tokenizer=tokenizer,
                            max_context=8192,
                        )
                        answers = []
                        try:
                            for thinking, _ in outputs:
                                request = weather_request(path, stream, thinking)
                                if not tools:
                                    del request["tools"]
                                status, _, payload = harness.request(
                                    "POST", path, request
                                )
                                self.assertEqual(status, 200, payload)
                                answers.append(parse(payload, stream)[0])
                        finally:
                            harness.close()
                        self.assertEqual(
                            answers[0], [("reasoning", "why\n"), ("text", "Hi\n\nyou")]
                        )
                        # Only newlines are dropped.
                        self.assertEqual(
                            answers[1], [("reasoning", "why\n"), ("text", " \n\tHi")]
                        )
                        # Newlines alone after </think> are no answer.
                        self.assertEqual(answers[2], answers[3])
                        # Without thinking, leading newlines are the model's own.
                        self.assertEqual(answers[4], [("text", "\n\nHi")])

    def test_reasoning_splitter_starts_the_answer_however_the_text_is_cut(self):
        text = "why\n</think>\n\nHi\n\nyou"
        for size in range(1, len(text) + 1):
            with self.subTest(size=size):
                splitter = model_output.ReasoningSplitter(True)
                parts = [
                    part
                    for offset in range(0, len(text), size)
                    for part in splitter.put(text[offset : offset + size])
                ]
                parts += splitter.finish()
                self.assertTrue(all(value for _, value in parts), parts)
                self.assertEqual(
                    [
                        "".join(value for kind, value in parts if kind == field)
                        for field in ("reasoning_content", "content")
                    ],
                    ["why\n", "Hi\n\nyou"],
                )

    def test_projector_preserves_whitespace_without_a_tool(self):
        for text in (" \n\t", " \nhello \t\n"):
            for incomplete in (False, True):
                content, calls, projected = project(
                    text, None, incomplete=incomplete, size=1
                )
                self.assertEqual((content, calls), (text, []))
                self.assertEqual(streamed_text(projected), text)

    def test_every_json_prefix_can_be_returned_in_tool_history(self):
        objects = [
            {"city": '北京😀\\"\n'},
            {"a": True, "b": False, "c": None, "d": [-1.25e-19, 2, {"e": 0}]},
        ]
        for value in objects:
            for ascii_only in (True, False):
                text = json.dumps(value, ensure_ascii=ascii_only)
                # An empty string is a complete call without arguments.
                for end in range(1, len(text)):
                    arguments = text[:end]
                    with self.subTest(arguments=arguments):
                        call = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        }
                        normalized = api_shapes.normalize_messages(
                            [
                                call,
                                {"role": "user", "content": "Continue"},
                            ],
                            vision=True,
                            deadline=FOREVER,
                        )
                        actual = normalized[0]["tool_calls"][0]["function"]
                        self.assertEqual(actual["arguments"], arguments)
        for invalid in (
            '{"a":01',
            '{"a":"\\q',
            '{"a":true garbage',
            '{"a":,',
            '{"a":1,}',
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(api_shapes._unfinished_json(invalid))

    def test_chat_preserves_same_partial_arguments_in_both_modes(self):
        harness = self.harness(
            Plan([[13]], reason="length"), Plan([[13]], reason="length")
        )
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [TOOL],
            "reasoning_effort": "none",
            "max_tokens": 16,
        }
        status, _, raw = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200, raw)
        choice = json.loads(raw)["choices"][0]
        self.assertEqual(choice["finish_reason"], "length")
        arguments = choice["message"]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments, '{"city":"Par')
        with self.assertRaises(ValueError):
            json.loads(arguments)
        status, _, raw = harness.request(
            "POST", "/v1/chat/completions", {**body, "stream": True}
        )
        self.assertEqual(status, 200, raw)
        chunks = events(raw)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        streamed = "".join(
            call.get("function", {}).get("arguments", "")
            for item in chunks
            for call in item["choices"][0]["delta"].get("tool_calls", [])
        )
        self.assertEqual(streamed, arguments)

    def test_responses_partial_history_is_retained_and_continues(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                harness = self.harness(Plan([[13]], reason="length"), Plan([[4]]))
                body = {
                    "model": "test-model",
                    "input": "weather",
                    "stream": stream,
                    "reasoning": {"effort": "none"},
                    "tools": [{"type": "function", **TOOL["function"]}],
                    "max_output_tokens": 16,
                }
                status, _, raw = harness.request("POST", "/v1/responses", body)
                self.assertEqual(status, 200, raw)
                response = events(raw)[-1]["response"] if stream else json.loads(raw)
                self.assertEqual(response["status"], "incomplete")
                call = response["output"][0]
                self.assertEqual(call["type"], "function_call")
                self.assertEqual(call["status"], "incomplete")
                self.assertEqual(call["arguments"], '{"city":"Par')
                status, _, raw = harness.request(
                    "POST",
                    "/v1/responses",
                    {
                        "model": "test-model",
                        "previous_response_id": response["id"],
                        "input": "Continue without that unfinished tool call.",
                        "reasoning": {"effort": "none"},
                        "max_output_tokens": 16,
                    },
                )
                self.assertEqual(status, 200, raw)
                self.assertEqual(json.loads(raw)["status"], "completed")
                status, _, stored = harness.request(
                    "GET", "/v1/responses/" + response["id"]
                )
                self.assertEqual(status, 200, stored)
                self.assertEqual(json.loads(stored)["output"], response["output"])
                history = harness.tokenizer.templates[-1][0]
                unfinished = [
                    json.loads(item["content"].split("\n", 1)[1])
                    for item in history
                    if str(item.get("content")).startswith("Incomplete tool call:\n")
                ]
                self.assertEqual(
                    unfinished, [{"name": "weather", "arguments": '{"city":"Par'}]
                )

    def test_anthropic_preserves_closed_calls_without_fabricating_partial_input(self):
        for tokens, expected in (([[13]], []), ([[5], [13]], [{"city": "Paris"}])):
            with self.subTest(tokens=tokens):
                harness = self.harness(Plan(tokens, reason="length"))
                status, _, raw = harness.request(
                    "POST",
                    "/v1/messages",
                    {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "weather"}],
                        "thinking": {"type": "disabled"},
                        "max_tokens": 16,
                        "tools": [{"name": "weather", "input_schema": SCHEMA}],
                    },
                )
                self.assertEqual(status, 200, raw)
                response = json.loads(raw)
                self.assertEqual(response["stop_reason"], "max_tokens")
                self.assertTrue(response["content"])
                self.assertEqual(
                    [
                        item["input"]
                        for item in response["content"]
                        if item["type"] == "tool_use"
                    ],
                    expected,
                )

    def test_complete_and_partial_calls_survive_arbitrary_chunk_boundaries(self):
        tokenizer = FakeTokenizer()
        text = tokenizer.fragments[5] + tokenizer.fragments[13]
        policy = ToolPolicy(
            {"weather": Draft202012Validator(SCHEMA)}, {"weather": SCHEMA}, False, True
        )
        expected = None
        for size in (1, 3, 17, len(text)):
            result = project(text, policy, "owned", incomplete=True, size=size)[:2]
            if expected is None:
                expected = result
            self.assertEqual(result, expected)
        self.assertEqual(
            [call["function"]["arguments"] for call in expected[1]],
            ['{"city":"Paris"}', '{"city":"Par'],
        )

    def test_typed_parameter_value_keeps_pending_bounded(self):
        # A JSON value held whole until its close would be rescanned and
        # copied by every put, quadratic in its length.
        schema = {"type": "object", "properties": {"items": {"type": "array"}}}
        policy = ToolPolicy(
            {"store": Draft202012Validator(schema)}, {"store": schema}, False, True
        )
        items = ["x" * 100] * 2048
        text = (
            "<tool_call>\n<function=store>\n<parameter=items>\n"
            + json.dumps(items)
            + "\n</parameter>\n</function>\n</tool_call>"
        )
        projector = model_output.StreamingToolCallProjector(policy, "typed")
        projected = []
        for offset in range(0, len(text), 4):
            projected += projector.put(text[offset : offset + 4])
            if projector.state == "parameter_value":
                self.assertLess(len(projector.pending), len(PARAMETER_CLOSE))
        self.assertEqual(
            streamed_arguments(projected),
            json.dumps({"items": items}, separators=(",", ":")),
        )

    def test_closed_calls_still_require_schema_validation_at_length(self):
        schema = {
            **SCHEMA,
            "properties": {"city": {"type": "string", "enum": ["Berlin"]}},
        }
        policy = ToolPolicy(
            {"weather": Draft202012Validator(schema)}, {"weather": schema}, False, True
        )
        projector = model_output.StreamingToolCallProjector(policy, "owned")
        with self.assertRaises(api.APIError):
            projector.put(FakeTokenizer().fragments[5])


class TextAfterToolCallTests(unittest.TestCase):
    # Text around calls: the output in #231, text between calls, a preface
    # to parallel calls, a tool-only turn, text with whitespace at its
    # edges, and markup-like text after a call.
    OUTPUTS = [
        "First message. "
        + weather_call("Paris")
        + "POST-CALL TEXT THAT SHOULD BE VISIBLE",
        "Text. "
        + weather_call("Paris")
        + "\nMore text.\n"
        + weather_call("Rome")
        + "\nEnd.",
        "I'll check both.\n\n" + weather_call("Paris") + "\n" + weather_call("Rome"),
        weather_call("Paris") + "\n" + weather_call("Rome") + "\n",
        " \nalpha \t" + weather_call("Paris") + "\n beta \t\n",
        "answer <" + weather_call("Paris") + "<b> & </tool_ok>",
    ]
    # Outputs with their content and their runs of text and calls. The
    # template's whitespace around calls, before the first text or after the
    # last, is neither streamed nor reported, at a normal finish or a cut.
    # Between two texts it is their separator and streams with the later
    # one, and text keeps its own whitespace.
    WHITESPACE = [
        (
            "Text. " + PARIS + "\nMore text.\n" + ROME + "\n",
            "Text. \nMore text.\n",
            [
                ("text", "Text. "),
                ("call", "weather"),
                ("text", "\nMore text.\n"),
                ("call", "weather"),
            ],
        ),
        (
            "I'll check both.\n\n" + PARIS + "\n" + ROME + "\n",
            "I'll check both.\n\n",
            [
                ("text", "I'll check both.\n\n"),
                ("call", "weather"),
                ("call", "weather"),
            ],
        ),
        (
            " \n" + PARIS + "\n" + ROME + "\nDone.",
            "\nDone.",
            [("call", "weather"), ("call", "weather"), ("text", "\nDone.")],
        ),
        (
            "Checking both." + PARIS + "\n" + ROME + "Done.",
            "Checking both.\nDone.",
            [
                ("text", "Checking both."),
                ("call", "weather"),
                ("call", "weather"),
                ("text", "\nDone."),
            ],
        ),
        (
            "Hi" + PARIS + "\n" + ROME + "\nBye",
            "Hi\n\nBye",
            [
                ("text", "Hi"),
                ("call", "weather"),
                ("call", "weather"),
                ("text", "\n\nBye"),
            ],
        ),
        (PARIS + "\n" + ROME + "\n", "", [("call", "weather")] * 2),
    ]

    def test_text_after_a_call_streams_and_survives_a_cut(self):
        # The reproduction in #231.
        policy = normalize_tools(
            [
                {
                    "type": "function",
                    "function": {"name": "read", "parameters": {"type": "object"}},
                }
            ],
            "auto",
            True,
        )[1]
        projector = model_output.StreamingToolCallProjector(policy, 1)
        call = "<tool_call>\n<function=read>\n</function>\n</tool_call>"
        first = projector.put("First message. " + call)
        second = projector.put("POST-CALL TEXT THAT SHOULD BE VISIBLE")
        self.assertEqual(
            [kind for kind, _ in first], ["content", "tool", "tool", "tool"]
        )
        self.assertEqual(second, [("content", "POST-CALL TEXT THAT SHOULD BE VISIBLE")])
        content, calls, unsent = projector.finish(True)
        self.assertEqual(
            content, "First message. POST-CALL TEXT THAT SHOULD BE VISIBLE"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(unsent, "")

    def test_a_cut_reports_the_text_it_streamed_without_markup(self):
        for text in self.OUTPUTS:
            for end in range(len(text) + 1):
                cut = text[:end]
                results = set()
                for size in (1, 3, 17, None):
                    content, calls, projected = project(
                        cut, weather_policy(), incomplete=True, size=size
                    )
                    with self.subTest(cut=cut, size=size):
                        self.assertEqual(streamed_text(projected), content)
                        self.assertNotIn("<tool_call", content)
                        self.assertNotIn("</function", content)
                        # No text the model wrote outside a call is lost.
                        self.assertEqual(
                            re.sub(r"\s", "", content), visible_outside_calls(cut)
                        )
                    results.add((content, json.dumps(calls)))
                with self.subTest(cut=cut):
                    self.assertEqual(len(results), 1, results)

    def test_a_completed_output_streams_its_content(self):
        for text, content in zip(
            self.OUTPUTS,
            [
                "First message. POST-CALL TEXT THAT SHOULD BE VISIBLE",
                "Text. \nMore text.\n\nEnd.",
                "I'll check both.\n\n",
                "",
                " \nalpha \t\n beta \t\n",
                "answer <<b> & </tool_ok>",
            ],
            strict=True,
        ):
            for size in (1, 3, 17, None):
                with self.subTest(text=text, size=size):
                    reported, _, projected = project(text, weather_policy(), size=size)
                    self.assertEqual(reported, content)
                    self.assertEqual(streamed_text(projected), content)

    def test_whitespace_alone_is_text_only_between_texts(self):
        for text, content, runs in self.WHITESPACE:
            for incomplete in (False, True):
                for size in (1, 3, None):
                    with self.subTest(text=text, incomplete=incomplete, size=size):
                        reported, _, projected = project(
                            text, weather_policy(), incomplete=incomplete, size=size
                        )
                        self.assertEqual(text_runs(projected), runs)
                        self.assertEqual(reported, content)

    def test_non_streaming_tool_output_uses_the_streaming_projector(self):
        def joined(items):
            text = "".join(value for kind, value in items if kind == "text")
            return text, [value for kind, value in items if kind == "call"]

        for text, _, _ in self.WHITESPACE:
            outputs = []
            for stream in (False, True):
                with self.subTest(text=text, stream=stream):
                    status, payload = respond(
                        "/v1/chat/completions", text, stream, False
                    )
                    self.assertEqual(status, 200, payload)
                    outputs.append(joined(chat_output(payload, stream)[0]))
            with self.subTest(text=text):
                self.assertEqual(outputs[0], outputs[1])
        # The model writes a call without its required argument, then text
        # after a pause. The projector checks the call as it is read, so the
        # request fails and cancels the model while it is still writing.
        parts = "<tool_call>\n<function=weather>\n</function>\n</tool_call>", "Done."
        tokenizer, token_ids = character_tokenizer("".join(parts))
        plan = Plan([[token_ids[char] for char in part] for part in parts], delay=1)
        harness = Harness(FakeRuntime(plan), tokenizer=tokenizer, max_context=8192)
        try:
            status, _, payload = harness.request(
                "POST",
                "/v1/chat/completions",
                weather_request("/v1/chat/completions", False, False),
            )
            self.assertTrue(plan.cancelled.is_set())
        finally:
            harness.close()
        self.assertEqual(status, 500, payload)
        self.assertEqual(json.loads(payload)["error"]["code"], "invalid_model_output")

    def test_whitespace_around_parallel_calls_in_every_protocol(self):
        # After a preface and parallel calls, the newlines between and after
        # the calls are not text. With text after the calls as well, the
        # newline between them separates the two texts. Only a complete chat
        # message joins its text ahead of its calls.
        paris, rome = weather_call("Paris"), weather_call("Rome")
        calls = [("call", '{"city":"Paris"}'), ("call", '{"city":"Rome"}')]
        for text, joined, ordered in (
            (
                "I'll check both.\n\n" + paris + "\n" + rome + "\n",
                [("text", "I'll check both.\n\n")] + calls,
                [("text", "I'll check both.\n\n")] + calls,
            ),
            (
                "Checking both." + paris + "\n" + rome + "Done.",
                [("text", "Checking both.\nDone.")] + calls,
                [("text", "Checking both.")] + calls + [("text", "\nDone.")],
            ),
        ):
            for path, output in (
                ("/v1/chat/completions", chat_output),
                ("/v1/responses", responses_output),
                ("/v1/messages", messages_output),
            ):
                for stream in (False, True):
                    with self.subTest(text=text, path=path, stream=stream):
                        status, payload = respond(path, text, stream, False)
                        self.assertEqual(status, 200, payload)
                        items, _ = output(payload, stream)
                        chat = path == "/v1/chat/completions" and not stream
                        self.assertEqual(items, joined if chat else ordered)

    def check_cut_after_text_following_a_call(self, path, output, complete, finish):
        # The model writes text, a call and more text, and the token limit
        # cuts a second call inside its argument.
        text = (
            "First message. "
            + weather_call("Paris")
            + "\nPost-call text.\n"
            + "<tool_call>\n<function=weather>\n<parameter=city>\nRo"
        )
        streamed = [
            ("text", "First message. "),
            ("call", '{"city":"Paris"}'),
            ("text", "\nPost-call text.\n"),
            ("call", '{"city":"Ro'),
        ]
        for stream in (False, True):
            for thinking in (False, True):
                with self.subTest(stream=stream, thinking=thinking):
                    reasoning = "Reasoning.</think>" if thinking else ""
                    status, payload = respond(
                        path, reasoning + text, stream, thinking, "length"
                    )
                    self.assertEqual(status, 200, payload)
                    self.assertNotIn(b"<tool_call", payload)
                    items = [("reasoning", "Reasoning.")] if thinking else []
                    items += streamed if stream else complete
                    self.assertEqual(output(payload, stream), (items, finish))

    def test_chat_keeps_text_after_a_call_at_length(self):
        self.check_cut_after_text_following_a_call(
            "/v1/chat/completions",
            chat_output,
            [
                ("text", "First message. \nPost-call text.\n"),
                ("call", '{"city":"Paris"}'),
                ("call", '{"city":"Ro'),
            ],
            "length",
        )

    def test_responses_keep_text_after_a_call_at_length(self):
        self.check_cut_after_text_following_a_call(
            "/v1/responses",
            responses_output,
            [
                ("text", "First message. "),
                ("call", '{"city":"Paris"}'),
                ("text", "\nPost-call text.\n"),
                ("call", '{"city":"Ro'),
            ],
            "incomplete",
        )

    def test_messages_keep_text_after_a_call_at_max_tokens(self):
        # A complete message leaves out the unfinished call's partial input.
        self.check_cut_after_text_following_a_call(
            "/v1/messages",
            messages_output,
            [
                ("text", "First message. "),
                ("call", '{"city":"Paris"}'),
                ("text", "\nPost-call text.\n"),
            ],
            "max_tokens",
        )

    def test_responses_stored_object_is_the_same_streamed_or_not(self):
        # Items closed before the cut are complete in both modes; only the
        # call the token limit cut is incomplete.
        text = (
            "First message. "
            + PARIS
            + "\nPost-call text.\n"
            + "<tool_call>\n<function=weather>\n<parameter=city>\nRo"
        )
        stored = []
        for stream in (False, True):
            harness = character_harness(text, "length")
            try:
                status, _, payload = harness.request(
                    "POST",
                    "/v1/responses",
                    weather_request("/v1/responses", stream, False),
                )
                self.assertEqual(status, 200, payload)
                response = (
                    events(payload)[-1]["response"] if stream else json.loads(payload)
                )
                status, _, payload = harness.request(
                    "GET", "/v1/responses/" + response["id"]
                )
                self.assertEqual(status, 200, payload)
            finally:
                harness.close()
            public_id = response["id"].removeprefix("resp_")
            stored.append(
                json.loads(payload.decode().replace(public_id, "id"))["output"]
            )
        self.assertEqual(stored[0], stored[1])
        self.assertEqual(
            [(item["type"], item["status"]) for item in stored[0]],
            [
                ("message", "completed"),
                ("function_call", "completed"),
                ("message", "completed"),
                ("function_call", "incomplete"),
            ],
        )

    def test_thinking_only_output_ends_with_an_empty_text_block_in_both_modes(self):
        for path, output in (
            ("/v1/responses", responses_output),
            ("/v1/messages", messages_output),
        ):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    status, payload = respond(path, "Reasoning.</think>", stream, True)
                    self.assertEqual(status, 200, payload)
                    self.assertEqual(
                        output(payload, stream)[0],
                        [("reasoning", "Reasoning."), ("text", "")],
                    )
