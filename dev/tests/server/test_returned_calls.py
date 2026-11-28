"""Which calls the model writes are returned as calls, in every dialect,
streamed or not: a call is returned under the declared tool its name names,
whatever its case; one whose name matches no declared tool, or several, is
the text the model wrote, as is a held call whose arguments are not a JSON
object, while one that streamed ends as it was sent. A call the output cuts,
by the token limit or a stop sequence, is dropped where it was held and ends
with the arguments it sent where it streamed, the text before it kept and
the finish reason the true one; the server counts each call it does not
return and says so in one line."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from dev.tests import server_fixtures as fixtures
from dev.tests.server import test_partial_tool_output as partial
from dev.tests.tool_output import project, put, streamed_text, tool_policy
from server import output as output_module
from server import server as api
from server.chat_format import INKLING2, QWEN_XML

PATHS = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
INKLING2_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "fixtures/chat_templates/inkling2.jinja"
).read_text()
MODEL, THINKING, TEXT, END, JSON_CALL = (
    "<|message_model|>",
    "<|content_thinking|>",
    "<|content_text|>",
    "<|end_message|>",
    "<|content_invoke_tool_json|>",
)
# The output of Inkling 2 asked to print its own call syntax, as the probe
# saw it (serving-probe-api.md, K2): the model's message named "then", whose
# body is another call's header and object, not a JSON object.
THEN_CALL = (
    MODEL
    + " then "
    + MODEL
    + "weather"
    + JSON_CALL
    + '{"args":{"city":"Paris"},"name":"weather"}'
    + END
)


def output(path, payload, stream):
    """The output items of a response, in order, and how it ended."""
    read = {
        "/v1/chat/completions": partial.chat_output,
        "/v1/messages": partial.messages_output,
        "/v1/responses": partial.responses_output,
    }[path]
    return read(payload, stream)


def text_of(items):
    return "".join(value for kind, value in items if kind == "text")


def called_names(path, payload, stream):
    """The names of the calls a response returns, in order."""
    if path == "/v1/chat/completions":
        if not stream:
            message = json.loads(payload)["choices"][0]["message"]
            return [call["function"]["name"] for call in message.get("tool_calls", [])]
        return [
            call["function"]["name"]
            for chunk in partial.events(payload)
            for choice in chunk["choices"]
            for call in choice["delta"].get("tool_calls", [])
            if "name" in call["function"]
        ]
    if path == "/v1/messages":
        blocks = (
            [
                event["content_block"]
                for event in partial.events(payload)
                if event["type"] == "content_block_start"
            ]
            if stream
            else json.loads(payload)["content"]
        )
        return [block["name"] for block in blocks if block["type"] == "tool_use"]
    response = (
        partial.events(payload)[-1]["response"] if stream else json.loads(payload)
    )
    return [
        item["name"] for item in response["output"] if item["type"] == "function_call"
    ]


def calls_of(items):
    return [value for kind, value in items if kind == "call"]


class InklingHarness:
    """A server whose chat format is Inkling 2's, with the weather tool, its
    tags each a token of the fake vocabulary and the model's output the
    text of a plan's tokens."""

    TAGS = (MODEL, THINKING, TEXT, END, JSON_CALL)

    def __init__(self, test, *texts, reason="stop"):
        fragments, plans = {}, []
        for text in texts:
            tokens = []
            for piece in _pieces(text, self.TAGS):
                token = next(
                    (key for key, value in fragments.items() if value == piece), None
                )
                if token is None:
                    token = 60 + len(fragments)
                    fragments[token] = piece
                tokens.append([token])
            plans.append(fixtures.Plan(tokens, reason=reason))
        for tag in self.TAGS:
            if tag not in fragments.values():
                fragments[60 + len(fragments)] = tag
        ids = {value: key for key, value in fragments.items()}
        tokenizer = fixtures.TemplateTokenizer(INKLING2_TEMPLATE)
        tokenizer.fragments = fragments
        tokenizer.backend_tokenizer = fixtures.byte_backend(fragments)
        chat_format = replace(
            INKLING2,
            think_end_id=ids[END],
            think_start_ids=(ids[MODEL], ids[THINKING]),
            call_open_id=ids[MODEL],
        )
        self.harness = fixtures.Harness(
            fixtures.FakeRuntime(*plans),
            tokenizer=tokenizer,
            chat_format=chat_format,
            max_context=8192,
        )
        test.addCleanup(self.harness.close)

    def request(self, path, stream, **extra):
        return self.harness.request(
            "POST", path, partial.weather_request(path, stream, False) | extra
        )


def _pieces(text, tags):
    """`text` as tokens: each tag one, and each character between."""
    pieces = []
    while text:
        tag = next((tag for tag in tags if text.startswith(tag)), None)
        piece = tag or text[0]
        pieces.append(piece)
        text = text[len(piece) :]
    return pieces


class RejectedCallTests(unittest.TestCase):
    def test_a_call_of_an_undeclared_tool_is_its_text_in_every_dialect(self):
        # Qwen's tags: the request declares only weather.
        call = (
            "<tool_call>\n<function=search>\n<parameter=query>\nllamas\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    status, payload = partial.respond(
                        path, "Look: " + call, stream, False
                    )
                    self.assertEqual(status, 200, payload)
                    items, end = output(path, payload, stream)
                    self.assertEqual(calls_of(items), [])
                    self.assertEqual(text_of(items), "Look: " + call)
                    self.assertIn(end, ("stop", "end_turn", "completed"))

    def test_an_undeclared_call_beside_a_declared_one_leaves_it_whole(self):
        undeclared = "<tool_call>\n<function=search>\n</function>\n</tool_call>"
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    status, payload = partial.respond(
                        path, undeclared + partial.PARIS, stream, False
                    )
                    self.assertEqual(status, 200, payload)
                    items, end = output(path, payload, stream)
                    self.assertEqual(calls_of(items), ['{"city":"Paris"}'])
                    self.assertEqual(text_of(items), undeclared)
                    self.assertIn(end, ("tool_calls", "tool_use", "completed"))

    def test_inkling_s_call_of_an_undeclared_tool_is_text_not_a_500(self):
        # The probe's K2: a call to "then" whose arguments are not JSON came
        # back as a call, and as a 500 from a complete Messages response.
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    harness = InklingHarness(self, MODEL + TEXT + "Sure: " + THEN_CALL)
                    status, _, payload = harness.request(path, stream)
                    self.assertEqual(status, 200, payload)
                    items, end = output(path, payload, stream)
                    self.assertEqual(calls_of(items), [])
                    self.assertEqual(text_of(items), "Sure: " + THEN_CALL)
                    self.assertIn(end, ("stop", "end_turn", "completed"))

    def test_inkling_s_call_whose_arguments_are_not_an_object_is_text(self):
        # Arguments that do not begin as an object are text, streamed or not.
        for arguments in ("[1,2]", '"Paris"'):
            call = MODEL + "weather" + JSON_CALL
            call += '{"name":"weather","args":' + arguments + "}" + END
            for path in PATHS:
                for stream in (False, True):
                    with self.subTest(arguments=arguments, path=path, stream=stream):
                        harness = InklingHarness(self, call)
                        status, _, payload = harness.request(path, stream)
                        self.assertEqual(status, 200, payload)
                        items, end = output(path, payload, stream)
                        self.assertEqual(calls_of(items), [])
                        self.assertEqual(text_of(items), call)
                        self.assertIn(end, ("stop", "end_turn", "completed"))

    def test_inkling_s_arguments_that_end_unreadable_end_as_each_mode_ends(self):
        # Arguments that begin as an object stream; not JSON at the close,
        # the call ends as sent, counted. Not streaming, the call is text.
        call = MODEL + "weather" + JSON_CALL
        call += '{"name":"weather","args":{"city":"Pa}' + END
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    harness = InklingHarness(self, call)
                    with mock.patch.object(api, "print_status") as printed:
                        status, _, payload = harness.request(path, stream)
                    self.assertEqual(status, 200, payload)
                    items, end = output(path, payload, stream)
                    if stream:
                        self.assertEqual(calls_of(items), ['{"city":"Pa'])
                        self.assertEqual(text_of(items), "")
                    else:
                        self.assertEqual(calls_of(items), [])
                        self.assertEqual(text_of(items), call)
                    self.assertIn(
                        "Tool calls · request 1 · 1 arguments not a JSON object",
                        [args[0] for args, _ in printed.call_args_list],
                    )
                    self.assertEqual(
                        harness.harness.app.status()["tool_calls"]["rejected"][
                            "invalid_arguments"
                        ],
                        1,
                    )

    def test_inkling_s_declared_call_is_still_returned(self):
        # Under the declared name, whatever the case the model wrote it in.
        for name in ("weather", "Weather"):
            call = MODEL + name + JSON_CALL
            call += '{"name":"' + name + '","args":{"city":"Paris"}}' + END
            for path in PATHS:
                for stream in (False, True):
                    with self.subTest(name=name, path=path, stream=stream):
                        harness = InklingHarness(self, call)
                        status, _, payload = harness.request(path, stream)
                        self.assertEqual(status, 200, payload)
                        items, end = output(path, payload, stream)
                        self.assertEqual(calls_of(items), ['{"city":"Paris"}'])
                        self.assertEqual(
                            called_names(path, payload, stream), ["weather"]
                        )
                        self.assertIn(end, ("tool_calls", "tool_use", "completed"))

    def test_a_call_naming_a_declared_tool_in_another_case_is_that_tool_s(self):
        # The request declares weather; the model writes Weather, as Flint
        # writes Claude Code's Bash as bash.
        call = partial.PARIS.replace("<function=weather>", "<function=Weather>")
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    status, payload = partial.respond(path, call, stream, False)
                    self.assertEqual(status, 200, payload)
                    items, end = output(path, payload, stream)
                    self.assertEqual(calls_of(items), ['{"city":"Paris"}'])
                    self.assertEqual(called_names(path, payload, stream), ["weather"])
                    self.assertIn(end, ("tool_calls", "tool_use", "completed"))

    def test_a_name_that_matches_several_declared_tools_by_case_is_text(self):
        # Read and READ are both declared: READ is its own tool, and read
        # names neither, streamed or not.
        policy = tool_policy({"Read": partial.SCHEMA, "READ": partial.SCHEMA})
        for name, called in (("READ", ["READ"]), ("read", [])):
            call = (
                f"<tool_call>\n<function={name}>\n<parameter=city>\nParis\n"
                "</parameter>\n</function>\n</tool_call>"
            )
            for streaming in (False, True):
                with self.subTest(name=name, streaming=streaming):
                    projector = output_module.StreamingToolCallProjector(
                        policy, "test", syntax=QWEN_XML, streaming=streaming
                    )
                    events = put(projector, call, 1)
                    content, calls, owed = projector.finish(False)
                    self.assertEqual(
                        [value["function"]["name"] for value in calls], called
                    )
                    self.assertEqual(content, "" if called else call)
                    headers = [
                        value["function"]["name"]
                        for kind, value in events + owed
                        if kind == "tool" and "name" in value["function"]
                    ]
                    self.assertEqual(headers, called)

    def test_inkling_s_call_whose_header_names_no_function_is_its_object_s(self):
        # Inkling at temperature 1 may leave the name out of a call's header:
        # the call its object names is returned, held until it closes.
        call = MODEL + JSON_CALL + '{"name":"weather","args":{"city":"Austin"}}' + END
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    harness = InklingHarness(self, call)
                    status, _, payload = harness.request(path, stream)
                    self.assertEqual(status, 200, payload)
                    items, end = output(path, payload, stream)
                    self.assertEqual(calls_of(items), ['{"city":"Austin"}'])
                    self.assertEqual(text_of(items), "")
                    self.assertEqual(called_names(path, payload, stream), ["weather"])
                    self.assertIn(end, ("tool_calls", "tool_use", "completed"))

    def test_the_projector_returns_the_text_however_it_arrives(self):
        policy = tool_policy({"weather": partial.SCHEMA})
        text = (
            "A <tool_call>\n<function=search>\n<parameter=q>\nx\n</parameter>\n"
            "</function>\n</tool_call> B"
        )
        for size in (None, 1, 3, 7):
            with self.subTest(size=size):
                content, calls, events = project(text, policy, size=size)
                self.assertEqual((content, calls), (text, []))
                self.assertEqual(streamed_text(events), text)
                self.assertNotIn("tool", [kind for kind, _ in events])

    def test_each_call_not_returned_is_counted_and_logged(self):
        call = "<tool_call>\n<function=search>\n</function>\n</tool_call>"
        tokenizer, token_ids = partial.character_tokenizer(call)
        harness = fixtures.Harness(
            fixtures.FakeRuntime(
                fixtures.Plan([[token_ids[char]] for char in call + call])
            ),
            tokenizer=tokenizer,
            max_context=8192,
        )
        self.addCleanup(harness.close)
        with mock.patch.object(api, "print_status") as printed:
            status, _, payload = harness.request(
                "POST",
                "/v1/chat/completions",
                partial.weather_request("/v1/chat/completions", False, False),
            )
        self.assertEqual(status, 200, payload)
        lines = [args[0] for args, _ in printed.call_args_list]
        self.assertEqual(
            lines,
            ["Tool calls · request 1 · 2 undeclared tool"],
        )
        self.assertEqual(
            harness.app.status()["tool_calls"]["rejected"],
            {"undeclared": 2, "invalid_arguments": 0, "cut": 0},
        )
        status, _, metrics = harness.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn(
            'splash_tool_calls_rejected_total{reason="undeclared"} 2',
            metrics.decode(),
        )


# The finish reason each dialect reports for each end.
ENDS = {
    "length": {
        "/v1/chat/completions": "length",
        "/v1/messages": "max_tokens",
        "/v1/responses": "incomplete",
    },
    "stop": {
        "/v1/chat/completions": "stop",
        "/v1/messages": "stop_sequence",
        "/v1/responses": "completed",
    },
}


def stop_field(path, stop):
    return {"stop_sequences" if path == "/v1/messages" else "stop": [stop]}


class CutCallTests(unittest.TestCase):
    def test_a_streamed_call_s_arguments_arrive_as_the_model_writes_them(self):
        # Qwen's tags and Inkling's JSON: the arguments arrive in several
        # deltas, the call's header before them.
        value = "x" * 40
        qwen = (
            "<tool_call>\n<function=weather>\n<parameter=city>\n"
            + value
            + "\n</parameter>\n</function>\n</tool_call>"
        )
        inkling = (
            MODEL
            + "weather"
            + JSON_CALL
            + '{"name":"weather","args":{"city":"'
            + value
            + '"}}'
            + END
        )

        def from_inkling(path):
            status, _, payload = InklingHarness(self, inkling).request(path, True)
            return status, payload

        for name, respond in (
            ("qwen", lambda path: partial.respond(path, qwen, True, False)),
            ("inkling", from_inkling),
        ):
            for path in PATHS:
                with self.subTest(model=name, path=path):
                    status, payload = respond(path)
                    self.assertEqual(status, 200, payload)
                    deltas = [
                        line
                        for line in payload.decode().splitlines()
                        if '"arguments":' in line
                        and '"name"' not in line
                        or "input_json_delta" in line
                        or "function_call_arguments.delta" in line
                    ]
                    self.assertGreater(len(deltas), 3)
                    items, _ = output(path, payload, True)
                    self.assertEqual(calls_of(items), ['{"city":"' + value + '"}'])

    def test_a_call_a_stop_sequence_cuts_ends_as_each_mode_ends_one(self):
        # Dropped from a complete response; a stream, which sent the call,
        # ends it incomplete. The finish reason is the stop in both.
        # The probe: stop ["Paris"] inside the call returned finish "stop"
        # with arguments {"city":".
        text = "Checking.\n" + partial.PARIS
        tokenizer, token_ids = partial.character_tokenizer(text)
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    harness = fixtures.Harness(
                        fixtures.FakeRuntime(
                            fixtures.Plan([[token_ids[char]] for char in text])
                        ),
                        tokenizer=tokenizer,
                        max_context=8192,
                    )
                    self.addCleanup(harness.close)
                    status, _, payload = harness.request(
                        "POST",
                        path,
                        partial.weather_request(path, stream, False)
                        | stop_field(path, "Paris"),
                    )
                    self.assertEqual(status, 200, payload)
                    self.assertNotIn(b"<tool_call", payload)
                    items, end = output(path, payload, stream)
                    self.assertEqual(calls_of(items), ['{"city":"'] if stream else [])
                    self.assertEqual(text_of(items), "Checking.\n")
                    self.assertEqual(end, ENDS["stop"][path])
                    self.assertEqual(
                        harness.app.status()["tool_calls"]["rejected"]["cut"], 1
                    )

    def test_a_call_the_token_limit_cuts_ends_as_each_mode_ends_one(self):
        # The text before it and a call that closed before it stay.
        for before, calls in (
            ("Checking.\n", []),
            (partial.PARIS, ['{"city":"Paris"}']),
        ):
            text = before + "<tool_call>\n<function=weather>\n<parameter=city>\nRo"
            for path in PATHS:
                for stream in (False, True):
                    with self.subTest(before=before, path=path, stream=stream):
                        status, payload = partial.respond(
                            path, text, stream, False, "length"
                        )
                        self.assertEqual(status, 200, payload)
                        if not stream:
                            self.assertNotIn(b"Ro", payload.replace(b"Rome", b""))
                        items, end = output(path, payload, stream)
                        self.assertEqual(
                            calls_of(items),
                            calls + (['{"city":"Ro'] if stream else []),
                        )
                        self.assertEqual(
                            text_of(items), before.removesuffix(partial.PARIS)
                        )
                        self.assertEqual(end, ENDS["length"][path])

    def test_inkling_s_call_the_token_limit_cuts_ends_as_each_mode_ends_one(self):
        # The probe: max_tokens inside a call returned {"q":"llamas" with
        # finish length.
        cut = MODEL + "weather" + JSON_CALL + '{"name":"weather","args":{"city":"Pa'
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    harness = InklingHarness(
                        self, MODEL + TEXT + "Checking." + END + cut, reason="length"
                    )
                    status, _, payload = harness.request(path, stream)
                    self.assertEqual(status, 200, payload)
                    items, end = output(path, payload, stream)
                    self.assertEqual(calls_of(items), ['{"city":"Pa'] if stream else [])
                    self.assertEqual(text_of(items), "Checking.")
                    self.assertEqual(end, ENDS["length"][path])

    def test_a_cut_call_is_counted_and_logged(self):
        text = "<tool_call>\n<function=weather>\n<parameter=city>\nRo"
        tokenizer, token_ids = partial.character_tokenizer(text)
        harness = fixtures.Harness(
            fixtures.FakeRuntime(
                fixtures.Plan([[token_ids[char]] for char in text], reason="length")
            ),
            tokenizer=tokenizer,
            max_context=8192,
        )
        self.addCleanup(harness.close)
        with mock.patch.object(api, "print_status") as printed:
            status, _, payload = harness.request(
                "POST",
                "/v1/chat/completions",
                partial.weather_request("/v1/chat/completions", True, False),
            )
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            [args[0] for args, _ in printed.call_args_list],
            ["Tool calls · request 1 · 1 cut before its close"],
        )
        self.assertEqual(
            harness.app.status()["tool_calls"]["rejected"],
            {"undeclared": 0, "invalid_arguments": 0, "cut": 1},
        )


if __name__ == "__main__":
    unittest.main()
