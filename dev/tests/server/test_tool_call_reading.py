"""Tool calls as they read out of model output.

Each expectation is the content and calls an output reads as, or a value as
it converts by its declared types. The edge cases at the end say why they
read as they do.
"""

import json
import unittest

from dev.tests.server_fixtures import FakeRuntime, Harness, anthropic_body
from dev.tests.tool_output import (
    argument_grammar,
    project,
    streamed_arguments,
    streamed_text,
    tool_policy,
)
from server import json_codec
from server import output as model_output
from server import server as api
from server.output import MAX_ARGUMENTS_DEPTH

# The deepest a parameter value reads as JSON: one level less than its call's
# arguments.
MAX_VALUE_DEPTH = MAX_ARGUMENTS_DEPTH - 1

SCHEMAS = {
    "roll_cut": {
        "type": "object",
        "properties": {
            "cut_at": {"type": "string"},
            "shift": {"type": "number"},
            "new_time": {"type": "string"},
            "item": {"type": "string"},
        },
    },
    "cut_at": {
        "type": "object",
        "properties": {"time": {"type": ["string", "number"]}},
        "required": ["time"],
    },
}


def call(name, *arguments):
    return (
        f"<tool_call>\n<function={name}>\n"
        + "".join(
            f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments
        )
        + "</function>\n</tool_call>"
    )


PARSES = {
    "the order the model writes (#293)": (
        call("roll_cut", ("item", "s1"), ("shift", "-0.5")),
        [("roll_cut", {"item": "s1", "shift": -0.5})],
    ),
    "a union with a string (#294)": (
        call("cut_at", ("time", "12:00")),
        [("cut_at", {"time": "12:00"})],
    ),
    "no </tool_call>": (
        "<tool_call>\n<function=roll_cut>\n<parameter=item>\ns1\n</parameter>\n"
        "</function>",
        [("roll_cut", {"item": "s1"})],
    ),
    "the call's close in place of its function's": (
        "<tool_call>\n<function=roll_cut>\n<parameter=item>\ns1\n</parameter>\n"
        "</tool_call>",
        [("roll_cut", {"item": "s1"})],
    ),
    "space around the names": (
        "<tool_call>\n<function= roll_cut >\n<parameter= item >\ns1\n"
        "</parameter>\n</function>\n</tool_call>",
        [("roll_cut", {"item": "s1"})],
    ),
    "a tool not offered": (
        call("search", ("limit", "5"), ("query", "a b")),
        [("search", {"limit": 5, "query": "a b"})],
    ),
    "no function name": (call("", ("item", "x")), []),
    "a value keeps inner newlines and markup": (
        call("roll_cut", ("item", "\nline 1\n<b>line 2</b>\n")),
        [("roll_cut", {"item": "\nline 1\n<b>line 2</b>\n"})],
    ),
    "a value holds the tags in any other order": (
        call(
            "roll_cut",
            (
                "item",
                '<p>\n</parameter>\n</p>\nEND = "</function>"\n'
                'OPEN = "<parameter=" "<tool_call>"\n</parameter>',
            ),
        ),
        [
            (
                "roll_cut",
                {
                    "item": '<p>\n</parameter>\n</p>\nEND = "</function>"\n'
                    'OPEN = "<parameter=" "<tool_call>"\n</parameter>'
                },
            )
        ],
    ),
    "output that ends inside a call": (
        "<tool_call>\n<function=roll_cut>\n<parameter=item>\ns1\n</parameter>\n",
        [("roll_cut", {"item": "s1"})],
    ),
}

# The value a parameter's text converts to by its declared types: the JSON
# the text spells, but a string where one may be the value and the JSON is
# not of another declared type.
CONVERSIONS = (
    ({"type": "string"}, "123", "123"),
    ({"type": "string"}, "true", "true"),
    ({"type": "integer"}, "42", 42),
    ({"type": "integer"}, " 12 ", 12),
    ({"type": "integer"}, "1_000", "1_000"),
    ({"type": "integer"}, "+5", "+5"),
    ({"type": "integer"}, "012", "012"),
    ({"type": "integer"}, "4.5", 4.5),
    ({"type": "integer"}, "abc", "abc"),
    ({"type": "number"}, "-0.5", -0.5),
    ({"type": "number"}, "2.0", 2.0),
    ({"type": "number"}, "1e3", 1000.0),
    ({"type": "number"}, "12345678901234567890", 12345678901234567890),
    ({"type": "number"}, "inf", "inf"),
    ({"type": "number"}, "NaN", "NaN"),
    ({"type": "boolean"}, "true", True),
    ({"type": "boolean"}, "True", True),
    ({"type": "boolean"}, " False ", False),
    ({"type": "boolean"}, "TRUE", "TRUE"),
    ({"type": ["integer", "null"]}, "None", None),
    ({"type": "integer"}, "True", "True"),
    ({"type": ["string", "boolean"]}, "True", "True"),
    ({"type": ["string", "null"]}, "None", "None"),
    ({"type": "object"}, '{"a": 1}', {"a": 1}),
    ({"type": "object"}, "{", "{"),
    ({"type": "array"}, "[1, 2]", [1, 2]),
    ({"type": "array"}, '"x"', "x"),
    ({"type": ["string", "null"]}, "null", None),
    ({"type": ["string", "null"]}, "NULL", "NULL"),
    ({"type": ["string", "null"]}, '"null"', '"null"'),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "null", None),
    ({"oneOf": [{"type": "string"}, {"type": "null"}]}, "none", "none"),
    ({"type": ["string", "number"]}, "12:00", "12:00"),
    ({"type": ["string", "number"]}, "12", 12),
    ({"type": ["string", "integer"]}, "0800", "0800"),
    ({"type": ["string", "boolean"]}, "1", "1"),
    ({"type": ["string", "object"]}, "123", "123"),
    ({"type": ["string", "object"]}, '{"a": 1}', {"a": 1}),
    ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "abc", "abc"),
    ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "7", 7),
    ({"anyOf": [{"type": "integer"}, {}]}, "x", "x"),
    ({"allOf": [{"type": "integer"}, {"minimum": 0}]}, "5", 5),
    ({"enum": [1, 2, "x"]}, "1", 1),
    ({"enum": [1, 2, "x"]}, "x", "x"),
    ({}, '{"a": 1}', {"a": 1}),
    ({}, "7", 7),
    ({}, "seven", "seven"),
    ({}, "True", "True"),
)


def unique_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) < len(keys):
        raise AssertionError(f"repeated keys: {keys}")
    return dict(pairs)


class ToolCallReadingTests(unittest.TestCase):
    def read(self, text, schemas=None):
        """The content and calls read from `text`, the same however the text
        arrives, and streamed as they are reported, each object's keys
        once."""
        policy = tool_policy(SCHEMAS if schemas is None else schemas)
        results = set()
        for size in (None, 1, 5):
            content, calls, events = project(text, policy, size=size)
            self.assertEqual(streamed_text(events), content)
            self.assertEqual(
                streamed_arguments(events),
                "".join(call["function"]["arguments"] for call in calls),
            )
            results.add(json.dumps([content, calls]))
        self.assertEqual(len(results), 1)
        return content, [
            (
                call["function"]["name"],
                json.loads(
                    call["function"]["arguments"], object_pairs_hook=unique_keys
                ),
            )
            for call in calls
        ]

    def test_outputs_read_into_calls_and_content(self):
        for name, (text, calls) in PARSES.items():
            with self.subTest(name):
                self.assertEqual(self.read(text), ("", calls))
        content, calls = self.read("Let me cut it.\n\n" + call("cut_at", ("time", "1")))
        self.assertEqual(
            (content.strip(), calls), ("Let me cut it.", [("cut_at", {"time": 1})])
        )
        # A </think> in text is dropped.
        content, calls = self.read("Sure.</think> Here" + call("roll_cut"))
        self.assertEqual((content, calls), ("Sure. Here", [("roll_cut", {})]))

    def test_values_convert_by_their_declared_types(self):
        for schema, text, value in CONVERSIONS:
            with self.subTest(schema=schema, text=text):
                tool = {"type": "object", "properties": {"value": schema}}
                _, calls = self.read(call("f", ("value", text)), {"f": tool})
                self.assertEqual(calls, [("f", {"value": value})])
                self.assertIs(type(calls[0][1]["value"]), type(value))

    def test_tool_arguments_always_serialize_as_strict_json(self):
        nested = "[" * 10000 + "0" + "]" * 10000
        huge_integer = "9" * 5000
        text = (
            "<tool_call>\n<function=f>\n"
            "<parameter=constant>\nNaN\n</parameter>\n"
            "<parameter=overflow>\n1e10000\n</parameter>\n"
            f"<parameter=integer>\n{huge_integer}\n</parameter>\n"
            f"<parameter=nested>\n{nested}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls, _ = project(text, tool_policy({"f": {}}))
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(
            arguments,
            {
                "constant": "NaN",
                "overflow": "1e10000",
                "integer": huge_integer,
                "nested": nested,
            },
        )

    def test_json_values_have_an_explicit_nesting_limit(self):
        allowed = "[" * MAX_VALUE_DEPTH + "0" + "]" * MAX_VALUE_DEPTH
        deep_array = "[" * (MAX_VALUE_DEPTH + 1) + "0" + "]" * (MAX_VALUE_DEPTH + 1)
        deep_object = (
            '{"value":' * (MAX_VALUE_DEPTH + 1) + "0" + "}" * (MAX_VALUE_DEPTH + 1)
        )
        escaped = r"brackets in a string: \"[{]}\"" * (MAX_VALUE_DEPTH + 1)
        convert = model_output.convert_value
        self.assertIsInstance(convert(allowed, {"array"}), list)
        # Too deep to write back, the value keeps its text.
        self.assertEqual(convert(deep_array, {"array"}), deep_array)
        self.assertEqual(convert(deep_object, {"object"}), deep_object)
        # Text that reads as none of the types is the JSON value it spells.
        self.assertEqual(convert(json.dumps(escaped), {"integer"}), escaped)
        # The arguments of a call holding the deepest value read back.
        _, calls, _ = project(
            "<tool_call>\n<function=f>\n"
            f"<parameter=value>\n{allowed}\n</parameter>\n"
            "</function>\n</tool_call>",
            tool_policy({"f": {}}),
        )
        arguments = json_codec.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments, {"value": json.loads(allowed)})

    def test_the_deepest_arguments_read_back_in_a_messages_history(self):
        # A Messages history carries a call's arguments as an object five
        # levels into the request's body, deeper than any other request does.
        harness = Harness(FakeRuntime())
        self.addCleanup(harness.close)

        def history(arguments):
            return anthropic_body(
                tools=[{"name": "f", "input_schema": {"type": "object"}}],
                messages=[
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "f",
                                "input": arguments,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "ok",
                            }
                        ],
                    },
                ],
            )

        for depth in (MAX_VALUE_DEPTH, MAX_VALUE_DEPTH + 1):
            with self.subTest(depth=depth):
                value = "[" * depth + "0" + "]" * depth
                _, calls, _ = project(
                    "<tool_call>\n<function=f>\n"
                    f"<parameter=value>\n{value}\n</parameter>\n"
                    "</function>\n</tool_call>",
                    tool_policy({"f": {}}),
                )
                # The input a complete Messages response holds.
                arguments = json_codec.loads(calls[0]["function"]["arguments"])
                # A value one level deeper keeps its text.
                expected = json.loads(value) if depth == MAX_VALUE_DEPTH else value
                self.assertEqual(arguments, {"value": expected})
                status, _, payload = harness.request(
                    "POST", "/v1/messages", history(arguments)
                )
                self.assertEqual(status, 200, payload)
        # Arguments one level deeper than the server writes would not read back.
        deeper = {
            "value": json.loads(
                "[" * MAX_ARGUMENTS_DEPTH + "0" + "]" * MAX_ARGUMENTS_DEPTH
            )
        }
        self.assertEqual(
            harness.request("POST", "/v1/messages", history(deeper))[0], 400
        )

    def test_tool_parser_preserves_schema_typed_strings(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["text", "count"],
        }
        text = (
            "<tool_call>\n<function=echo>\n"
            "<parameter=text>\n123\n</parameter>\n"
            "<parameter=count>\n3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls, _ = project(text, tool_policy({"echo": schema}))
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"text": "123", "count": 3},
        )

    def test_tool_parser_preserves_raw_string_edge_whitespace(self):
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        text = (
            "<tool_call>\n<function=echo>\n"
            '<parameter=text>\n "line"\n\n</parameter>\n'
            "</function>\n</tool_call>"
        )
        _, calls, _ = project(text, tool_policy({"echo": schema}))
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"text": ' "line"\n'},
        )

    def test_tool_parser_handles_mixed_parameter_types(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
                "flags": {"type": "array", "items": {"type": "boolean"}},
                "maybe": {"type": ["number", "null"]},
            },
            "required": ["text", "count", "flags", "maybe"],
            "additionalProperties": False,
        }
        raw = (
            "visible prefix "
            "<tool_call>\n<function=echo>\n"
            "<parameter=text>\n 3 \n</parameter>\n"
            "<parameter=count>\n3\n</parameter>\n"
            "<parameter=flags>\n[true,false]\n</parameter>\n"
            "<parameter=maybe>\nnull\n</parameter>\n"
            "</function>\n</tool_call> visible suffix"
        )
        content, calls, _ = project(raw, tool_policy({"echo": schema}), "fuzz")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"text": " 3 ", "count": 3, "flags": [True, False], "maybe": None},
        )
        self.assertEqual(content, "visible prefix  visible suffix")

    def test_tool_parser_preserves_enum_edge_whitespace(self):
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string", "enum": [" edge "]}},
        }
        text = (
            "<tool_call>\n<function=echo>\n"
            "<parameter=text>\n edge \n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls, _ = project(text, tool_policy({"echo": schema}))
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"text": " edge "}
        )
        # An enumerated value may hold the tags, but not the close the grammar
        # writes after a value.
        for value, framing in (
            ("a</parameter>b", False),
            ("x<parameter=y", False),
            ("f</function>g", False),
            ("bad\n</parameter>\nvalue", True),
        ):
            schema = {
                "type": "object",
                "properties": {"text": {"type": "string", "enum": [value]}},
            }
            with self.subTest(value=value):
                if not framing:
                    self.assertIn(json.dumps(value), argument_grammar(schema))
                    continue
                with self.assertRaisesRegex(api.APIError, "XML framing"):
                    argument_grammar(schema)

    def test_tool_parser_keeps_nested_closing_tags_inside_raw_value(self):
        text = (
            "<tool_call>\n<function=echo>\n<parameter=x>\nhello"
            "</function>\n</tool_call>world\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        content, calls, _ = project(text, tool_policy({"echo": {}}))
        self.assertEqual(content, "")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"x": "hello</function>\n</tool_call>world"},
        )

    def test_edge_cases(self):
        # Calls open only where the template writes them: the tags in prose,
        # and a call written as JSON, stay text.
        for prose in (
            "Fix the </parameter> and <function= handling.\n",
            "Wrap calls in <tool_call> tags with <function=NAME> inside.\n",
            '<tool_call>\n{"name": "roll_cut", "arguments": {}}\n</tool_call>\n',
        ):
            with self.subTest(prose=prose):
                self.assertEqual(
                    self.read(prose + call("roll_cut", ("item", "x"))),
                    (prose, [("roll_cut", {"item": "x"})]),
                )
        # A repeated parameter keeps its first value, which has streamed, so
        # the arguments stay valid JSON.
        repeated = call("roll_cut", ("item", "a"), ("shift", "1"), ("item", "b"))
        self.assertEqual(
            self.read(repeated)[1], [("roll_cut", {"item": "a", "shift": 1})]
        )
        # A const converts as an enum of its value.
        schema = {
            "type": "object",
            "properties": {"n": {"const": 5}, "on": {"const": True}},
        }
        self.assertEqual(
            self.read(call("f", ("n", "5"), ("on", "true")), {"f": schema})[1],
            [("f", {"n": 5, "on": True})],
        )
        # Text after a call stays content (#231), with or without the call's
        # close.
        for closed in (
            call("roll_cut"),
            call("roll_cut").removesuffix("\n</tool_call>"),
        ):
            with self.subTest(closed=closed):
                self.assertEqual(self.read(closed + "\nDone.")[0].strip(), "Done.")
        # A call the token limit cuts keeps the arguments it streamed.
        cut = (
            "<tool_call>\n<function=roll_cut>\n<parameter=item>\ns1\n"
            "</parameter>\n<parameter=cut_at>\n12:"
        )
        _, calls, events = project(cut, tool_policy(SCHEMAS), incomplete=True)
        arguments = '{"item":"s1","cut_at":"12:'
        self.assertEqual(
            [(c["function"]["name"], c["function"]["arguments"]) for c in calls],
            [("roll_cut", arguments)],
        )
        self.assertEqual(streamed_arguments(events), arguments)
        # Types are read through local references, a parameter's own and the
        # one a model such as pydantic's recursive one puts at the top.
        point = {
            "type": "object",
            "properties": {"point": {"$ref": "#/$defs/Point"}},
            "$defs": {"Point": {"type": "object"}},
        }
        node = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "size": {"type": "integer"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Node"},
                        },
                    },
                }
            },
            "$ref": "#/$defs/Node",
        }
        for schema, arguments, expected in (
            (point, (("point", '{"x": 1}'),), {"point": {"x": 1}}),
            (
                node,
                (("name", "root"), ("size", "3"), ("children", "[]")),
                {"name": "root", "size": 3, "children": []},
            ),
        ):
            with self.subTest(schema=schema):
                self.assertEqual(
                    self.read(call("f", *arguments), {"f": schema})[1],
                    [("f", expected)],
                )


if __name__ == "__main__":
    unittest.main()
