import json
import unittest

from server import output, tool_schema
from server.errors import APIError


def policy(schema):
    return tool_schema.normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": schema},
                    },
                },
            }
        ],
        "auto",
        True,
    )[1]


def tool_xml(value, name="value"):
    return (
        f"<tool_call>\n<function=echo>\n<parameter={name}>\n{value}"
        "\n</parameter>\n</function>\n</tool_call>"
    )


class ToolUnicodeTests(unittest.TestCase):
    def test_all_surrogates_rejected_in_decoded_values_and_keys(self):
        tool_policy = policy({})
        for codepoint in range(0xD800, 0xE000):
            char = chr(codepoint)
            for value in (char, {"nested": [char]}, {char: "value"}):
                with self.subTest(codepoint=codepoint, kind=type(value).__name__):
                    text = tool_xml(json.dumps(value))
                    with self.assertRaisesRegex(APIError, "invalid Unicode") as error:
                        output.parse_tool_calls(text, "test", tool_policy)
                    self.assertEqual(error.exception.status, 500)
                    self.assertEqual(error.exception.code, "invalid_model_output")

    def test_invalid_tools_fail_before_emitting_bad_arguments_at_every_split(self):
        cases = (
            ({}, r'"\ud800"'),
            ({}, r'"\udfff"'),
            ({}, r'"\udfff\ud800"'),
            ({}, r'{"nested":[{"\ud800":"x"}]}'),
            ({"type": "string"}, "valid prefix\ud800suffix"),
        )
        for schema, raw in cases:
            tool_policy = policy(schema)
            text = tool_xml(raw)
            for split in range(len(text) + 1):
                with self.subTest(schema=schema, raw=ascii(raw), split=split):
                    projector = output.StreamingToolCallProjector(tool_policy, "test")
                    events = []
                    with self.assertRaisesRegex(APIError, "invalid Unicode") as error:
                        events.extend(projector.put(text[:split]))
                        events.extend(projector.put(text[split:]))
                    self.assertEqual(error.exception.code, "invalid_model_output")
                    emitted = "".join(
                        value.get("function", {}).get("arguments", "")
                        for kind, value in events
                        if kind == "tool"
                    )
                    self.assertNotIn(r"\ud800", emitted)
                    self.assertNotIn(r"\udfff", emitted)
                    self.assertEqual(projector.closed_calls, [])

    def test_valid_pairs_and_literal_escapes_preserve_streaming(self):
        cases = (
            ({}, r'"\ud83c\udf0d"', "🌍"),
            ({}, json.dumps({"🌍": ["中文", r"\ud800"]}), {"🌍": ["中文", r"\ud800"]}),
            ({"type": "string"}, r"\ud800", r"\ud800"),
            ({"type": "string"}, "中文🌍", "中文🌍"),
        )
        for schema, raw, expected in cases:
            tool_policy = policy(schema)
            text = tool_xml(raw)
            content, calls = output.parse_tool_calls(text, "test", tool_policy)
            canonical = calls[0]["function"]["arguments"]
            self.assertEqual(json.loads(canonical), {"value": expected})
            output.validate_tool_calls(calls, tool_policy)
            for split in range(len(text) + 1):
                with self.subTest(raw=raw, split=split):
                    projector = output.StreamingToolCallProjector(tool_policy, "test")
                    events = projector.put(text[:split]) + projector.put(text[split:])
                    projector.finish(content, calls, False)
                    emitted = "".join(
                        value.get("function", {}).get("arguments", "")
                        for kind, value in events
                        if kind == "tool"
                    )
                    self.assertEqual(emitted, canonical)

    def test_invalid_parameter_names_and_final_validation(self):
        tool_policy = policy({})
        text = tool_xml('"ok"', "bad\ud800")
        with self.assertRaisesRegex(APIError, "invalid Unicode"):
            output.parse_tool_calls(text, "test", tool_policy)
        projector = output.StreamingToolCallProjector(tool_policy, "test")
        with self.assertRaisesRegex(APIError, "invalid Unicode"):
            projector.put(text)
        calls = [{"function": {"name": "echo", "arguments": r'{"value":"\ud800"}'}}]
        with self.assertRaisesRegex(APIError, "invalid Unicode"):
            output.validate_tool_calls(calls, tool_policy)
