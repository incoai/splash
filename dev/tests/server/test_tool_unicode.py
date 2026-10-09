import json
import unittest

from dev.tests.tool_output import project, streamed_arguments
from server import output, tool_schema


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
    def test_json_that_decodes_to_lone_surrogates_keeps_its_text(self):
        # Arguments are written back as JSON that the next turn reads; a value
        # that would decode to a lone surrogate keeps its text instead, which
        # writes back as valid JSON.
        for schema in ({"type": "object"}, {"type": "array"}, {"type": "integer"}):
            for raw in (
                r'"\ud800"',
                r'"\udfff\ud800"',
                r'{"nested":[{"\ud800":"x"}]}',
                r'["\udfff"]',
            ):
                text = tool_xml(raw)
                for split in range(0, len(text) + 1, 7):
                    with self.subTest(schema=schema, raw=raw, split=split):
                        projector = output.StreamingToolCallProjector(
                            policy(schema), "test"
                        )
                        events = projector.put(text[:split]) + projector.put(
                            text[split:]
                        )
                        _, calls, owed = projector.finish(False)
                        arguments = calls[0]["function"]["arguments"]
                        self.assertEqual(json.loads(arguments), {"value": raw})
                        self.assertEqual(streamed_arguments(events + owed), arguments)

    def test_valid_pairs_and_literal_escapes_preserve_streaming(self):
        cases = (
            # Any type, a string among them, so a JSON string stays text.
            ({}, r'"🌍"', r'"🌍"'),
            # No string type, so the value is the JSON the text spells.
            ({"type": "integer"}, r'"🌍"', "🌍"),
            (
                {"type": "object"},
                json.dumps({"🌍": ["中文", r"\ud800"]}),
                {"🌍": ["中文", r"\ud800"]},
            ),
            ({"type": "string"}, r"\ud800", r"\ud800"),
            ({"type": "string"}, "中文🌍", "中文🌍"),
        )
        for schema, raw, expected in cases:
            tool_policy = policy(schema)
            text = tool_xml(raw)
            canonical = json.dumps(
                {"value": expected}, ensure_ascii=False, separators=(",", ":")
            )
            for split in range(len(text) + 1):
                with self.subTest(raw=raw, split=split):
                    projector = output.StreamingToolCallProjector(tool_policy, "test")
                    events = projector.put(text[:split]) + projector.put(text[split:])
                    _, calls, owed = projector.finish(False)
                    self.assertEqual(calls[0]["function"]["arguments"], canonical)
                    self.assertEqual(streamed_arguments(events + owed), canonical)

    def test_parameter_names_lose_surrounding_space(self):
        # A name reads up to ">", without the space around it (#293).
        for written, name in (("shift ", "shift"), (" a b\t", "a b"), ("名前", "名前")):
            with self.subTest(name=written):
                _, calls, _ = project(tool_xml("x", written), policy({}))
                self.assertEqual(
                    json.loads(calls[0]["function"]["arguments"]), {name: "x"}
                )


if __name__ == "__main__":
    unittest.main()
