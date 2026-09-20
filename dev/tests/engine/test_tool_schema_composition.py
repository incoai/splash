import copy
import json
import unittest

from llguidance import LLMatcher

from dev.tests.engine import test_structured_tools as structured
from server import output, tool_schema
from server.errors import APIError


class ToolSchemaCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        structured.StructuredToolGrammarTest.setUpClass()
        cls.tokenizer = structured.StructuredToolGrammarTest.tokenizer
        cls.guidance = structured.StructuredToolGrammarTest.guidance

    def check_arguments(self, schema, arguments, invalid):
        original = copy.deepcopy(schema)
        tools = [
            {"type": "function", "function": {"name": "test", "parameters": schema}}
        ]
        policy = tool_schema.normalize_tools(tools, "required", False)[1]
        grammar = tool_schema.tool_grammar(policy, False)
        self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
        shape = policy.argument_schemas["test"]
        names = [name for name in shape["properties"] if name in arguments]
        names += [name for name in arguments if name not in shape["properties"]]
        xml = "<tool_call>\n<function=test>\n"
        for name in names:
            value = arguments[name]
            value_schema = shape["properties"].get(name, shape["additionalProperties"])
            raw = tool_schema.raw_string_schema(value_schema, value_schema)
            encoded = value if isinstance(value, str) and raw else json.dumps(value)
            xml += f"<parameter={name}>\n{encoded}\n</parameter>\n"
        xml += "</function>\n</tool_call>"
        matcher = LLMatcher(self.guidance, grammar)
        tokens = self.tokenizer.encode(xml).ids
        self.assertEqual(matcher.validate_tokens(tokens), len(tokens), xml)
        self.assertTrue(matcher.consume_tokens(tokens))
        self.assertTrue(matcher.is_accepting())
        content, calls = output.parse_tool_calls(xml, 1, policy)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), arguments)
        output.validate_tool_calls(calls, policy)
        projector = output.StreamingToolCallProjector(policy, 1)
        events = []
        for char in xml:
            events.extend(projector.put(char))
        events.extend(projector.finish(content, calls, False))
        streamed = "".join(
            value.get("function", {}).get("arguments", "")
            for kind, value in events
            if kind == "tool"
        )
        self.assertEqual(json.loads(streamed), arguments)
        self.assertEqual(schema, original)
        calls[0]["function"]["arguments"] = json.dumps(invalid)
        with self.assertRaises(APIError):
            output.validate_tool_calls(calls, policy)
        return policy, xml

    def test_cyclic_alternatives_fail_without_recursing(self):
        for keyword in ("anyOf", "oneOf"):
            with self.subTest(keyword=keyword):
                schema = {
                    "$defs": {
                        "node": {
                            keyword: [{"$ref": "#/$defs/node"}, {"type": "string"}]
                        }
                    },
                    "$ref": "#/$defs/node",
                }
                with self.assertRaisesRegex(
                    APIError, "cyclic tool parameter alternatives"
                ) as error:
                    tool_schema.raw_string_schema(schema, schema)
                self.assertEqual(error.exception.status, 400)
                parameters = {
                    "$defs": schema["$defs"],
                    "properties": {"value": {"$ref": "#/$defs/node"}},
                }
                policy = tool_schema.normalize_tools(
                    [
                        {
                            "type": "function",
                            "function": {"name": "test", "parameters": parameters},
                        }
                    ],
                    "required",
                    False,
                )[1]
                with self.assertRaisesRegex(
                    APIError, "cyclic tool parameter alternatives"
                ):
                    tool_schema.tool_grammar(policy, False)

    def test_shared_string_alternatives_are_not_cycles(self):
        schema = {
            "$defs": {"text": {"type": "string"}},
            "anyOf": [{"$ref": "#/$defs/text"}, {"$ref": "#/$defs/text"}],
        }
        self.assertEqual(
            tool_schema.raw_string_schema({"anyOf": schema["anyOf"]}, schema),
            ("raw", None),
        )

    def test_recursive_objects_keep_json_framing_and_validation(self):
        self.check_arguments(
            {
                "type": "object",
                "$defs": {
                    "node": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {"child": {"$ref": "#/$defs/node"}},
                                "additionalProperties": False,
                            },
                        ]
                    }
                },
                "properties": {"value": {"$ref": "#/$defs/node"}},
                "required": ["value"],
            },
            {"value": {"child": {"child": None}}},
            {"value": {"child": "wrong"}},
        )

    def test_root_reference_and_chained_field_reference(self):
        self.check_arguments(
            {
                "$defs": {
                    "text": {"type": "string"},
                    "args": {
                        "type": "object",
                        "properties": {"value": {"$ref": "#/$defs/text"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
                "$ref": "#/$defs/args",
            },
            {"value": "123"},
            {"value": 123},
        )

    def test_root_reference_keeps_sibling_assertions_and_object_union(self):
        schema = {
            "$defs": {
                "base": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "args": {"$ref": "#/$defs/base", "minProperties": 2},
            },
            "$ref": "#/$defs/args",
        }
        self.check_arguments(schema, {"name": "123", "extra": True}, {"name": "123"})
        self.check_arguments(
            {
                "$defs": schema["$defs"],
                "anyOf": [{"$ref": "#/$defs/args"}, {"type": "null"}],
            },
            {"name": "123", "extra": True},
            {"name": "123"},
        )

    def test_intersection_union_and_conditional_fields(self):
        self.check_arguments(
            {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                    {
                        "properties": {"count": {"type": "integer", "minimum": 1}},
                        "required": ["count"],
                    },
                ]
            },
            {"path": "123", "count": 2},
            {"path": "x", "count": 0},
        )
        for keyword in ("anyOf", "oneOf"):
            schema = {
                keyword: [
                    {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"number": {"type": "integer"}},
                        "required": ["number"],
                        "additionalProperties": False,
                    },
                ]
            }
            for args in ({"text": "123"}, {"number": 3}):
                with self.subTest(keyword=keyword, args=args):
                    self.check_arguments(schema, args, {"text": "x", "number": 3})
        self.check_arguments(
            {
                "type": "object",
                "properties": {"kind": {"enum": ["file", "text"]}},
                "required": ["kind"],
                "if": {"properties": {"kind": {"const": "file"}}},
                "then": {
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                "else": {
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            {"kind": "file", "path": "123"},
            {"kind": "file"},
        )

    def test_dynamic_names_and_cross_field_assertions_remain_validated(self):
        self.check_arguments(
            {
                "type": "object",
                "patternProperties": {"^x_": {"type": "integer"}},
                "additionalProperties": False,
                "minProperties": 1,
                "maxProperties": 2,
            },
            {"x_a": 3},
            {"wrong": 3},
        )
        self.check_arguments(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "dependentRequired": {"a": ["b"]},
                "propertyNames": {"pattern": "^[ab]$"},
            },
            {"a": "123", "b": 7},
            {"a": "123"},
        )
        self.check_arguments(
            {"type": "object", "not": {"required": ["forbidden"]}},
            {"allowed": "123"},
            {"forbidden": 1},
        )
        self.check_arguments(
            {"enum": [{"a": "123"}, {"b": 3}]}, {"a": "123"}, {"a": "other"}
        )

    def test_additional_typed_strings_and_local_anchors(self):
        self.check_arguments(
            {"type": "object", "additionalProperties": {"type": "string"}},
            {"extra": "123"},
            {"extra": 123},
        )
        self.check_arguments(
            {
                "$defs": {"text": {"$anchor": "text", "type": "string"}},
                "properties": {"value": {"$ref": "#text"}},
                "required": ["value"],
            },
            {"value": "123"},
            {"value": 123},
        )
        self.check_arguments(
            {
                "$defs": {"a/b": {"type": "string"}},
                "properties": {"value": {"$ref": "#/$defs/a~1b"}},
                "required": ["value"],
            },
            {"value": "123"},
            {"value": 123},
        )

    def test_forbidden_optional_fields_do_not_make_the_object_impossible(self):
        self.check_arguments(
            {
                "type": "object",
                "properties": {"disabled": False},
                "additionalProperties": False,
            },
            {},
            {"disabled": 1},
        )
        self.check_arguments(
            {
                "allOf": [
                    {
                        "properties": {"a": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    {
                        "properties": {"b": {"type": "string"}},
                        "additionalProperties": False,
                    },
                ]
            },
            {},
            {"a": "x"},
        )

    def test_generic_name_rule_cannot_reencode_declared_strings(self):
        policy, _ = self.check_arguments(
            {"properties": {"a": {"type": "string", "const": "hello"}}},
            {"a": "hello", "ab": 3},
            {"a": 1},
        )
        grammar = tool_schema.tool_grammar(policy, False)
        bad = '<tool_call>\n<function=test>\n<parameter=a>\n"hello"\n</parameter>\n</function>\n</tool_call>'
        tokens = self.tokenizer.encode(bad).ids
        self.assertLess(
            LLMatcher(self.guidance, grammar).validate_tokens(tokens), len(tokens)
        )


if __name__ == "__main__":
    unittest.main()
