import copy
import json
import unittest
from unittest import mock

from llguidance import LLMatcher

from dev.tests.engine import test_structured_tools as structured
from dev.tests.test_server import no_signed_thinking
from dev.tests.tool_output import project, streamed_arguments
from server import api_shapes, output, tool_schema
from server.errors import APIError


class ToolSchemaCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        structured.StructuredToolGrammarTest.setUpClass()
        cls.tokenizer = structured.StructuredToolGrammarTest.tokenizer
        cls.guidance = structured.StructuredToolGrammarTest.guidance

    def check_arguments(self, schema, arguments, invalid, *, order=None):
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
        if order is not None:
            self.assertEqual(set(order), set(names))
            names = order
        xml = "<tool_call>\n<function=test>\n"
        for name in names:
            value = arguments[name]
            value_schema = shape["properties"].get(name, shape["additionalProperties"])
            raw = tool_schema.raw_string_schema(value_schema)
            encoded = value if isinstance(value, str) and raw else json.dumps(value)
            xml += f"<parameter={name}>\n{encoded}\n</parameter>\n"
        xml += "</function>\n</tool_call>"
        matcher = LLMatcher(self.guidance, grammar)
        tokens = self.tokenizer.encode(xml).ids
        self.assertEqual(matcher.validate_tokens(tokens), len(tokens), xml)
        self.assertTrue(matcher.consume_tokens(tokens))
        self.assertTrue(matcher.is_accepting())
        _, calls, events = project(xml, policy, size=1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), arguments)
        self.assertEqual(streamed_arguments(events), calls[0]["function"]["arguments"])
        output.validate_tool_calls(calls, policy)
        self.assertEqual(schema, original)
        calls[0]["function"]["arguments"] = json.dumps(invalid)
        with self.assertRaises(APIError):
            output.validate_tool_calls(calls, policy)
        return policy, xml

    def test_early_optional_fields_remain_available_after_required_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "type": {"enum": ["txt", "group"]},
            },
            "required": ["name", "type"],
            "additionalProperties": False,
        }
        arguments = {
            "content": "正文\nSecond line",
            "name": "Note",
            "tags": [],
            "type": "txt",
        }
        for order in (list(schema["properties"]), ["name", "type", "content", "tags"]):
            with self.subTest(order=order):
                policy, xml = self.check_arguments(
                    schema, arguments, {"content": "missing name/type"}, order=order
                )
                grammar = tool_schema.tool_grammar(policy, False)
                for bad in (
                    xml.replace("<parameter=name>\nNote\n</parameter>\n", ""),
                    xml.replace(
                        "<parameter=name>\nNote\n</parameter>\n",
                        "<parameter=name>\nNote\n</parameter>\n" * 2,
                    ),
                ):
                    matcher = LLMatcher(self.guidance, grammar)
                    tokens = self.tokenizer.encode(bad).ids
                    self.assertLess(matcher.validate_tokens(tokens), len(tokens))
        self.check_arguments(schema, {"name": "Folder", "type": "group"}, {})

    def test_note_content_survives_protocol_conversion_and_streaming(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "content": {"type": "string", "description": "Optional note body"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        function = {"name": "test", "parameters": schema}
        chat = {"tools": [{"type": "function", "function": function}]}
        responses, _ = api_shapes.responses_to_chat_body(
            {"tools": [{"type": "function", **function}]},
            [{"role": "user", "content": "Create a note"}],
        )
        messages, _ = api_shapes.anthropic_to_chat_body(
            {
                "model": "test",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "Create a note"}],
                "tools": [{"name": "test", "input_schema": schema}],
            },
            thinking_resolver=no_signed_thinking,
        )
        for body in (chat, responses, messages):
            with self.subTest(body=body):
                converted = body["tools"][0]["function"]["parameters"]
                self.assertEqual(converted, schema)
                self.check_arguments(
                    converted,
                    {"name": "Release", "content": '第一行\n"quoted"\nliteral \\n'},
                    {"content": "missing required name"},
                )
                # Optional means optional: folder/group creation must not be
                # forced to invent a note body by the transport or grammar.
                self.check_arguments(converted, {"name": "Folder"}, {})

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
                    tool_schema.raw_string_schema(schema)
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
            "$defs": {
                "text": {"type": "string"},
                "choice": {
                    "anyOf": [{"$ref": "#/$defs/text"}, {"$ref": "#/$defs/text"}]
                },
            },
            "$ref": "#/$defs/choice",
        }
        self.assertEqual(tool_schema.raw_string_schema(schema), ("raw", None))

    def test_shared_references_are_projected_once(self):
        # Two references per level to the next definition used to double the
        # work at every level; a 1.6 KB schema took hours.
        depth = 14
        definitions = {
            f"d{i}": {
                "anyOf": [{"$ref": f"#/$defs/d{i + 1}"}, {"$ref": f"#/$defs/d{i + 1}"}]
            }
            for i in range(depth)
        }
        definitions[f"d{depth}"] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        field = {"$ref": "#/$defs/d0"}
        lookup = tool_schema._lookup_tool_reference
        for schema, arguments, invalid in (
            ({"$defs": definitions, **field}, {"x": "a"}, {"x": 1}),
            (
                {"$defs": definitions, "properties": {"value": field}},
                {"value": {"x": "a"}},
                {"value": {"x": 1}},
            ),
        ):
            tool = {"type": "function", "function": {"name": "t", "parameters": schema}}
            policy = tool_schema.normalize_tools([tool], "required", False)[1]
            with (
                self.subTest(schema=sorted(schema)),
                mock.patch.object(
                    tool_schema, "_lookup_tool_reference", side_effect=lookup
                ) as counted,
            ):
                tool_schema.tool_grammar(policy, False)
            # A few lookups per reference, rather than one per path.
            self.assertLess(counted.call_count, 4 * (depth + 1))
            self.check_arguments(schema, arguments, invalid)

    def test_framing_is_bounded_across_the_tools_of_a_request(self):
        # Composition and root copies can make the framed schemas quadratic or
        # exponential in the tool schemas; one budget covers all tools.
        def policy(*schemas):
            tools = [
                {"type": "function", "function": {"name": f"t{i}", "parameters": s}}
                for i, s in enumerate(schemas)
            ]
            return tool_schema.normalize_tools(tools, "auto", True)[1]

        wide = {
            "anyOf": [
                {
                    "properties": {f"b{i}_{j}": {"type": "integer"} for j in range(5)},
                    "additionalProperties": {"type": "string", "description": str(i)},
                }
                for i in range(30)
            ]
        }
        copies = {
            "$defs": {"x": {"type": "integer"}},
            "properties": {f"p{i}": {"$ref": "#/$defs/x"} for i in range(200)},
        }
        nested = {"$defs": {"d16": {"properties": {"x": {"type": "integer"}}}}}
        for i in range(16):
            ref = {"$ref": f"#/$defs/d{i + 1}"}
            narrower = {"allOf": [ref, {"properties": {"x": {"minimum": i}}}]}
            nested["$defs"][f"d{i}"] = {"anyOf": [ref, narrower]}
        nested["$ref"] = "#/$defs/d0"
        half = {"properties": {"x": {"description": "d" * 12_000}}}
        with mock.patch.object(tool_schema, "MAX_FRAMED_SCHEMA_BYTES", 20_000):
            tool_schema.tool_grammar(policy(half), False)
            for name, schemas in (
                ("wide union", [wide]),
                ("root copies", [copies]),
                ("nested alternatives", [nested]),
                ("two tools", [half, half]),
            ):
                with (
                    self.subTest(name),
                    self.assertRaisesRegex(APIError, "too complex") as caught,
                ):
                    tool_schema.tool_grammar(policy(*schemas), False)
                self.assertEqual(caught.exception.status, 400)

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

    def test_extra_names_may_start_like_unused_declared_names(self):
        schema = {
            "properties": {"url": {"type": "string"}, "ab": {"type": "integer"}},
            "additionalProperties": {"type": "integer"},
        }
        for extra in ("user-agent", "urls", "u", "a", "abc"):
            with self.subTest(extra=extra):
                self.check_arguments(schema, {extra: 1}, {extra: "1"})

        def rejected(policy, xml):
            matcher = LLMatcher(self.guidance, tool_schema.tool_grammar(policy, False))
            tokens = self.tokenizer.encode(xml).ids
            return matcher.validate_tokens(tokens) < len(tokens)

        # Declared names still appear once, and forbidden ones not at all.
        policy, xml = self.check_arguments(
            schema, {"url": "x", "ab": 1, "user-agent": 2}, {"url": 1}
        )
        repeated = "<parameter=url>\nx\n</parameter>\n</function>"
        self.assertTrue(rejected(policy, xml.replace("</function>", repeated)))
        policy, xml = self.check_arguments(
            {"properties": {"secret": False}, "additionalProperties": {}},
            {"secrets": 1},
            {"secret": 1},
        )
        self.assertTrue(rejected(policy, xml.replace("secrets", "secret")))


if __name__ == "__main__":
    unittest.main()
