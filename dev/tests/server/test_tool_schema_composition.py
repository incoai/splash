import copy
import json
import unittest
from unittest import mock

from llguidance import LLMatcher

from dev.tests.server_fixtures import (
    FakeConstraintFactory,
    FakeRuntime,
    HarnessTestCase,
    chat_body,
    grammar_tokenizers,
    no_signed_thinking,
)
from dev.tests.tool_output import (
    argument_grammar,
    project,
    streamed_arguments,
    tool_policy,
)
from server import api_shapes, tool_schema
from server.errors import APIError
from server.tool_schema import _grammar_compatible_schema


def strict_policy(*schemas):
    tools = [
        {
            "type": "function",
            "function": {"name": f"t{index}", "parameters": schema, "strict": True},
        }
        for index, schema in enumerate(schemas)
    ]
    return tool_schema.normalize_tools(tools, "required", False)[1]


class ToolSchemaCompositionTests(HarnessTestCase):
    """A strict tool's arguments framed through schema composition, as its
    grammar generates them and the projector reads them back."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.guidance = grammar_tokenizers()

    def accepts(self, policy, xml):
        matcher = LLMatcher(self.guidance, tool_schema.tool_grammar(policy, False))
        tokens = self.tokenizer.encode(xml).ids
        return (
            matcher.validate_tokens(tokens) == len(tokens)
            and matcher.consume_tokens(tokens)
            and matcher.is_accepting()
        )

    def check_arguments(self, schema, arguments, expected=None):
        """Write `arguments` in the strict grammar's order, a string as its
        text and any other value as JSON, check that the grammar takes them,
        and read them back: `expected`, or the arguments themselves."""
        original = copy.deepcopy(schema)
        policy = strict_policy(schema)
        self.assertFalse(
            LLMatcher.validate_grammar(
                tool_schema.tool_grammar(policy, False), self.guidance
            )
        )
        shape = policy.argument_schemas["t0"]
        names = [name for name in shape["properties"] if name in arguments]
        names += [name for name in arguments if name not in shape["properties"]]
        xml = "<tool_call>\n<function=t0>\n"
        for name in names:
            value = arguments[name]
            encoded = value if isinstance(value, str) else json.dumps(value)
            xml += f"<parameter={name}>\n{encoded}\n</parameter>\n"
        xml += "</function>\n</tool_call>"
        self.assertTrue(self.accepts(policy, xml), xml)
        _, calls, events = project(xml, policy, size=1)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            arguments if expected is None else expected,
        )
        self.assertEqual(streamed_arguments(events), calls[0]["function"]["arguments"])
        self.assertEqual(schema, original)
        return policy, xml

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
                )
                # Optional means optional: folder/group creation must not be
                # forced to invent a note body by the transport or grammar.
                self.check_arguments(converted, {"name": "Folder"})

    def test_cyclic_and_shared_alternatives_are_read_without_recursing(self):
        for keyword in ("anyOf", "oneOf"):
            with self.subTest(keyword=keyword):
                definitions = {
                    "node": {keyword: [{"$ref": "#/$defs/node"}, {"type": "string"}]}
                }
                node = {"$defs": definitions, "$ref": "#/$defs/node"}
                self.assertEqual(tool_schema.schema_types(node, node), {"string"})
                self.check_arguments(
                    {
                        "$defs": definitions,
                        "properties": {"value": {"$ref": "#/$defs/node"}},
                    },
                    {"value": "text"},
                )
        shared = {
            "$defs": {
                "text": {"type": "string"},
                "choice": {
                    "anyOf": [{"$ref": "#/$defs/text"}, {"$ref": "#/$defs/text"}]
                },
            },
            "$ref": "#/$defs/choice",
        }
        self.assertEqual(tool_schema.schema_types(shared, shared), {"string"})

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
        for schema, arguments in (
            ({"$defs": definitions, **field}, {"x": "a"}),
            (
                {"$defs": definitions, "properties": {"value": field}},
                {"value": {"x": "a"}},
            ),
        ):
            policy = strict_policy(schema)
            with (
                self.subTest(schema=sorted(schema)),
                mock.patch.object(
                    tool_schema, "_lookup_tool_reference", side_effect=lookup
                ) as counted,
            ):
                tool_schema.tool_grammar(policy, False)
            # A few lookups per reference, rather than one per path.
            self.assertLess(counted.call_count, 4 * (depth + 1))
            self.check_arguments(schema, arguments)

    def test_shared_enumerations_are_read_once(self):
        # Two references per level to an enumeration below: read along every
        # path, deciding the values would take 2**40 steps.
        depth = 40
        definitions = {
            f"d{i}": {
                "anyOf": [{"$ref": f"#/$defs/d{i + 1}"}, {"$ref": f"#/$defs/d{i + 1}"}]
            }
            for i in range(depth)
        }
        definitions[f"d{depth}"] = {"enum": ["a", "b"]}
        schema = {"$defs": definitions, "properties": {"value": {"$ref": "#/$defs/d0"}}}
        grammar = tool_schema.tool_grammar(strict_policy(schema), False)
        arguments = json.loads(grammar)["grammars"][1]["lark_grammar"]
        self.assertIn('("a" | "b")', arguments)
        self.check_arguments(schema, {"value": "b"})

    def test_framing_is_bounded_across_the_tools_of_a_request(self):
        # Composition and root copies can make the framed schemas quadratic or
        # exponential in the tool schemas; one budget covers all tools.
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
            tool_schema.tool_grammar(strict_policy(half), False)
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
                    tool_schema.tool_grammar(strict_policy(*schemas), False)
                self.assertEqual(caught.exception.status, 400)

    def test_recursive_objects_keep_json_framing(self):
        schema = {
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
        }
        policy, xml = self.check_arguments(
            schema, {"value": {"child": {"child": None}}}
        )
        self.assertFalse(
            self.accepts(policy, xml.replace('{"child": null}', '"wrong"'))
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
        written = {"name": "123", "extra": True}
        self.check_arguments(schema, written)
        self.check_arguments(
            {
                "$defs": schema["$defs"],
                "anyOf": [{"$ref": "#/$defs/args"}, {"type": "null"}],
            },
            written,
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
            for arguments in ({"text": "123"}, {"number": 3}):
                with self.subTest(keyword=keyword, arguments=arguments):
                    self.check_arguments(schema, arguments)
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
            {"kind": "file", "path": "src/main.py"},
        )

    def test_dynamic_names_and_cross_field_assertions_compile(self):
        # Framing takes these schemas; the assertions across fields go
        # unenforced.
        self.check_arguments(
            {
                "type": "object",
                "patternProperties": {"^x_": {"type": "integer"}},
                "additionalProperties": False,
                "minProperties": 1,
                "maxProperties": 2,
            },
            {"x_a": 3},
        )
        # Strict closing leaves no room for the property dependentRequired
        # names.
        policy, xml = self.check_arguments(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "dependentRequired": {"a": ["b"]},
                "propertyNames": {"pattern": "^[ab]$"},
            },
            {"a": "123"},
        )
        extra = "<parameter=b>\n7\n</parameter>\n</function>"
        self.assertFalse(self.accepts(policy, xml.replace("</function>", extra)))
        self.check_arguments(
            {"type": "object", "not": {"required": ["forbidden"]}},
            {"allowed": 123},
        )
        self.check_arguments({"enum": [{"a": "123"}, {"b": 3}]}, {"a": "123"})

    def test_additional_typed_strings_and_local_anchors(self):
        self.check_arguments(
            {"type": "object", "additionalProperties": {"type": "string"}},
            {"extra": "123"},
        )
        self.check_arguments(
            {
                "$defs": {"text": {"$anchor": "text", "type": "string"}},
                "properties": {"value": {"$ref": "#text"}},
                "required": ["value"],
            },
            {"value": "123"},
        )
        self.check_arguments(
            {
                "$defs": {"a/b": {"type": "string"}},
                "properties": {"value": {"$ref": "#/$defs/a~1b"}},
                "required": ["value"],
            },
            {"value": "123"},
        )

    def test_forbidden_optional_fields_do_not_make_the_object_impossible(self):
        self.check_arguments(
            {
                "type": "object",
                "properties": {"disabled": False},
                "additionalProperties": False,
            },
            {},
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
        )

    def test_generic_name_rule_cannot_reencode_declared_strings(self):
        policy, _ = self.check_arguments(
            {
                "properties": {"a": {"type": "string", "const": "hello"}},
                "additionalProperties": True,
            },
            {"a": "hello", "ab": 3},
        )
        bad = (
            '<tool_call>\n<function=t0>\n<parameter=a>\n"hello"\n'
            "</parameter>\n</function>\n</tool_call>"
        )
        self.assertFalse(self.accepts(policy, bad))

    def test_extra_names_may_start_like_unused_declared_names(self):
        schema = {
            "properties": {"url": {"type": "string"}, "ab": {"type": "integer"}},
            "additionalProperties": {"type": "integer"},
        }
        for extra in ("user-agent", "urls", "u", "a", "abc"):
            with self.subTest(extra=extra):
                self.check_arguments(schema, {extra: 1})
        # Declared names still appear once, and forbidden ones not at all.
        policy, xml = self.check_arguments(
            schema,
            {"url": "x", "ab": 1, "user-agent": 2},
        )
        repeated = "<parameter=url>\nx\n</parameter>\n</function>"
        self.assertFalse(self.accepts(policy, xml.replace("</function>", repeated)))
        policy, xml = self.check_arguments(
            {"properties": {"secret": False}, "additionalProperties": {}},
            {"secrets": 1},
        )
        self.assertFalse(self.accepts(policy, xml.replace("secrets", "secret")))

    def test_tool_grammar_preserves_keyword_named_properties_and_literals(self):
        literal = {"pattern": "(?=literal)", "propertyNames": "literal"}
        references = [{"$ref": "#/x"}, {"$ref": "https://example.com/x.json"}]
        schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "propertyNames": {"const": literal},
                "nested": {"enum": [literal]},
                "constant": {"const": references[0]},
                "choice": {"enum": references},
                "defaulted": {"type": "object", "default": references[1]},
                "shown": {"type": "object", "examples": references},
                "linked": {"type": "array", "items": {"$ref": "#/$defs/x"}},
            },
            "$defs": {"x": {"type": "integer"}},
            "required": ["pattern", "propertyNames", "nested", "constant", "choice"],
        }
        projected = _grammar_compatible_schema(schema)
        self.assertEqual(
            projected.pop("x-guidance"),
            {"lenient": True, "whitespace_pattern": tool_schema.WHITESPACE},
        )
        self.assertEqual(projected, schema)
        grammar = argument_grammar(schema)
        self.assertFalse(LLMatcher.validate_grammar(grammar))
        # Literal values are written as their JSON text, annotations kept as
        # data, and references rebased only where they are schemas.
        for value in (literal, *references):
            text = json.dumps(value, separators=(",", ":"))
            self.assertIn(json.dumps(text), grammar)
        self.assertIn('"default":{"$ref":"https://example.com/x.json"}', grammar)
        self.assertIn('"items":{"$ref":"#/$defs/__splash_root/$defs/x"}', grammar)
        self.assertNotIn("__splash_root/x", grammar)

    def test_tool_parser_resolves_root_and_chained_string_refs(self):
        schema = {
            "type": "object",
            "properties": {
                "base": {"type": "string"},
                "direct": {"$ref": "#/properties/base"},
                "chained": {"$ref": "#/$defs/alias"},
            },
            "$defs": {
                "alias": {"$ref": "#/$defs/text"},
                "text": {"type": "string"},
            },
        }
        text = (
            "<tool_call>\n<function=echo>\n"
            "<parameter=direct>\n123\n</parameter>\n"
            "<parameter=chained>\n456\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls, _ = project(text, tool_policy({"echo": schema}))
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"direct": "123", "chained": "456"},
        )
        grammar = argument_grammar(schema)
        self.assertIn('[suffix="\\n</parameter>\\n"]', grammar)
        self.assertEqual(grammar.count("/(?s:.*)/"), 3)
        self.assertNotIn(r'[^"\s]', grammar)
        self.assertNotIn("RAW_MIDDLE", grammar)

    def test_tool_constraint_accepts_nullable_non_strings(self):
        for keyword, value_type in (("anyOf", "integer"), ("oneOf", "object")):
            schema = {
                "type": "object",
                "properties": {
                    "value": {keyword: [{"type": value_type}, {"type": "null"}]}
                },
            }
            grammar = argument_grammar(schema)
            self.assertIn("%json", grammar)
            self.assertNotIn("RAW_START", grammar)

    def test_cyclic_tool_alternatives_compile_without_recursing(self):
        runtime = FakeRuntime()
        factory = FakeConstraintFactory()
        harness = self.harness(runtime, constraint_factory=factory)
        for keyword in ("anyOf", "oneOf"):
            schema = {
                "$defs": {
                    "node": {keyword: [{"$ref": "#/$defs/node"}, {"type": "string"}]}
                },
                "properties": {"value": {"$ref": "#/$defs/node"}},
            }
            function = {"name": "test", "parameters": schema, "strict": True}
            with self.subTest(keyword=keyword):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/chat/completions",
                    chat_body(
                        tools=[{"type": "function", "function": function}],
                        tool_choice="required",
                        reasoning_effort="none",
                    ),
                )
                self.assertEqual(status, 200, payload)
                # A value that may be a string is raw text.
                self.assertIn("/(?s:.*)/", factory.grammars[-1])
        self.assertEqual(len(runtime.requests), 2)


if __name__ == "__main__":
    unittest.main()
