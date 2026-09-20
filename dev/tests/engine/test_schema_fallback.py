import copy
import json
import unittest
from itertools import product
from unittest import mock

from llguidance import LLMatcher

from dev.tests.engine import test_structured_tools as helpers
from dev.tests.test_server import (
    FakeConstraintFactory,
    FakeRuntime,
    FakeTokenizer,
    Harness,
    Plan,
    _byte_backend,
)
from server import output as model_output
from server import server as api
from server import tool_schema
from server.tool_schema import raw_string_schema


class SchemaFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers.StructuredToolGrammarTest.setUpClass()
        cls.guidance = helpers.StructuredToolGrammarTest.guidance
        cls.tokenizer = helpers.StructuredToolGrammarTest.tokenizer

    def policy(self, schema):
        return tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "test", "parameters": schema}}],
            "required",
            False,
        )[1]

    @staticmethod
    def call(value):
        return (
            "<tool_call>\n<function=test>\n<parameter=value>\n"
            + value
            + "\n</parameter>\n</function>\n</tool_call>"
        )

    def verify(self, schema, accepted, rejected, expected):
        original = copy.deepcopy(schema)
        policy = self.policy(schema)
        with mock.patch.object(
            tool_schema, "THINK_END_TOKEN_ID", self.tokenizer.token_to_id("</think>")
        ):
            grammar = tool_schema.tool_grammar(policy, False)
        self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
        for value, valid in [(accepted, True), *((v, False) for v in rejected)]:
            with self.subTest(value=value):
                matcher = LLMatcher(self.guidance, grammar)
                tokens = self.tokenizer.encode(self.call(value)).ids
                length = matcher.validate_tokens(tokens)
                complete = False
                if length == len(tokens):
                    self.assertTrue(matcher.consume_tokens(tokens))
                    complete = matcher.is_accepting()
                self.assertEqual(complete, valid)
        projector = model_output.StreamingToolCallProjector(policy, "owned")
        deltas = []
        for character in self.call(accepted):
            deltas.extend(projector.put(character))
        arguments = "".join(
            value.get("function", {}).get("arguments", "")
            for kind, value in deltas
            if kind == "tool"
        )
        self.assertEqual(json.loads(arguments), {"value": expected})
        self.assertIs(type(json.loads(arguments)["value"]), type(expected))
        self.assertEqual(schema, original)
        self.assertEqual(policy.schemas["test"], original)
        # Generation grammar is not the final authority: retain the exact
        # original schema for checking completed calls.
        with self.assertRaises(api.APIError):
            model_output.validate_tool_calls(
                [
                    {
                        "id": "bad",
                        "type": "function",
                        "function": {
                            "name": "test",
                            "arguments": json.dumps({"value": 7}),
                        },
                    }
                ],
                policy,
            )

    def test_string_union_with_outer_assertions_uses_json_encoding(self):
        for kind in ("anyOf", "oneOf"):
            schema = {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        kind: [{"const": "red"}, {"const": "blue"}],
                        "const": "red",
                    }
                },
                "required": ["value"],
            }
            with self.subTest(kind=kind):
                self.assertIsNone(
                    raw_string_schema(schema["properties"]["value"], schema)
                )
                self.verify(schema, '"red"', ['"blue"', "red", "7"], "red")

    def test_local_reference_and_sibling_constraints_are_both_retained(self):
        schema = {
            "type": "object",
            "$defs": {"Choice": {"type": "string", "enum": ["1", "2"]}},
            "properties": {"value": {"$ref": "#/$defs/Choice", "const": "1"}},
            "required": ["value"],
        }
        self.assertIsNone(raw_string_schema(schema["properties"]["value"], schema))
        self.verify(schema, '"1"', ['"2"', "1", "7"], "1")
        plain = {
            "type": "object",
            "properties": {"value": {"type": "string", "const": "1"}},
            "required": ["value"],
        }
        self.verify(plain, "1", ['"1"', "2"], "1")

    def test_reference_with_array_constraint_keeps_json_type(self):
        schema = {
            "type": "object",
            "$defs": {"Names": {"type": "array", "items": {"type": "string"}}},
            "properties": {"value": {"$ref": "#/$defs/Names", "maxItems": 1}},
            "required": ["value"],
        }
        self.verify(schema, '["red"]', ['["red","blue"]', "[7]", '"red"'], ["red"])

    def test_unsupported_generation_constraints_remain_enforced_on_output(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                }
            },
            "required": ["value"],
        }
        original = copy.deepcopy(schema)
        policy = self.policy(schema)
        _, validator = tool_schema.normalize_response_format(
            {"type": "json_schema", "json_schema": {"schema": schema}}
        )
        optional = tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "test", "parameters": schema}}],
            "auto",
            False,
        )[1]
        for grammar, text in (
            (tool_schema.tool_grammar(policy, False), self.call('["red","red"]')),
            (tool_schema.json_grammar(schema, False), '{"value":["red","red"]}'),
            (
                tool_schema.tool_grammar(optional, False, schema),
                '{"value":["red","red"]}',
            ),
        ):
            with self.subTest(grammar=grammar):
                self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
                matcher = LLMatcher(self.guidance, grammar)
                self.assertTrue(matcher.consume_tokens(self.tokenizer.encode(text).ids))
                self.assertTrue(matcher.is_accepting())
        model_output.validate_response_content('{"value":["red","blue"]}', validator)
        with self.assertRaises(api.APIError) as caught:
            model_output.validate_response_content('{"value":["red","red"]}', validator)
        self.assertEqual(caught.exception.code, "invalid_model_output")
        projector = model_output.StreamingToolCallProjector(policy, "owned")
        deltas = []
        with self.assertRaises(api.APIError) as caught:
            for character in self.call('["red","red"]'):
                deltas.extend(projector.put(character))
        self.assertEqual(caught.exception.code, "invalid_model_output")
        arguments = "".join(
            value.get("function", {}).get("arguments", "")
            for kind, value in deltas
            if kind == "tool"
        )
        with self.assertRaises(json.JSONDecodeError):
            json.loads(arguments)
        self.assertEqual(schema, original)
        self.assertEqual(policy.schemas["test"], original)

    def test_deferred_assertions_across_protocols_and_streaming(self):
        guidance = self.guidance

        class CompilingFactory(FakeConstraintFactory):
            def create(self, grammar):
                error = LLMatcher.validate_grammar(grammar, guidance)
                if error:
                    raise AssertionError(error)
                return super().create(grammar)

        for path, tool, stream in product(
            ("/v1/chat/completions", "/v1/responses", "/v1/messages"),
            (False, True),
            (False, True),
        ):
            with self.subTest(path=path, tool=tool, stream=stream):
                field = "city" if tool else "x"
                schema = {
                    "type": "object",
                    "properties": {
                        field: {
                            "type": "array",
                            "items": {"type": "string"},
                            "uniqueItems": True,
                        }
                    },
                    "required": [field],
                }
                tokenizer = FakeTokenizer()
                tokenizer.fragments[5] = tokenizer.fragments[5].replace(
                    "Paris", '["Paris","Paris"]'
                )
                tokenizer.fragments[10] = '{"x":["Paris","Paris"]}'
                tokenizer.backend_tokenizer = _byte_backend(tokenizer.fragments)
                function = {"name": "weather", "parameters": schema}
                body = {"model": "test-model", "stream": stream}
                if path == "/v1/responses":
                    body.update(
                        input="hello",
                        max_output_tokens=16,
                        reasoning={"effort": "none"},
                    )
                    if tool:
                        body["tools"] = [{"type": "function", **function}]
                    else:
                        body["text"] = {
                            "format": {
                                "type": "json_schema",
                                "name": "answer",
                                "schema": schema,
                            }
                        }
                else:
                    body.update(
                        messages=[{"role": "user", "content": "hello"}], max_tokens=16
                    )
                    if path == "/v1/messages":
                        body["thinking"] = {"type": "disabled"}
                        if tool:
                            body["tools"] = [
                                {"name": "weather", "input_schema": schema}
                            ]
                        else:
                            body["output_config"] = {
                                "format": {"type": "json_schema", "schema": schema}
                            }
                    else:
                        body["reasoning_effort"] = "none"
                        if tool:
                            body["tools"] = [{"type": "function", "function": function}]
                        else:
                            body["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {"schema": schema},
                            }
                harness = Harness(
                    FakeRuntime(Plan([[5 if tool else 10]])),
                    constraint_factory=CompilingFactory(),
                    tokenizer=tokenizer,
                )
                try:
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200 if stream else 500, payload)
                    self.assertIn(b"invalid", payload)
                    self.assertNotIn(b'"type":"message_stop"', payload)
                    self.assertNotIn(b'"type":"response.completed"', payload)
                    self.assertNotIn(b'"finish_reason":"tool_calls"', payload)
                finally:
                    harness.close()

    def test_top_level_tool_layout_constraints_compile_and_preserve_validation(self):
        constraints = {
            "$ref": "#/$defs/Value",
            "$dynamicRef": "#/$defs/Value",
            "allOf": [{}],
            "anyOf": [{}],
            "oneOf": [{}],
            "not": {},
            "if": {},
            "then": {},
            "else": {},
            "dependentRequired": {"value": ["other"]},
            "dependentSchemas": {"value": {}},
            "enum": [{"value": "ok"}],
            "const": {"value": "ok"},
            "minProperties": 1,
            "maxProperties": 1,
            "patternProperties": {"^value$": {"type": "string"}},
        }
        for keyword, value in constraints.items():
            schema = {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "$defs": {"Value": {"type": "object"}},
                keyword: value,
            }
            original = copy.deepcopy(schema)
            with self.subTest(keyword=keyword):
                policy = self.policy(schema)
                grammar = tool_schema.tool_grammar(policy, False)
                self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
                self.assertEqual(schema, original)
                self.assertEqual(policy.schemas["test"], original)
                from jsonschema import Draft202012Validator

                validator = Draft202012Validator(original)
                for arguments in (
                    {},
                    {"value": "ok"},
                    {"value": 7},
                    {"value": "ok", "other": True},
                ):
                    calls = [
                        {
                            "function": {
                                "name": "test",
                                "arguments": json.dumps(arguments),
                            }
                        }
                    ]
                    if validator.is_valid(arguments):
                        model_output.validate_tool_calls(calls, policy)
                    else:
                        with self.assertRaises(api.APIError):
                            model_output.validate_tool_calls(calls, policy)

    def test_missing_and_remote_references_remain_explicit_errors(self):
        for reference in ("#/$defs/missing", "https://example.com/schema.json"):
            schema = {
                "type": "object",
                "properties": {"value": {"$ref": reference, "type": "string"}},
            }
            with (
                self.subTest(reference=reference),
                self.assertRaises(api.APIError) as caught,
            ):
                tool_schema.tool_grammar(self.policy(schema), False)
            self.assertEqual(caught.exception.status, 400)
        with self.assertRaises(api.APIError) as caught:
            tool_schema.tool_grammar(self.policy({"$ref": "#/$defs/missing"}), False)
        self.assertEqual(caught.exception.status, 400)
