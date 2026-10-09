import copy
import json
from itertools import product
from unittest import mock

from llguidance import LLMatcher

from dev.tests.server_fixtures import (
    FakeConstraintFactory,
    FakeRuntime,
    FakeTokenizer,
    Harness,
    HarnessTestCase,
    Plan,
    byte_backend,
    chat_body,
    grammar_tokenizers,
    rich_weather_tool,
)
from dev.tests.tool_output import argument_grammar, project, tool_policy
from server import output as model_output
from server import server as api
from server import tool_schema


class SchemaFallbackTests(HarnessTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.guidance = grammar_tokenizers()

    def policy(self, schema):
        """The policy of a required call of a strict tool, whose arguments
        its grammar frames."""
        function = {"name": "test", "parameters": schema, "strict": True}
        return tool_schema.normalize_tools(
            [{"type": "function", "function": function}], "required", False
        )[1]

    @staticmethod
    def json_schemas(grammar):
        """The schemas `grammar` compiles as JSON values."""
        try:
            sources = [part["lark_grammar"] for part in json.loads(grammar)["grammars"]]
        except ValueError:
            sources = [grammar]
        decoder = json.JSONDecoder()
        for source in sources:
            for _, _, rest in (line.partition("%json ") for line in source.split("\n")):
                if rest:
                    yield decoder.raw_decode(rest)[0]

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

    def test_enumerated_strings_are_written_as_their_text(self):
        # An enumerated value is written in Qwen XML as its text, whatever
        # other assertions stand beside it.
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
                self.verify(schema, "red", ['"red"', "blue", "7"], "red")

    def test_local_reference_and_sibling_constraints_are_both_retained(self):
        schema = {
            "type": "object",
            "$defs": {"Choice": {"type": "string", "enum": ["1", "2"]}},
            "properties": {"value": {"$ref": "#/$defs/Choice", "const": "1"}},
            "required": ["value"],
        }
        self.verify(schema, "1", ['"1"', "2", "7"], "1")
        plain = {
            "type": "object",
            "properties": {"value": {"type": "string", "const": "1"}},
            "required": ["value"],
        }
        self.verify(plain, "1", ['"1"', "2"], "1")

    def test_compilable_patterns_constrain_generation(self):
        order = {"type": "string", "pattern": "^ORD-[0-9]{4}$"}
        nested = {"type": "object", "properties": {"id": order}, "required": ["id"]}
        self.verify(
            {"type": "object", "properties": {"value": nested}, "required": ["value"]},
            '{"id":"ORD-1234"}',
            ['{"id":"ORD-2024-8892"}', '{"id":"ord-1234"}'],
            {"id": "ORD-1234"},
        )
        # An unanchored pattern keeps its search semantics; look-around cannot
        # compile and a response format checks it on the complete output.
        schema = {
            "type": "object",
            "properties": {
                "order": order,
                "code": {"type": "string", "pattern": "[A-Z]{3}"},
                "path": {"type": "string", "pattern": r"^(?!\.\.)[a-z/.]+$"},
            },
            "required": ["order", "code", "path"],
            "additionalProperties": False,
        }
        grammar = tool_schema.json_grammar(schema, False)
        self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
        for order_id, code, valid in (
            ("ORD-1234", "xABCx", True),
            ("ORD-2024-8892-GOLD", "ABC", False),
            ("ORD-1234", "abc", False),
        ):
            with self.subTest(order=order_id, code=code):
                text = json.dumps({"order": order_id, "code": code, "path": "../x"})
                tokens = self.tokenizer.encode(text).ids
                matcher = LLMatcher(self.guidance, grammar)
                accepted = matcher.validate_tokens(tokens) == len(tokens)
                if accepted:
                    self.assertTrue(matcher.consume_tokens(tokens))
                    accepted = matcher.is_accepting()
                self.assertEqual(accepted, valid)
        _, validator = tool_schema.normalize_response_format(
            {"type": "json_schema", "json_schema": {"schema": schema}}
        )
        with self.assertRaises(api.APIError) as caught:
            model_output.validate_response_content(
                json.dumps({"order": "ORD-1234", "code": "ABC", "path": "../x"}),
                validator,
            )
        self.assertEqual(caught.exception.code, "invalid_model_output")

    def test_reference_with_array_constraint_keeps_json_type(self):
        schema = {
            "type": "object",
            "$defs": {"Names": {"type": "array", "items": {"type": "string"}}},
            "properties": {"value": {"$ref": "#/$defs/Names", "maxItems": 1}},
            "required": ["value"],
        }
        self.verify(schema, '["red"]', ['["red","blue"]', "[7]", '"red"'], ["red"])

    def test_unsupported_generation_constraints_remain_enforced_on_answers(self):
        # A grammar leaves out what it cannot compile. A response format
        # checks it on the complete answer; a strict tool's arguments go
        # without it.
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
        for character in self.call('["red","red"]'):
            projector.put(character)
        _, calls, _ = projector.finish(False)
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments, {"value": ["red", "red"]})
        self.assertEqual(schema, original)
        self.assertEqual(policy.schemas["test"], original)

    def test_huge_bounds_never_reach_the_grammar_compiler(self):
        # Compiling a bound costs memory in proportion to its value. A local
        # reference can make any object a schema, even literal data.
        huge = 10**9
        for keyword, kind in (
            ("minItems", "array"),
            ("maxItems", "array"),
            ("multipleOf", "integer"),
        ):
            bounded = {"type": kind, keyword: huge}
            schema = {
                "type": "object",
                "properties": {
                    "value": bounded,
                    "linked": {"$ref": "#/x-stash"},
                    "literal": {"$ref": "#/properties/fixed/const"},
                    "fixed": {"const": bounded},
                },
                "x-stash": bounded,
            }
            policy = self.policy(schema)
            for grammar in (
                tool_schema.tool_grammar(policy, False),
                tool_schema.json_grammar(schema, False),
                tool_schema.tool_grammar(policy, False, schema),
            ):
                with self.subTest(keyword=keyword, grammar=grammar[:60]):
                    for compiled in self.json_schemas(grammar):
                        self.assertNotIn(str(huge), json.dumps(compiled))
                    self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))

    def test_bounds_above_the_grammar_ceiling_are_left_out(self):
        limit = tool_schema.MAX_GRAMMAR_BOUND
        violations = (
            (
                "maxItems",
                {"type": "array", "items": {}},
                lambda bound: [0] * (bound + 1),
            ),
            ("minItems", {"type": "array", "items": {}}, lambda bound: [0]),
            ("multipleOf", {"type": "integer"}, lambda bound: 1),
        )
        for (keyword, kind, violation), bound in product(
            violations, (limit, limit + 1)
        ):
            value = violation(bound)
            schema = {
                "type": "object",
                "properties": {"value": {**kind, keyword: bound}},
                "required": ["value"],
            }
            with self.subTest(keyword=keyword, bound=bound):
                policy = self.policy(schema)
                matcher = LLMatcher(
                    self.guidance, tool_schema.tool_grammar(policy, False)
                )
                text = self.call(json.dumps(value, separators=(",", ":")))
                tokens = self.tokenizer.encode(text).ids
                accepted = (
                    matcher.validate_tokens(tokens) == len(tokens)
                    and matcher.consume_tokens(tokens)
                    and matcher.is_accepting()
                )
                self.assertEqual(accepted, bound > limit)

    def test_deferred_assertions_check_answers_but_not_calls(self):
        # A response format checks what its grammar left out on the complete
        # answer; a tool's arguments go unchecked.
        guidance = self.guidance

        class CompilingFactory(FakeConstraintFactory):
            def create(self, grammar, *, timeout=None, prefixes=None):
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
                tokenizer.backend_tokenizer = byte_backend(tokenizer.fragments)
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
                    if tool:
                        self.assertEqual(status, 200, payload)
                        self.assertNotIn(b"invalid", payload)
                        self.assertIn(b"Paris", payload)
                        continue
                    self.assertEqual(status, 200 if stream else 500, payload)
                    self.assertIn(b"invalid", payload)
                    self.assertNotIn(b'"type":"message_stop"', payload)
                    self.assertNotIn(b'"type":"response.completed"', payload)
                    self.assertNotIn(b'"finish_reason":"tool_calls"', payload)
                finally:
                    harness.close()

    def test_top_level_tool_layout_constraints_compile(self):
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

    def test_remote_references_in_draft3_schema_keywords_are_request_errors(self):
        # Draft 3 nests schemas under extends and in type and disallow lists.
        # A remote reference there was accepted, and every output then failed
        # validation with an internal error after generation.
        remote = {"$ref": "https://example.com/schema.json"}
        for keywords in (
            {"extends": remote},
            {"extends": [remote]},
            {"disallow": [remote]},
            {"properties": {"value": {"type": ["string", remote]}}},
        ):
            schema = {
                "$schema": "http://json-schema.org/draft-03/schema#",
                "type": "object",
                **keywords,
            }
            tools = [
                {"type": "function", "function": {"name": "t", "parameters": schema}}
            ]
            response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
            for normalize in (
                lambda: tool_schema.normalize_tools(tools, None, None),
                lambda: tool_schema.normalize_response_format(response_format),
            ):
                with self.subTest(keywords=keywords):
                    with self.assertRaises(api.APIError) as caught:
                        normalize()
                    self.assertEqual(caught.exception.status, 400)
                    self.assertIn("remote", caught.exception.message)
        schema = {
            "$schema": "http://json-schema.org/draft-03/schema#",
            "definitions": {"text": {"type": "string"}},
            "properties": {"value": {"extends": {"$ref": "#/definitions/text"}}},
        }
        _, validator = tool_schema.normalize_response_format(
            {"type": "json_schema", "json_schema": {"schema": schema}}
        )
        self.assertTrue(validator.is_valid({"value": "x"}))
        self.assertFalse(validator.is_valid({"value": 1}))

    def test_schemas_validation_cannot_evaluate_are_request_errors(self):
        # Draft 4 leaves $ref unchecked and draft 3 accepts any type name;
        # validating an output against either failed with an internal error.
        # A tool's calls are not validated, so only a reference that cannot
        # be read refuses its schema.
        for draft, value in (
            ("draft-04", {"$ref": None}),
            ("draft-04", {"$ref": 5}),
            ("draft-03", {"type": "x"}),
            ("draft-03", {"type": ["string", "x"]}),
            ("draft-03", {"disallow": "x"}),
        ):
            schema = {
                "$schema": f"http://json-schema.org/{draft}/schema#",
                "type": "object",
                "properties": {"value": value},
            }
            tools = [
                {"type": "function", "function": {"name": "t", "parameters": schema}}
            ]
            response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
            normalizers = [
                lambda: tool_schema.normalize_response_format(response_format)
            ]
            if "$ref" in value:
                normalizers.append(
                    lambda: tool_schema.normalize_tools(tools, None, None)
                )
            else:
                tool_schema.normalize_tools(tools, None, None)
            for normalize in normalizers:
                with (
                    self.subTest(value=value),
                    self.assertRaises(api.APIError) as caught,
                ):
                    normalize()
                self.assertEqual(caught.exception.status, 400)
        schema = {
            "$schema": "http://json-schema.org/draft-03/schema#",
            "properties": {"value": {"type": ["any", {"type": "string"}]}},
        }
        _, validator = tool_schema.normalize_response_format(
            {"type": "json_schema", "json_schema": {"schema": schema}}
        )
        self.assertTrue(validator.is_valid({"value": 1}))

    def test_unchecked_keywords_of_older_dialects_are_request_errors(self):
        # An older declared dialect leaves newer keywords unchecked, so they
        # can hold any value. That is the client's schema error, not a crash;
        # a value's keyword that names nothing leaves the value unconstrained.
        schema = {
            "$schema": "http://json-schema.org/draft-03/schema#",
            "type": "object",
            "properties": {"value": {"anyOf": 5}},
        }
        self.assertIn("/(?s:.*)/", tool_schema.tool_grammar(self.policy(schema), False))
        for draft, keywords in (
            ("draft-07", {"dependentSchemas": 5}),
            ("draft-03", {"allOf": 5}),
            ("draft-03", {"required": True}),
            ("draft-04", {"$ref": {"a": 1}}),
        ):
            schema = {
                "$schema": f"http://json-schema.org/{draft}/schema#",
                "type": "object",
                **keywords,
            }
            with self.subTest(schema=schema), self.assertRaises(api.APIError) as caught:
                tool_schema.tool_grammar(self.policy(schema), False)
            self.assertEqual(caught.exception.status, 400)

    def test_tool_grammar_leaves_out_property_name_assertions(self):
        schema = {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "object",
                    "propertyNames": {"pattern": "^[a-z]+$"},
                    "additionalProperties": {"type": "string"},
                }
            },
        }
        grammar = argument_grammar(schema)
        self.assertNotIn("propertyNames", grammar)
        self.assertFalse(LLMatcher.validate_grammar(grammar))

    def test_tool_grammar_leaves_out_nested_lookaround_patterns(self):
        pattern = (
            r"^(?!\.\.?(?:/|$))[A-Za-z0-9_\-.~:@+]{1,200}"
            r"(?:/(?!\.\.?(?:/|$))[A-Za-z0-9_\-.~:@+]{1,200}){0,14}$"
        )
        schema = {
            "type": "object",
            "properties": {
                "writes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"path": {"$ref": "#/$defs/path"}},
                        "required": ["path"],
                    },
                }
            },
            "$defs": {"path": {"type": "string", "pattern": pattern}},
            "required": ["writes"],
        }
        original = json.dumps(schema)
        grammar = argument_grammar(schema)
        self.assertFalse(LLMatcher.validate_grammar(grammar))
        self.assertNotIn("?!", grammar)
        self.assertEqual(json.dumps(schema), original)
        policy = tool_policy({"write": schema})
        for path in ("website/index.html", "../secret"):
            with self.subTest(path=path):
                _, calls, _ = project(
                    "<tool_call>\n<function=write>\n<parameter=writes>\n"
                    + json.dumps([{"path": path}])
                    + "\n</parameter>\n</function>\n</tool_call>",
                    policy,
                )
                self.assertEqual(
                    json.loads(calls[0]["function"]["arguments"]),
                    {"writes": [{"path": path}]},
                )

    def test_remote_tool_schema_ref_is_rejected_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        remote = {"$ref": "https://example.com/city.json"}
        tool = rich_weather_tool()
        tool["function"]["parameters"]["$defs"]["city"] = remote
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "city",
                "schema": {"type": "object", "properties": {"city": remote}},
            },
        }
        for parameter, value in (
            ("tools", [tool]),
            ("response_format", response_format),
        ):
            with self.subTest(parameter=parameter):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(**{parameter: value})
                )
                self.assertEqual(status, 400)
                self.assertIn(
                    "schema reference is not allowed: https://example.com/city.json",
                    json.loads(payload)["error"]["message"],
                )
        self.assertEqual(runtime.requests, [])
