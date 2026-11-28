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
from server.chat_format import QWEN, QWEN_XML


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
        grammar = tool_schema.tool_grammar(policy, False, chat_format=QWEN)
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
        projector = model_output.StreamingToolCallProjector(
            policy, "owned", syntax=QWEN_XML
        )
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
        # compile, so a grammar leaves it out and a response format holding
        # it is refused.
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
        grammar = tool_schema.json_grammar(schema, False, chat_format=QWEN)
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
        with self.assertRaises(api.APIError) as caught:
            tool_schema.normalize_response_format(
                {
                    "type": "json_schema",
                    "json_schema": {"schema": schema, "strict": True},
                }
            )
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("pattern at #/properties/path", caught.exception.message)

    def test_every_keyword_the_grammar_drops_refuses_a_strict_response_format(self):
        # Each is refused, under strict, naming the keyword and where it is; a
        # value that asks nothing, and a keyword the schema's dialect does
        # not assert, are accepted, as are the keywords the grammar enforces.
        draft7 = "http://json-schema.org/draft-07/schema#"
        draft3 = "http://json-schema.org/draft-03/schema#"
        refused = (
            ("not", {"not": {"type": "string"}}, None),
            ("if", {"if": {"type": "string"}, "then": {"maxLength": 2}}, None),
            (
                "dependentSchemas",
                {"dependentSchemas": {"a": {"required": ["b"]}}},
                None,
            ),
            ("dependentRequired", {"dependentRequired": {"a": ["b"]}}, None),
            ("dependencies", {"dependencies": {"a": ["b"]}}, draft7),
            ("propertyNames", {"propertyNames": {"pattern": "^a"}}, None),
            ("unevaluatedProperties", {"unevaluatedProperties": False}, None),
            ("unevaluatedItems", {"unevaluatedItems": False}, None),
            ("contains", {"contains": {"type": "string"}, "minContains": 2}, None),
            ("uniqueItems", {"uniqueItems": True}, None),
            ("$dynamicRef", {"$dynamicRef": "#"}, None),
            ("extends", {"extends": {"type": "string"}}, draft3),
            ("disallow", {"disallow": "string"}, draft3),
            ("divisibleBy", {"divisibleBy": 2}, draft3),
        )
        for keyword, value, dialect in refused:
            schema = {"type": "object", "properties": {"a": value}}
            if dialect:
                schema["$schema"] = dialect
            with self.subTest(keyword=keyword):
                with self.assertRaises(api.APIError) as caught:
                    tool_schema.normalize_response_format(
                        {
                            "type": "json_schema",
                            "json_schema": {"schema": schema, "strict": True},
                        }
                    )
                self.assertEqual(
                    (caught.exception.status, caught.exception.message),
                    (
                        400,
                        f"unsupported output schema: {keyword} at #/properties/a "
                        "is not enforced by the output grammar",
                    ),
                )
                # Where llguidance names what it drops, it names this one; the
                # server drops propertyNames before it, and llguidance ignores
                # draft 3's keywords without a word.
                warnings = LLMatcher.validate_grammar_with_warnings(
                    tool_schema.json_grammar(schema, False, chat_format=QWEN)
                )[1]
                if dialect != draft3 and keyword != "propertyNames":
                    self.assertIn(keyword, " ".join(warnings))
        for keyword, value in (
            ("maxItems", {"type": "array", "maxItems": 257}),
            ("minItems", {"type": "array", "minItems": 1000}),
            ("multipleOf", {"type": "integer", "multipleOf": 1000}),
            ("oneOf", {"oneOf": [{"type": "string"}, {"type": "integer"}]}),
        ):
            schema = {"type": "object", "properties": {"a": value}}
            with (
                self.subTest(keyword=keyword),
                self.assertRaises(api.APIError) as caught,
            ):
                tool_schema.normalize_response_format(
                    {
                        "type": "json_schema",
                        "json_schema": {"schema": schema, "strict": True},
                    }
                )
            self.assertIn(f"{keyword} ", caught.exception.message)
            self.assertIn("at #/properties/a ", caught.exception.message)
        for value in (
            {"uniqueItems": False},
            {"propertyNames": {}},
            {"unevaluatedProperties": True},
            {"if": {"type": "string"}},
            {"dependencies": {"a": ["b"]}},
            {"minProperties": 1, "maxProperties": 3},
            {"type": "array", "maxItems": 256, "minItems": 100},
            {"type": "number", "multipleOf": 0.5, "exclusiveMinimum": 0},
            {"type": "string", "pattern": "^[a-z]+$", "format": "date-time"},
            {"properties": {"not": {"type": "string"}}},
        ):
            schema = {"type": "object", "properties": {"a": value}}
            with self.subTest(accepted=value):
                _, validator = tool_schema.normalize_response_format(
                    {
                        "type": "json_schema",
                        "json_schema": {"schema": schema, "strict": True},
                    }
                )
                self.assertFalse(hasattr(validator, "unenforced"))

    def test_a_schema_not_asked_strict_is_best_effort(self):
        # OpenAI's default, strict false: the grammar enforces what it can,
        # and an answer that breaks only a keyword it did not enforce is
        # returned; one that breaks an enforced keyword is still invalid.
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "maxItems": 1000,
                },
                "pick": {"oneOf": [{"type": "integer"}, {"type": "number"}]},
                "code": {"type": "string", "pattern": "^(?!x)[a-z]+$"},
                "n": {"type": "integer", "minimum": 0},
            },
            "required": ["tags", "n"],
            "not": {"required": ["forbidden"]},
        }
        for strict in (None, False):
            wrapper = {"schema": schema} | (
                {} if strict is None else {"strict": strict}
            )
            _, validator = tool_schema.normalize_response_format(
                {"type": "json_schema", "json_schema": wrapper}
            )
            with self.subTest(strict=strict):
                self.assertEqual(
                    sorted(keyword for keyword, _ in validator.unenforced),
                    ["maxItems", "not", "oneOf", "pattern", "uniqueItems"],
                )
                model_output.validate_response_content(
                    json.dumps(
                        {
                            "tags": ["a", "a"],
                            "ids": list(range(1001)),
                            "pick": 1,
                            "code": "xy",
                            "n": 1,
                            "forbidden": True,
                        }
                    ),
                    validator,
                )
                with self.assertRaises(api.APIError) as caught:
                    model_output.validate_response_content(
                        json.dumps({"tags": ["a"], "n": -1}), validator
                    )
                self.assertEqual(caught.exception.code, "invalid_model_output")
        # The schema itself is left as sent: its grammar is built from it.
        self.assertIn("uniqueItems", json.dumps(schema))

    def test_reference_with_array_constraint_keeps_json_type(self):
        schema = {
            "type": "object",
            "$defs": {"Names": {"type": "array", "items": {"type": "string"}}},
            "properties": {"value": {"$ref": "#/$defs/Names", "maxItems": 1}},
            "required": ["value"],
        }
        self.verify(schema, '["red"]', ['["red","blue"]', "[7]", '"red"'], ["red"])

    def test_unsupported_generation_constraints_are_refused_for_answers(self):
        # A grammar leaves out what it cannot compile. A response format
        # holding it is refused before generation, which would otherwise fail
        # after the whole answer; a strict tool's arguments go without it.
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
        with self.assertRaises(api.APIError) as caught:
            tool_schema.normalize_response_format(
                {
                    "type": "json_schema",
                    "json_schema": {"schema": schema, "strict": True},
                }
            )
        self.assertEqual(
            (caught.exception.status, caught.exception.message),
            (
                400,
                "unsupported output schema: uniqueItems at #/properties/value "
                "is not enforced by the output grammar",
            ),
        )
        optional = tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "test", "parameters": schema}}],
            "auto",
            False,
        )[1]
        for grammar, text in (
            (
                tool_schema.tool_grammar(policy, False, chat_format=QWEN),
                self.call('["red","red"]'),
            ),
            (
                tool_schema.json_grammar(schema, False, chat_format=QWEN),
                '{"value":["red","red"]}',
            ),
            (
                tool_schema.tool_grammar(optional, False, schema, chat_format=QWEN),
                '{"value":["red","red"]}',
            ),
        ):
            with self.subTest(grammar=grammar):
                self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
                matcher = LLMatcher(self.guidance, grammar)
                self.assertTrue(matcher.consume_tokens(self.tokenizer.encode(text).ids))
                self.assertTrue(matcher.is_accepting())
        projector = model_output.StreamingToolCallProjector(
            policy, "owned", syntax=QWEN_XML
        )
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
                tool_schema.tool_grammar(policy, False, chat_format=QWEN),
                tool_schema.json_grammar(schema, False, chat_format=QWEN),
                tool_schema.tool_grammar(policy, False, schema, chat_format=QWEN),
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
                    self.guidance,
                    tool_schema.tool_grammar(policy, False, chat_format=QWEN),
                )
                text = self.call(json.dumps(value, separators=(",", ":")))
                tokens = self.tokenizer.encode(text).ids
                accepted = (
                    matcher.validate_tokens(tokens) == len(tokens)
                    and matcher.consume_tokens(tokens)
                    and matcher.is_accepting()
                )
                self.assertEqual(accepted, bound > limit)

    def test_unenforced_assertions_refuse_strict_answers_only(self):
        # A strict response format whose grammar would leave a keyword out
        # is refused before generation, in each dialect's error shape (Chat
        # and Responses when asked strict, Messages always); one not asked
        # strict is best effort, its answer returned as the model wrote it,
        # counted and logged; a tool's arguments go unchecked.
        guidance = self.guidance

        class CompilingFactory(FakeConstraintFactory):
            def create(self, grammar, *, timeout=None, prefixes=None):
                error = LLMatcher.validate_grammar(grammar, guidance)
                if error:
                    raise AssertionError(error)
                return super().create(grammar)

        for path, tool, stream, strict in product(
            ("/v1/chat/completions", "/v1/responses", "/v1/messages"),
            (False, True),
            (False, True),
            (False, True),
        ):
            if path == "/v1/messages" and not strict:
                continue
            with self.subTest(path=path, tool=tool, stream=stream, strict=strict):
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
                                "strict": strict,
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
                                "json_schema": {"schema": schema, "strict": strict},
                            }
                runtime = FakeRuntime(Plan([[5 if tool else 10]]))
                harness = Harness(
                    runtime,
                    constraint_factory=CompilingFactory(),
                    tokenizer=tokenizer,
                )
                try:
                    with mock.patch.object(api, "print_status") as printed:
                        status, _, payload = harness.request("POST", path, body)
                    if tool:
                        self.assertEqual(status, 200, payload)
                        self.assertNotIn(b"invalid", payload)
                        self.assertIn(b"Paris", payload)
                        continue
                    if not strict:
                        self.assertEqual(status, 200, payload)
                        self.assertIn(b"Paris", payload)
                        self.assertNotIn(b"invalid", payload)
                        self.assertIn(
                            "Schema keywords not enforced · request 1 · uniqueItems",
                            [args[0] for args, _ in printed.call_args_list],
                        )
                        self.assertEqual(
                            harness.app.status()["structured_output"],
                            {"keywords_not_enforced": {"uniqueItems": 1}},
                        )
                        continue
                    self.assertEqual(status, 400, payload)
                    error = json.loads(payload)["error"]
                    self.assertEqual(
                        error["message"],
                        "unsupported output schema: uniqueItems at #/properties/x "
                        "is not enforced by the output grammar",
                    )
                    self.assertEqual(error["type"], "invalid_request_error")
                    self.assertEqual(runtime.requests, [])
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
                grammar = tool_schema.tool_grammar(policy, False, chat_format=QWEN)
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
                tool_schema.tool_grammar(self.policy(schema), False, chat_format=QWEN)
            self.assertEqual(caught.exception.status, 400)
        with self.assertRaises(api.APIError) as caught:
            tool_schema.tool_grammar(
                self.policy({"$ref": "#/$defs/missing"}), False, chat_format=QWEN
            )
        self.assertEqual(caught.exception.status, 400)

    def test_remote_references_in_draft3_schema_keywords_are_request_errors(self):
        # Draft 3 nests schemas under extends and in type and disallow lists.
        # A remote reference there was accepted, and every output then failed
        # validation with an internal error after generation. A strict tool's
        # grammar needs its schema too.
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
                {
                    "type": "function",
                    "function": {"name": "t", "parameters": schema, "strict": True},
                }
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
        # A local one is read, but draft 3's extends is not enforced by the
        # output grammar, so a response format holding it is refused.
        schema = {
            "$schema": "http://json-schema.org/draft-03/schema#",
            "definitions": {"text": {"type": "string"}},
            "properties": {"value": {"extends": {"$ref": "#/definitions/text"}}},
        }
        with self.assertRaises(api.APIError) as caught:
            tool_schema.normalize_response_format(
                {
                    "type": "json_schema",
                    "json_schema": {"schema": schema, "strict": True},
                }
            )
        self.assertIn("extends at #/properties/value", caught.exception.message)

    def test_schemas_validation_cannot_evaluate_are_request_errors(self):
        # Draft 4 leaves $ref unchecked and draft 3 accepts any type name;
        # validating an output against either failed with an internal error.
        # A tool's calls are not validated, so only a reference that cannot
        # be read refuses its schema, a strict tool's, whose grammar needs it.
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
                {
                    "type": "function",
                    "function": {"name": "t", "parameters": schema, "strict": True},
                }
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

    def test_unchecked_keywords_of_older_dialects_constrain_nothing(self):
        # An older declared dialect leaves newer keywords unchecked, so they
        # can hold any value. One of another kind than the keyword takes
        # constrains nothing, as the dialect, which does not define it, reads
        # it: a value's keyword leaves the value unconstrained, and one of the
        # arguments leaves them as their other keywords frame them. A $ref,
        # which every dialect defines, must still be a string.
        schema = {
            "$schema": "http://json-schema.org/draft-03/schema#",
            "type": "object",
            "properties": {"value": {"anyOf": 5}},
        }
        self.assertIn(
            "/(?s:.*)/",
            tool_schema.tool_grammar(self.policy(schema), False, chat_format=QWEN),
        )
        for draft, keywords in (
            ("draft-07", {"dependentSchemas": 5}),
            ("draft-03", {"allOf": 5}),
            ("draft-03", {"required": True}),
        ):
            schema = {
                "$schema": f"http://json-schema.org/{draft}/schema#",
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                **keywords,
            }
            with self.subTest(schema=schema):
                self.verify(schema, "7", ["seven"], 7)
        schema = {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "object",
            "$ref": {"a": 1},
        }
        with self.assertRaises(api.APIError) as caught:
            self.policy(schema)
        self.assertEqual(
            (caught.exception.status, caught.exception.message),
            (400, "$ref must be a string"),
        )

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
        tool["function"]["strict"] = True
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
