import json
import unittest
from types import SimpleNamespace
from unittest import mock

from llguidance import LLMatcher

from dev.tests.server_fixtures import (
    FOREVER,
    FakeConstraintFactory,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    Plan,
    chat_body,
    grammar_tokenizers,
    make_frontend,
    rich_weather_tool,
)
from dev.tests.tool_output import (
    argument_grammar,
    project,
    put,
    streamed_text,
    tool_policy,
)
from server import backend as backend_api
from server import output as model_output
from server import server as api
from server import tool_schema

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "enum": ["alpha"]}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {"type": "function", "function": {"name": "finish", "parameters": {}}},
]
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"const": 42}, "marker": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
ANSWER = '{"answer":42}'
CALL = (
    "<tool_call>\n<function=lookup>\n<parameter=query>\nalpha\n"
    "</parameter>\n</function>\n</tool_call>"
)
OTHER_CALL = "<tool_call>\n<function=finish>\n</function>\n</tool_call>"


def policy(choice="auto", parallel=True, tools=TOOLS):
    return tool_schema.normalize_tools(tools, choice, parallel)[1]


def strict_tools(*tools):
    return [
        {"type": "function", "function": {**tool["function"], "strict": True}}
        for tool in tools
    ]


class StructuredToolGrammarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.guidance = grammar_tokenizers()

    def matcher(self, thinking=False, schema=SCHEMA, **settings):
        with mock.patch.object(
            tool_schema, "THINK_END_TOKEN_ID", self.tokenizer.token_to_id("</think>")
        ):
            grammar = tool_schema.tool_grammar(policy(**settings), thinking, schema)
        self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
        return LLMatcher(self.guidance, grammar)

    def accepts(self, text, *, byte_tokens=False, **settings):
        matcher = self.matcher(**settings)
        tokens = list(text.encode()) if byte_tokens else self.tokenizer.encode(text).ids
        return (
            matcher.validate_tokens(tokens) == len(tokens)
            and matcher.consume_tokens(tokens)
            and matcher.is_accepting()
        )

    def assert_complete(self, text, **settings):
        self.assertTrue(self.accepts(text, **settings), text)

    def assert_not_complete(self, text, **settings):
        self.assertFalse(self.accepts(text, **settings), text)

    def test_a_grammar_applies_by_choice_and_strict_tools(self):
        # An auto choice of parallel tools none of which is strict leaves the
        # output free; a choice that requires a call, one that allows a single
        # call, and a strict tool need the call envelope, and strict tools
        # have their arguments pinned.
        named = {"type": "function", "function": {"name": "finish"}}
        cases = (
            ({}, frozenset(), False),
            ({"choice": "required"}, frozenset(), True),
            ({"choice": named}, frozenset(), True),
            ({"parallel": False}, frozenset(), True),
            (
                {"tools": [TOOLS[0], *strict_tools(TOOLS[1])]},
                frozenset({"finish"}),
                True,
            ),
        )
        for settings, strict, constrained in cases:
            with self.subTest(settings=settings):
                chosen = policy(**settings)
                self.assertEqual(
                    (chosen.strict, chosen.constrained), (strict, constrained)
                )
        named = policy(choice=named)
        self.assertEqual(
            (list(named.schemas), named.required, named.parallel),
            (["finish"], True, False),
        )

    def test_auto_selects_one_json_answer_or_tool_calls(self):
        # As under a required choice, beside the answer: a call first, then
        # text and further calls.
        for text in (
            ANSWER,
            " \n" + ANSWER + "\t",
            CALL,
            CALL + "\n" + OTHER_CALL,
            CALL + "\nDone.",
            CALL.replace("alpha", "beta"),
        ):
            with self.subTest(text=text):
                self.assert_complete(text)
        for text in (
            "",
            "plain answer",
            '{"answer":41}',
            '{"answer":42,"extra":1}',
            ANSWER + ANSWER,
            ANSWER + CALL,
            "explanation" + CALL,
        ):
            with self.subTest(text=text):
                self.assert_not_complete(text)

    def test_required_and_named_tool_choice_exclude_json(self):
        for choice in (
            "required",
            {"type": "function", "function": {"name": "lookup"}},
        ):
            with self.subTest(choice=choice):
                self.assert_complete(CALL, choice=choice)
                self.assert_not_complete(ANSWER, choice=choice)
        self.assert_not_complete(
            OTHER_CALL, choice={"type": "function", "function": {"name": "lookup"}}
        )

    def test_required_and_named_calls_follow_the_reasoning_directly(self):
        # Reasoning stays free; after it only whitespace may precede a call.
        named = {"type": "function", "function": {"name": "lookup"}}
        for choice in ("required", named):
            for thinking, reasoning in ((False, ""), (True, "Look it up.</think>")):
                settings = {"choice": choice, "thinking": thinking, "schema": None}
                with self.subTest(choice=choice, thinking=thinking):
                    for text in (CALL, "\n\n" + CALL, CALL + "\n"):
                        self.assert_complete(reasoning + text, **settings)
                    for text in ("", "plain answer", "Sure. " + CALL):
                        self.assert_not_complete(reasoning + text, **settings)
        # A required choice may call several tools, with text between and
        # after them; a named one calls exactly once.
        for text in (CALL + "\n" + OTHER_CALL, CALL + " Next. " + CALL + " Done."):
            self.assert_complete(text, choice="required", schema=None)
        self.assert_not_complete(CALL + "\n" + CALL, choice=named, schema=None)

    def test_a_call_with_too_many_parameters_fails_where_it_begins(self):
        # The grammar compiles, but the parser cannot admit every parameter
        # name of a strict tool at once: requests check each strict call's
        # opening before prefill.
        properties = {f"p{index}": {"type": "string"} for index in range(1100)}
        tools = strict_tools(
            {"function": {"name": "wide", "parameters": {"properties": properties}}}
        )
        grammar = tool_schema.tool_grammar(policy(tools=tools), False)
        self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
        opening = self.tokenizer.encode(
            tool_schema.TOOL_CALL_OPEN + tool_schema.function_opening("wide")
        ).ids
        matcher = LLMatcher(self.guidance, grammar, log_level=0)
        self.assertFalse(matcher.consume_tokens(opening))

    def test_a_strict_tool_takes_only_the_arguments_it_declares(self):
        # A strict tool's arguments come in schema order; a tool that is not
        # strict takes any arguments.
        parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {"depth": {"type": "integer"}},
                },
                "tags": {"type": "array"},
            },
            "required": ["path"],
        }

        def call(*arguments):
            return (
                "<tool_call>\n<function=search>\n"
                + "".join(
                    f"<parameter={name}>\n{value}\n</parameter>\n"
                    for name, value in arguments
                )
                + "</function>\n</tool_call>"
            )

        declared = call(("path", "src"), ("options", '{"depth":2}'), ("tags", "[]"))
        undeclared = (
            call(("path", "src"), ("target", '"files"')),
            call(("path", "src"), ("options", '{"depth":2,"deep":true}')),
            call(("path", "src"), ("tags", '["a"]')),
            call(("tags", "[]"), ("path", "src")),
            call(("options", "deep")),
        )
        for strict in (False, True):
            tool = {"name": "search", "parameters": parameters, "strict": strict}
            tools = [{"type": "function", "function": tool}]
            grammar = tool_schema.tool_grammar(policy("required", tools=tools), False)
            for text in (declared, *undeclared):
                with self.subTest(strict=strict, text=text):
                    matcher = LLMatcher(self.guidance, grammar)
                    tokens = self.tokenizer.encode(text).ids
                    accepted = (
                        matcher.validate_tokens(tokens) == len(tokens)
                        and matcher.consume_tokens(tokens)
                        and matcher.is_accepting()
                    )
                    self.assertEqual(accepted, text == declared or not strict)

    def test_parameters_of_a_strict_tool_come_in_schema_order(self):
        # A parameter left out cannot come later.
        tools = strict_tools(
            {
                "function": {
                    "name": "roll_cut",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cut_at": {"type": "string"},
                            "shift": {"type": "number"},
                            "item": {"type": "string"},
                        },
                        "required": ["item"],
                    },
                }
            }
        )

        def call(*arguments):
            return (
                "<tool_call>\n<function=roll_cut>\n"
                + "".join(
                    f"<parameter={name}>\n{value}\n</parameter>\n"
                    for name, value in arguments
                )
                + "</function>\n</tool_call>"
            )

        settings = {"choice": "required", "tools": tools, "schema": None}
        for text in (call(("shift", "-0.5"), ("item", "s1")), call(("item", "s1"))):
            self.assert_complete(text, **settings)
        for text in (
            call(("item", "s1"), ("shift", "-0.5")),
            call(("shift", "-0.5")),
            call(("shift", "soon"), ("item", "s1")),
        ):
            self.assert_not_complete(text, **settings)

    def test_strict_values_are_written_as_text_raw_or_json(self):
        # Enumerated values as their text, a value that may be a string as
        # raw text, any other as JSON.
        schema = {
            "type": "object",
            "properties": {
                "mode": {"enum": ["fast", 2, None]},
                "time": {"type": ["string", "number"]},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "count": {"type": "integer"},
            },
            "required": ["mode", "time", "note", "count"],
        }
        grammar = argument_grammar(schema)
        for mode in ("fast", "2", "null"):
            self.assertIn(json.dumps(mode), grammar)
        self.assertEqual(grammar.count("/(?s:.*)/"), 2)
        self.assertIn('%json {"type":"integer"', grammar)

    def test_strict_values_keep_their_constraints(self):
        # A union whose string member lists its values takes those, the other
        # members' JSON and nothing else; a refinement of a type keeps it.
        tool = {
            "type": "function",
            "function": {
                "name": "f",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "$defs": {"color": {"enum": ["red", "green"]}},
                    "properties": {
                        "unit": {
                            "anyOf": [
                                {"type": "string", "enum": ["celsius", "kelvin"]},
                                {"type": "null"},
                            ]
                        },
                        "limit": {
                            "anyOf": [
                                {"enum": ["auto"]},
                                {"type": "integer", "minimum": 1},
                            ]
                        },
                        "count": {"allOf": [{"type": "integer"}, {"minimum": 0}]},
                        "color": {"allOf": [{"$ref": "#/$defs/color"}]},
                        "time": {"type": ["string", "number"]},
                    },
                },
            },
        }
        settings = {"choice": "required", "schema": None, "tools": [tool]}
        call = (
            "<tool_call>\n<function=f>\n<parameter={}>\n{}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        for name, taken, refused in (
            ("unit", ("celsius", "null"), ("fahrenheit", '"celsius"')),
            ("limit", ("auto", "5"), ("banana", "0")),
            ("count", ("3",), ("abc", "-1")),
            ("color", ("red",), ("blue",)),
            ("time", ("12:00", "12"), ()),
        ):
            for value in taken:
                with self.subTest(name=name, value=value):
                    self.assert_complete(call.format(name, value), **settings)
            for value in refused:
                with self.subTest(name=name, value=value):
                    self.assert_not_complete(call.format(name, value), **settings)

    def test_whitespace_between_tokens_is_bounded(self):
        # A model preferring whitespace to every token the grammar allows next
        # must write one of them within 64 characters; strings keep theirs.
        bound = tool_schema.MAX_WHITESPACE
        for spaces, complete in ((bound, True), (bound + 1, False)):
            with self.subTest(spaces=spaces):
                pad = " " * spaces
                for text in ("{" + pad + '"answer":42}', pad + ANSWER):
                    self.assertEqual(self.accepts(text), complete, text)
                grammar = tool_schema.json_grammar(SCHEMA, False)
                matcher = LLMatcher(self.guidance, grammar)
                tokens = self.tokenizer.encode("{" + pad + '"answer":42}').ids
                self.assertEqual(
                    matcher.validate_tokens(tokens) == len(tokens)
                    and matcher.consume_tokens(tokens)
                    and matcher.is_accepting(),
                    complete,
                )
        self.assert_complete('{"answer":42,"marker":"' + " " * 200 + '"}')
        # A strict tool's JSON value is bounded alike.
        body = {"type": "object", "additionalProperties": True}
        tools = strict_tools(
            {"function": {"name": "note", "parameters": {"properties": {"body": body}}}}
        )
        grammar = tool_schema.tool_grammar(policy(tools=tools), False)
        for spaces, complete in ((bound, True), (bound + 1, False)):
            text = (
                "<tool_call>\n<function=note>\n<parameter=body>\n{"
                + " " * spaces
                + '"a":1}\n</parameter>\n</function>\n</tool_call>'
            )
            with self.subTest(strict_parameter=spaces):
                matcher = LLMatcher(self.guidance, grammar)
                tokens = self.tokenizer.encode(text).ids
                self.assertEqual(
                    matcher.validate_tokens(tokens) == len(tokens)
                    and matcher.consume_tokens(tokens)
                    and matcher.is_accepting(),
                    complete,
                )

    def test_strict_closing_leaves_a_schema_that_others_extend(self):
        schema = {
            "type": "object",
            "properties": {
                "point": {"$ref": "#/$defs/Point"},
                "shape": {
                    "anyOf": [
                        {"type": "object", "properties": {"r": {}}},
                        {"type": "null"},
                    ]
                },
                "pair": {"type": "array", "items": [{"type": "integer"}]},
                "open": {"type": "object", "additionalProperties": True},
                "named": {"type": "object", "patternProperties": {"^x": {}}},
                "any": {},
            },
            "$defs": {"Point": {"type": "object", "properties": {"x": {}}}},
        }
        strict = tool_schema._strict_schema(schema)
        self.assertNotIn("additionalProperties", schema)
        self.assertIs(strict["additionalProperties"], False)
        self.assertIs(strict["$defs"]["Point"]["additionalProperties"], False)
        shape = strict["properties"]["shape"]["anyOf"][0]
        self.assertIs(shape["additionalProperties"], False)
        self.assertEqual(strict["properties"]["point"], {"$ref": "#/$defs/Point"})
        self.assertIs(strict["properties"]["pair"]["additionalItems"], False)
        self.assertIs(strict["properties"]["open"]["additionalProperties"], True)
        self.assertNotIn("additionalProperties", strict["properties"]["named"])
        self.assertEqual(strict["properties"]["any"], {})
        # An allOf extends its members: closing one would refuse the others'
        # properties, so the schema is left as declared.
        extended = {
            "allOf": [
                {"type": "object", "properties": {"a": {}}},
                {"properties": {"b": {}}},
            ]
        }
        self.assertIs(tool_schema._strict_schema(extended), extended)
        tool = {"name": "search", "parameters": {}, "strict": "yes"}
        with self.assertRaisesRegex(tool_schema.APIError, "strict must be a boolean"):
            tool_schema.normalize_tools(
                [{"type": "function", "function": tool}], "auto", True
            )

    def test_none_keeps_the_tools_but_lets_no_call_start(self):
        # The prompt renders the tools as for any choice; only output changes.
        tools, none = tool_schema.normalize_tools(TOOLS, "none", True)
        self.assertEqual(
            (tools, none.schemas, none.required, none.strict, none.constrained),
            (TOOLS, {}, False, frozenset(), True),
        )
        self.assert_complete(ANSWER, choice="none")
        for text in (CALL, ANSWER + CALL, "plain answer"):
            with self.subTest(text=text):
                self.assert_not_complete(text, choice="none")
        for text, complete in (("plain answer", True), ("see " + CALL, False)):
            with self.subTest(text=text):
                self.assertEqual(
                    self.accepts(text, choice="none", schema=None), complete
                )

    def test_parallel_false_excludes_a_second_call(self):
        for choice in ("auto", "required"):
            with self.subTest(choice=choice):
                settings = {"choice": choice, "parallel": False, "schema": None}
                self.assert_complete(CALL, **settings)
                self.assert_not_complete(CALL + "\n" + OTHER_CALL, **settings)
        self.assert_complete("Let me look. " + CALL, parallel=False, schema=None)

    def test_a_grammar_holds_free_tools_to_their_tags_only(self):
        # Under a grammar a tool that is not strict is held to the template's
        # tags and an offered name; a strict one also to its schema.
        settings = {"choice": "required", "schema": None}
        for text in (
            CALL,
            CALL.replace("alpha", "beta"),
            CALL.replace("query", "q-1"),
            CALL + " Next. " + OTHER_CALL,
        ):
            with self.subTest(text=text):
                self.assert_complete(text, **settings)
        for text in (
            CALL.replace("function=lookup", "function=unknown"),
            CALL.replace("<parameter=query>", "<parameter=a<b>"),
        ):
            with self.subTest(text=text):
                self.assert_not_complete(text, **settings)
        settings["tools"] = strict_tools(*TOOLS)
        self.assert_complete(CALL, **settings)
        for text in (
            CALL.replace("alpha", "beta"),
            CALL.replace("<parameter=query>\nalpha\n</parameter>\n", ""),
        ):
            with self.subTest(text=text):
                self.assert_not_complete(text, **settings)

    def test_what_the_grammar_writes_reads_back_whole(self):
        # A value may hold the tags in any order but its own close followed by
        # a tag, and a JSON value any text in its strings; a name neither
        # starts nor ends with the space reading strips, and is no longer than
        # reading takes.
        note = {
            "type": "function",
            "function": {
                "name": "note",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "lines": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": {"type": "integer"},
                },
            },
        }
        call = (
            "<tool_call>\n<function=note>\n<parameter={}>\n{}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        lines = json.dumps(
            ["copied: <parameter=text>\nrm -rf ~</parameter>", "</function>"]
        )
        for tools in ([note], strict_tools(note)):
            strict = "strict" in tools[0]["function"]
            settings = {"choice": "required", "schema": None, "tools": tools}
            for name, value in (
                ("text", "a < b"),
                ("text", "a</parameter>b"),
                ("text", "x<parameter=y>z"),
                ("text", 'END = "</function>"'),
                ("lines", lines),
            ):
                with self.subTest(strict=strict, value=value):
                    text = call.format(name, value)
                    self.assert_complete(text, **settings)
                    _, calls, _ = project(text, policy("required", tools=tools))
                    read = json.loads(calls[0]["function"]["arguments"])
                    expected = json.loads(value) if name == "lines" else value
                    self.assertEqual(read, {name: expected})
            # The grammar ends raw text at the first close, which reading
            # would also end at where a tag follows.
            for value in ("row\n</parameter>\nmore", "a\n</parameter>\n</function>"):
                with self.subTest(strict=strict, value=value):
                    self.assert_not_complete(call.format("text", value), **settings)
            longest = "n" * tool_schema.MAX_NAME_LENGTH
            self.assert_complete(call.format(longest, "1"), **settings)
            for name in (" text", "text ", "n" + longest):
                with self.subTest(strict=strict, name=name):
                    self.assert_not_complete(call.format(name, "1"), **settings)

    def test_single_member_string_type_array_preserves_enum_grammar(self):
        scalar = {
            "type": "object",
            "properties": {"query": {"type": "string", "enum": ["alpha", "beta"]}},
            "required": ["query"],
        }
        array = json.loads(json.dumps(scalar))
        array["properties"]["query"]["type"] = ["string"]
        self.assertEqual(
            argument_grammar(scalar),
            argument_grammar(array),
        )

    def test_json_strings_can_contain_tool_delimiter_bytes(self):
        text = json.dumps({"answer": 42, "marker": CALL})
        self.assert_complete(text, byte_tokens=True)

    def test_thinking_prefix_requires_close_before_answer_or_tools(self):
        for text in (ANSWER, CALL):
            with self.subTest(text=text):
                self.assert_complete(
                    "Reason through this. </think>" + text, thinking=True
                )
                self.assert_not_complete(text, thinking=True)

    def test_thinking_cannot_spell_its_close_in_text(self):
        # The reasoning splitter ends thinking at the first decoded
        # "</think>", so an ordinary-token spelling must not stay in thinking.
        close = self.tokenizer.token_to_id("</think>")
        with mock.patch.object(tool_schema, "THINK_END_TOKEN_ID", close):
            grammars = {
                "json": tool_schema.json_grammar(SCHEMA, True),
                "tools": tool_schema.tool_grammar(policy(), True, SCHEMA),
            }
        tokens = [*b"Reason. </think> More.", close, *ANSWER.encode()]
        for name, grammar in grammars.items():
            with self.subTest(grammar=name):
                self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
                matcher = LLMatcher(self.guidance, grammar)
                self.assertEqual(
                    matcher.validate_tokens(tokens), len(b"Reason. </think")
                )

    def test_truncated_json_and_tool_prefixes_remain_nonterminal(self):
        for text in ('{"answer":', CALL.partition("</parameter>")[0]):
            with self.subTest(text=text):
                matcher = self.matcher()
                tokens = self.tokenizer.encode(text).ids
                self.assertEqual(matcher.validate_tokens(tokens), len(tokens))
                self.assertTrue(matcher.consume_tokens(tokens))
                self.assertFalse(matcher.is_accepting())

    def test_strict_string_constraints_leave_the_value_raw(self):
        # A strict tool writes a string as raw text, which its grammar does
        # not pattern or bound; nothing checks the call after.
        for value_schema in (
            {"type": "string", "minLength": 4, "maxLength": 16, "pattern": "^[A-Z]+$"},
            {
                "type": "string",
                "description": "Recipient",
                "allOf": [
                    {"pattern": r"^[^\n\r]*$"},
                    {"pattern": r"^[\s\S]{0,300}$"},
                ],
            },
        ):
            with self.subTest(schema=value_schema):
                schema = {
                    "type": "object",
                    "properties": {"text": value_schema},
                    "required": ["text"],
                }
                self.assertIn("/(?s:.*)/", argument_grammar(schema))
                text = (
                    "<tool_call>\n<function=echo>\n<parameter=text>\nx\ny\n"
                    "</parameter>\n</function>\n</tool_call>"
                )
                _, calls, _ = project(text, tool_policy({"echo": schema}))
                self.assertEqual(
                    json.loads(calls[0]["function"]["arguments"]), {"text": "x\ny"}
                )

    def test_tool_grammar_excludes_text_spellings_of_control_tokens(self):
        schema = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        policy = tool_schema.ToolPolicy(
            schemas={"bash": schema},
            required=False,
            parallel=False,
            strict=frozenset(),
            constrained=True,
        )
        grammar = json.loads(tool_schema.tool_grammar(policy, True))
        main = grammar["grammars"][0]["lark_grammar"]
        self.assertIn(
            r"TEXT: /(?s:.*)/ & ~/(?s:.*)(<tool_call>|<\/think>)(?s:.*)/",
            main,
        )

    def test_tool_choice_shapes_the_grammar_policy_not_the_prompt(self):
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(
            FakeRuntime(), tokenizer, lambda _record: None
        )
        self.addCleanup(backend.close)
        app = make_frontend(tokenizer, backend, "test-model", 128, 1, 2, vision=True)
        tools = [
            {"type": "function", "function": {"name": "f"}},
            {"type": "function", "function": {"name": "g"}},
        ]
        named = {"type": "function", "function": {"name": "g"}}
        both = "(call_0 | call_1)"
        cases = (
            ({}, False, True, None),
            ({"tool_choice": "none"}, False, True, "TEXT"),
            ({"tool_choice": "required"}, True, True, f"WS {both} (TEXT {both})* TEXT"),
            ({"tool_choice": named}, True, False, "WS (call_0) WS"),
            ({"parallel_tool_calls": False}, False, False, f"TEXT ({both} WS)?"),
        )
        for extra, required, parallel, start in cases:
            with self.subTest(**extra):
                job = app.prepare(chat_body(tools=tools, **extra), deadline=FOREVER)
                rendered, kwargs = tokenizer.templates[-1]
                self.assertEqual(rendered, [{"role": "user", "content": "hello"}])
                self.assertEqual(kwargs["tools"], tools)
                policy = job.tool_policy
                self.assertEqual(
                    (policy.required, policy.parallel), (required, parallel)
                )
                if start is None:
                    # An auto choice of tools none of which is strict leaves
                    # the output free.
                    self.assertFalse(policy.constrained)
                    continue
                main = json.loads(tool_schema.tool_grammar(policy, True))["grammars"][0]
                self.assertIn(f"start: think {start}\n", main["lark_grammar"])
                self.assertEqual(
                    [
                        name
                        for name in "fg"
                        if f"<function={name}>" in main["lark_grammar"]
                    ],
                    list(policy.schemas),
                )


class StructuredToolProjectionTest(unittest.TestCase):
    def job(self):
        _, validator = tool_schema.normalize_response_format(
            {"type": "json_schema", "json_schema": {"schema": SCHEMA}}
        )
        return SimpleNamespace(
            public_id="request", tool_policy=policy(), response_validator=validator
        )

    def finalize(self, text, job, incomplete=False, size=None):
        """Project `text` as a request does, whole or in chunks of `size`
        characters, and finalize it. Returns the content, the calls and the
        text the stream sends, the finish's owed text included."""
        projector = model_output.StreamingToolCallProjector(
            job.tool_policy, job.public_id, True
        )
        events = put(projector, text, size)
        content, calls, owed = api.FrontendHandler._finalize_content(
            None, "", job, incomplete, projector
        )
        return content, calls, streamed_text(events + owed)

    def test_json_tool_spellings_stream_as_text_at_every_split(self):
        text = json.dumps({"answer": 42, "marker": CALL})
        job = self.job()
        self.assertEqual(self.finalize(text, job), (text, [], text))
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                projector = model_output.StreamingToolCallProjector(
                    job.tool_policy, job.public_id, True
                )
                events = projector.put(text[:split]) + projector.put(text[split:])
                content, calls, owed = projector.finish(False)
                self.assertEqual((content, calls), (text, []))
                self.assertTrue(all(kind == "content" for kind, _ in events + owed))
                self.assertEqual(streamed_text(events + owed), text)

    def test_partial_json_is_preserved_and_partial_tool_is_not_completed(self):
        job = self.job()
        partial_json = '{"answer":42,"marker":"<tool_call>'
        for text, content in (
            (partial_json, partial_json),
            (" \n" + CALL[:20], " \n"),
            (" ", " "),
        ):
            for size in (1, None):
                with self.subTest(text=text, size=size):
                    self.assertEqual(
                        self.finalize(text, job, True, size), (content, [], content)
                    )

    def test_a_cut_after_a_call_reports_and_streams_no_text(self):
        job = self.job()
        text = " \n" + CALL + "\n" + OTHER_CALL + "\n"
        for end in range(len(" \n" + CALL), len(text) + 1):
            cut = text[:end]
            for size in (1, None):
                with self.subTest(cut=cut, size=size):
                    content, calls, streamed = self.finalize(cut, job, True, size)
                    self.assertEqual((content, streamed), ("", ""))
                    self.assertEqual(calls[0]["function"]["name"], "lookup")

    def test_finalization_validates_an_answer_without_calls(self):
        # Generation enforces the tool choice; only the answer's schema is
        # checked on the complete output.
        with self.assertRaisesRegex(api.APIError, "invalid structured output"):
            self.finalize('{"answer":41}', self.job())
        for text, names in (
            (CALL + "\n" + OTHER_CALL, ["lookup", "finish"]),
            (CALL + ' {"answer":41}', ["lookup"]),
        ):
            with self.subTest(text=text):
                _, calls, _ = self.finalize(text, self.job())
                self.assertEqual([call["function"]["name"] for call in calls], names)
        self.assertEqual(self.finalize(ANSWER, self.job())[:2], (ANSWER, []))

    def test_structured_stream_keeps_tool_markers_inside_json_strings(self):
        _, policy = tool_schema.normalize_tools([rich_weather_tool()], "auto", True)
        payload = ' \n{"text":"<tool_call>\\n<function=weather> and </think>"}'
        for width in (1, 2, 7, len(payload)):
            with self.subTest(width=width):
                projector = model_output.StreamingToolCallProjector(
                    policy, "test", structured=True
                )
                events = []
                for start in range(0, len(payload), width):
                    events.extend(projector.put(payload[start : start + width]))
                content, _, owed = projector.finish(False)
                self.assertEqual(content, payload)
                self.assertTrue(all(kind == "content" for kind, _ in events + owed))
                self.assertEqual("".join(value for _, value in events + owed), payload)


class StructuredToolRequestTest(HarnessTestCase):
    def test_structured_output_validation_and_constraint(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        }
        factory = FakeConstraintFactory()
        harness = self.harness(FakeRuntime(Plan([[10]])), constraint_factory=factory)
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(reasoning_effort="none", response_format=response_format),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"], '{"x":3}'
        )
        self.assertIn("%json", factory.grammars[0])

        invalid = self.harness(FakeRuntime(Plan([[4]])))
        status, _, payload = invalid.request(
            "POST",
            "/v1/chat/completions",
            chat_body(reasoning_effort="none", response_format=response_format),
        )
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(payload)["error"]["code"], "invalid_model_output")

    def test_tools_and_response_format_allow_calls_or_a_valid_final_answer(self):
        factory = FakeConstraintFactory()
        harness = self.harness(
            FakeRuntime(Plan([[10]]), Plan([[5]]), Plan([[4]])),
            constraint_factory=factory,
        )
        body = chat_body(
            tools=[rich_weather_tool()],
            reasoning_effort="none",
            response_format={"type": "json_object"},
        )
        status, _, payload = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"], '{"x":3}'
        )
        status, _, payload = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200, payload)
        call = json.loads(payload)["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})
        status, _, payload = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 500, payload)
        self.assertEqual(json.loads(payload)["error"]["code"], "invalid_model_output")
        grammar = json.loads(factory.grammars[0])["grammars"][0]["lark_grammar"]
        self.assertIn("| answer)", grammar)
        self.assertIn("%json", grammar)

    def test_structured_output_preserves_prompt_messages(self):
        harness = self.harness(FakeRuntime())
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        for system in ([], [{"role": "system", "content": "Original instructions."}]):
            with self.subTest(system=system):
                body = chat_body(
                    messages=[*system, {"role": "user", "content": "Calculate 6 * 7."}],
                    tools=[rich_weather_tool()],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"schema": schema},
                    },
                )
                original = json.dumps(body, sort_keys=True)
                prompt = harness.app._prepare_prompt(body, None, deadline=FOREVER)
                self.assertEqual(json.dumps(body, sort_keys=True), original)
                self.assertEqual(prompt.messages, body["messages"])
                self.assertEqual(prompt.response_schema, schema)
                self.assertIsNotNone(prompt.response_validator)

    def test_structured_output_keeps_required_tool_choice(self):
        # A forced call leaves the response format out of the grammar.
        factory = FakeConstraintFactory()
        harness = self.harness(FakeRuntime(Plan([[5]])), constraint_factory=factory)
        body = chat_body(
            tools=[rich_weather_tool()],
            tool_choice="required",
            reasoning_effort="none",
            response_format={"type": "json_object"},
        )
        status, _, payload = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200, payload)
        main = json.loads(factory.grammars[-1])["grammars"][0]["lark_grammar"]
        self.assertNotIn("answer", main)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["tool_calls"][0]["function"][
                "name"
            ],
            "weather",
        )

    def test_structured_tool_response_streams_and_recovers_from_length(self):
        harness = self.harness(FakeRuntime(Plan([[5]]), Plan([[13]], reason="length")))
        body = chat_body(
            tools=[rich_weather_tool()],
            reasoning_effort="none",
            response_format={"type": "json_object"},
            stream=True,
        )
        for incomplete in (False, True):
            status, _, payload = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 200, payload)
            self.assertNotIn(b"invalid_model_output", payload)
            chunks = [
                json.loads(line[6:])
                for line in payload.decode().splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            self.assertEqual(
                chunks[-1]["choices"][0]["finish_reason"],
                "length" if incomplete else "tool_calls",
            )

    def test_chat_length_safely_finishes_partial_structured_and_tool_output(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        }
        runtime = FakeRuntime(
            Plan([[12]], reason="length"),
            Plan([[13]], reason="length"),
            Plan([[13]], reason="length"),
        )
        harness = self.harness(runtime, constraint_factory=FakeConstraintFactory())
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(reasoning_effort="none", response_format=response_format),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(response["choices"][0]["finish_reason"], "length")
        self.assertEqual(response["choices"][0]["message"]["content"], '{"x":')

        tool = rich_weather_tool()
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(tools=[tool], reasoning_effort="none"),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(response["choices"][0]["finish_reason"], "length")
        self.assertNotIn("<tool_call>", payload.decode())

        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(tools=[tool], reasoning_effort="none", stream=True),
        )
        self.assertEqual(status, 200, payload)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertNotIn("<tool_call>", payload.decode())


if __name__ == "__main__":
    unittest.main()
