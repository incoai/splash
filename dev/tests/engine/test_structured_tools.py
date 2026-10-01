import json
import unittest
from types import SimpleNamespace
from unittest import mock

from llguidance import LLMatcher, LLTokenizer
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from dev.tests.test_server import _byte_alphabet
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


def policy(choice="auto", parallel=True):
    return tool_schema.normalize_tools(TOOLS, choice, parallel)[1]


class StructuredToolGrammarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocabulary = {char: token for token, char in _byte_alphabet().items()}
        vocabulary["[UNK]"] = len(vocabulary)
        vocabulary["<eos>"] = len(vocabulary)
        cls.tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        cls.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
                pre_tokenizers.Split(Regex(""), behavior="isolated"),
            ]
        )
        cls.tokenizer.decoder = decoders.ByteLevel()
        cls.tokenizer.add_special_tokens(
            ["<tool_call>", "</tool_call>", "</think>", "<eos>"]
        )
        cls.guidance = LLTokenizer(
            cls.tokenizer.to_str(), eos_token=cls.tokenizer.token_to_id("<eos>")
        )

    def matcher(self, choice="auto", parallel=True, thinking=False, schema=SCHEMA):
        with mock.patch.object(
            tool_schema, "THINK_END_TOKEN_ID", self.tokenizer.token_to_id("</think>")
        ):
            grammar = tool_schema.tool_grammar(
                policy(choice, parallel), thinking, schema
            )
        self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
        return LLMatcher(self.guidance, grammar)

    def assert_complete(self, text, *, byte_tokens=False, **settings):
        matcher = self.matcher(**settings)
        tokens = list(text.encode()) if byte_tokens else self.tokenizer.encode(text).ids
        self.assertEqual(matcher.validate_tokens(tokens), len(tokens), text)
        self.assertTrue(matcher.consume_tokens(tokens), text)
        self.assertTrue(matcher.is_accepting(), text)

    def assert_not_complete(self, text, **settings):
        matcher = self.matcher(**settings)
        tokens = self.tokenizer.encode(text).ids
        accepted = matcher.validate_tokens(tokens)
        if accepted == len(tokens):
            self.assertTrue(matcher.consume_tokens(tokens))
            self.assertFalse(matcher.is_accepting(), text)

    def test_auto_selects_one_json_answer_or_tool_calls(self):
        for text in (ANSWER, " \n" + ANSWER + "\t", CALL, CALL + "\n" + OTHER_CALL):
            with self.subTest(text=text):
                self.assert_complete(text)
        for text in (
            "",
            "plain answer",
            '{"answer":41}',
            '{"answer":42,"extra":1}',
            ANSWER + ANSWER,
            CALL + ANSWER,
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
        # A required choice may call several tools, a named one exactly once.
        self.assert_complete(CALL + "\n" + OTHER_CALL, choice="required", schema=None)
        self.assert_not_complete(CALL + "\n" + CALL, choice=named, schema=None)

    def test_a_call_with_too_many_parameters_fails_where_it_begins(self):
        # The grammar compiles, but the parser cannot admit every parameter
        # name at once: requests check each call's opening before prefill.
        properties = {f"p{index}": {"type": "string"} for index in range(1100)}
        tools = [
            {
                "type": "function",
                "function": {"name": "wide", "parameters": {"properties": properties}},
            }
        ]
        grammar = tool_schema.tool_grammar(
            tool_schema.normalize_tools(tools, "auto", True)[1], False
        )
        self.assertFalse(LLMatcher.validate_grammar(grammar, self.guidance))
        opening = self.tokenizer.encode(
            tool_schema.TOOL_CALL_OPEN + tool_schema.function_opening("wide")
        ).ids
        matcher = LLMatcher(self.guidance, grammar, log_level=0)
        self.assertFalse(matcher.consume_tokens(opening))

    def test_a_strict_tool_takes_only_the_arguments_it_declares(self):
        # As vLLM and SGLang generate a strict tool's arguments; a tool that is
        # not strict takes whatever its schema allows.
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
        )
        for strict in (False, True):
            tool = {"name": "search", "parameters": parameters, "strict": strict}
            grammar = tool_schema.tool_grammar(
                tool_schema.normalize_tools(
                    [{"type": "function", "function": tool}], "auto", True
                )[1],
                False,
            )
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

    def test_whitespace_between_tokens_is_bounded(self):
        # A model preferring whitespace to every token the grammar allows next
        # must write one of them within 64 characters; strings keep theirs.
        bound = tool_schema.MAX_WHITESPACE
        for spaces, complete in ((bound, True), (bound + 1, False)):
            with self.subTest(spaces=spaces):
                pad = " " * spaces
                for text in (
                    "{" + pad + '"answer":42}',
                    pad + ANSWER,
                    CALL + "\n" * spaces + OTHER_CALL,
                ):
                    if complete:
                        self.assert_complete(text)
                    else:
                        self.assert_not_complete(text)
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
        # Any value, as an open parameter takes, is bounded alike.
        tool = {"name": "note", "parameters": {"type": "object"}}
        grammar = tool_schema.tool_grammar(
            tool_schema.normalize_tools(
                [{"type": "function", "function": tool}], "auto", True
            )[1],
            False,
        )
        for spaces, complete in ((bound, True), (bound + 1, False)):
            text = (
                "<tool_call>\n<function=note>\n<parameter=body>\n{"
                + " " * spaces
                + '"a":1}\n</parameter>\n</function>\n</tool_call>'
            )
            with self.subTest(open_parameter=spaces):
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
        self.assertEqual((tools, none.schemas, none.required), (TOOLS, {}, False))
        self.assert_complete(ANSWER, choice="none")
        for text in (CALL, ANSWER + CALL, "plain answer"):
            with self.subTest(text=text):
                self.assert_not_complete(text, choice="none")
        grammar = tool_schema.tool_grammar(none, False)
        for text, complete in (("plain answer", True), ("see " + CALL, False)):
            with self.subTest(text=text):
                matcher = LLMatcher(self.guidance, grammar)
                tokens = self.tokenizer.encode(text).ids
                accepted = (
                    matcher.validate_tokens(tokens) == len(tokens)
                    and matcher.consume_tokens(tokens)
                    and matcher.is_accepting()
                )
                self.assertEqual(accepted, complete)

    def test_parallel_false_excludes_a_second_call(self):
        for choice in ("auto", "required"):
            with self.subTest(choice=choice):
                self.assert_complete(CALL, choice=choice, parallel=False)
                self.assert_not_complete(
                    CALL + "\n" + OTHER_CALL, choice=choice, parallel=False
                )

    def test_tool_arguments_are_constrained_independently_of_answer_schema(self):
        self.assert_complete(CALL)
        for text in (
            CALL.replace("alpha", "beta"),
            CALL.replace("<parameter=query>\nalpha\n</parameter>\n", ""),
            CALL.replace("function=lookup", "function=unknown"),
        ):
            with self.subTest(text=text):
                self.assert_not_complete(text)

    def test_single_member_string_type_array_preserves_enum_grammar(self):
        scalar = {
            "type": "object",
            "properties": {"query": {"type": "string", "enum": ["alpha", "beta"]}},
            "required": ["query"],
        }
        array = json.loads(json.dumps(scalar))
        array["properties"]["query"]["type"] = ["string"]
        self.assertEqual(
            tool_schema._tool_arguments_grammar(scalar),
            tool_schema._tool_arguments_grammar(array),
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


class StructuredToolProjectionTest(unittest.TestCase):
    def job(self, choice="auto", parallel=True):
        _, validator = tool_schema.normalize_response_format(
            {"type": "json_schema", "json_schema": {"schema": SCHEMA}}
        )
        return SimpleNamespace(
            public_id="request",
            tool_policy=policy(choice, parallel),
            response_validator=validator,
        )

    def finalize(self, text, job, incomplete=False):
        return api.FrontendHandler._finalize_content(None, text, job, True, incomplete)

    def test_json_tool_spellings_stream_as_text_at_every_split(self):
        text = json.dumps({"answer": 42, "marker": CALL})
        job = self.job()
        content, calls = self.finalize(text, job)
        self.assertEqual((content, calls), (text, []))
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                projector = model_output.StreamingToolCallProjector(
                    job.tool_policy, job.public_id, True
                )
                events = projector.put(text[:split]) + projector.put(text[split:])
                tail = projector.finish(content, calls, False)
                self.assertTrue(all(kind == "content" for kind, _ in events))
                self.assertEqual(
                    "".join(value for _, value in events) + "".join(tail), text
                )

    def test_partial_json_is_preserved_and_partial_tool_is_not_completed(self):
        job = self.job()
        partial_json = '{"answer":42,"marker":"<tool_call>'
        self.assertEqual(self.finalize(partial_json, job, True), (partial_json, []))
        self.assertEqual(self.finalize(" \n" + CALL[:20], job, True), (" \n", []))
        for text in (partial_json, " \n" + CALL[:20], " "):
            with self.subTest(text=text):
                canonical, calls = self.finalize(text, job, True)
                projector = model_output.StreamingToolCallProjector(
                    job.tool_policy, job.public_id, True
                )
                events = []
                for char in text:
                    events.extend(projector.put(char))
                tail = projector.finish(canonical, calls, True)
                content = "".join(value for kind, value in events if kind == "content")
                self.assertEqual(content + "".join(tail), canonical)

    def test_a_cut_after_a_call_reports_and_streams_no_text(self):
        # Beside an output schema only whitespace surrounds calls.
        job = self.job()
        text = " \n" + CALL + "\n" + OTHER_CALL + "\n"
        for end in range(len(" \n" + CALL), len(text) + 1):
            cut = text[:end]
            with self.subTest(cut=cut):
                content, calls = self.finalize(cut, job, True)
                self.assertEqual(content, "")
                self.assertEqual(calls[0]["function"]["name"], "lookup")
                projector = model_output.StreamingToolCallProjector(
                    job.tool_policy, job.public_id, True
                )
                events = []
                for char in cut:
                    events.extend(projector.put(char))
                tail = projector.finish(content, calls, True)
                self.assertFalse([value for kind, value in events if kind == "content"])
                self.assertEqual(tail, [])

    def test_finalization_enforces_required_parallel_and_output_schema(self):
        for text, job in (
            (ANSWER, self.job("required")),
            (CALL + OTHER_CALL, self.job(parallel=False)),
            (CALL + ANSWER, self.job()),
            ('{"answer":41}', self.job()),
        ):
            with self.subTest(text=text):
                with self.assertRaises(api.APIError):
                    self.finalize(text, job)
        content, calls = self.finalize(CALL + "\n" + OTHER_CALL, self.job())
        self.assertFalse(content.strip())
        self.assertEqual(
            [call["function"]["name"] for call in calls], ["lookup", "finish"]
        )


if __name__ == "__main__":
    unittest.main()
