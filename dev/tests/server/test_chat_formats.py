"""Each family's chat format: Flint's GLM argument tags and the reasoning its
model opens itself, read, constrained and rendered as Qwen's XML and Qwen's
prompted reasoning are, and the Qwen format unchanged beside it."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from llguidance import LLMatcher
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

from dev.tests import server_fixtures as fixtures
from dev.tests.tool_output import argument_grammar, project, put, tool_policy
from install import families
from server import backend as backend_api
from server import constraints, tool_schema
from server import output as model_output
from server import protocol as wire
from server import server as api
from server.chat_format import FLINT, GLM_ARGS, QWEN, QWEN_XML, ToolCallSyntax
from server.chat_templates import NATIVE, ChatTemplates
from server.tokenization import PromptTokenizer

TEMPLATE = (
    Path(__file__).resolve().parents[1] / "fixtures/chat_templates/flint.jinja"
).read_text()
# Flint's control tokens, the special ones, and the tags its template writes
# in text, added tokens that are not special.
FLINT_SPECIAL = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|developer|>",
)
FLINT_TAGS = (
    "<think>",
    "</think>",
    "<tools>",
    "</tools>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
)
SCHEMAS = {
    "weather": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "units": {"enum": ["metric", "imperial"]},
            "detail": {"type": "object"},
        },
        "required": ["city"],
    },
    "time": {"type": "object", "properties": {}},
}


def flint_tokenizer():
    """A byte-level BPE tokenizer with Flint's added tokens, each kind as
    Flint's tokenizer.json adds it, and its chat template."""
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Paris weather in metric units\n"],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    backend.add_special_tokens(
        [AddedToken(text, normalized=False) for text in FLINT_SPECIAL]
    )
    backend.add_tokens([AddedToken(text, normalized=False) for text in FLINT_TAGS])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|end_of_text|>"
    )
    tokenizer.chat_template = TEMPLATE
    return tokenizer


def flint_format(tokenizer):
    """Flint's chat format over `tokenizer`'s token IDs."""
    ids = tokenizer.convert_tokens_to_ids
    return replace(
        FLINT,
        vocabulary=len(tokenizer),
        stop_tokens=tuple((text, ids(text)) for text, _ in FLINT.stop_tokens),
        think_end_id=ids("</think>"),
        think_start_id=ids("<think>"),
        call_open_id=ids("<tool_call>"),
    )


def qwen_call(name, arguments):
    """A call as Qwen's template writes it: each value a string as its text,
    any other as JSON."""
    return (
        f"<tool_call>\n<function={name}>\n"
        + "".join(
            f"<parameter={key}>\n{_text(value)}\n</parameter>\n"
            for key, value in arguments.items()
        )
        + "</function>\n</tool_call>"
    )


def flint_call(name, arguments):
    """The call as Flint's template writes it."""
    return (
        f"<tool_call>{name}"
        + "".join(
            f"<arg_key>{key}</arg_key><arg_value>{_text(value)}</arg_value>"
            for key, value in arguments.items()
        )
        + "</tool_call>"
    )


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def split(text, size, chat_format, thinking=False, tool_calls=True):
    """The reasoning and the answer ReasoningSplitter reads from `text` in
    pieces of `size` characters, and whether it ends in its reasoning."""
    splitter = model_output.ReasoningSplitter(thinking, tool_calls, chat_format)
    parts = [
        part
        for offset in range(0, len(text), size)
        for part in splitter.put(text[offset : offset + size])
    ]
    parts += splitter.finish()
    return (
        "".join(value for kind, value in parts if kind == "reasoning_content"),
        "".join(value for kind, value in parts if kind == "content"),
        splitter.reasoning,
    )


class GlmArgumentReadingTests(unittest.TestCase):
    """Flint's calls read as Qwen's read: the same content and calls from the
    same calls in either layout, however the output is cut into pieces."""

    def assert_reads_alike(self, *parts, incomplete=False):
        policy = tool_policy(SCHEMAS)
        qwen = "".join(
            qwen_call(*part) if isinstance(part, tuple) else part for part in parts
        )
        flint = "".join(
            flint_call(*part) if isinstance(part, tuple) else part for part in parts
        )
        expected = project(qwen, policy, "x", incomplete)[:2]
        for size in (None, *range(1, 12)):
            with self.subTest(size=size):
                content, calls, _ = project(
                    flint, policy, "x", incomplete, size, syntax=GLM_ARGS
                )
                self.assertEqual((content, calls), expected)
        return expected

    def test_values_convert_by_the_types_their_tool_declares(self):
        content, calls = self.assert_reads_alike(
            "Let me check. ",
            (
                "weather",
                {"city": "São Paulo", "days": 3, "units": "metric", "detail": {}},
            ),
        )
        self.assertEqual(content, "Let me check. ")
        self.assertEqual(
            [
                (call["function"]["name"], call["function"]["arguments"])
                for call in calls
            ],
            [
                (
                    "weather",
                    '{"city":"São Paulo","days":3,"units":"metric","detail":{}}',
                )
            ],
        )

    def test_parallel_calls_text_between_and_after_and_calls_without_arguments(self):
        _, calls = self.assert_reads_alike(
            ("weather", {"city": "Paris"}),
            "\n",
            ("time", {}),
            " Then I answer.",
        )
        self.assertEqual(
            [call["function"]["arguments"] for call in calls],
            ['{"city":"Paris"}', "{}"],
        )

    def test_a_value_holds_any_tag_but_its_close_before_the_next_tag(self):
        # A file an agent writes may hold the tags; only the value's close
        # with an argument's opening or the call's close after it ends it.
        value = "a <arg_key>b</arg_key> </arg_value> c </tool_call> d\n"
        _, calls = self.assert_reads_alike(("weather", {"city": value}))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["city"], value)

    def test_a_cut_call_keeps_the_arguments_it_has(self):
        text = flint_call("weather", {"city": "Paris", "days": 3})
        # From the end of the function's name on.
        for cut in range(len("<tool_call>weather<"), len(text)):
            with self.subTest(cut=cut):
                _, calls, _ = project(
                    text[:cut],
                    tool_policy(SCHEMAS),
                    incomplete=True,
                    syntax=GLM_ARGS,
                )
                self.assertEqual(
                    [call["function"]["name"] for call in calls], ["weather"]
                )

    def test_names_lose_their_space_and_a_repeated_parameter_keeps_its_first(self):
        text = (
            "<tool_call> weather <arg_key> city </arg_key><arg_value>Paris"
            "</arg_value><arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>"
        )
        _, calls, _ = project(text, tool_policy(SCHEMAS), syntax=GLM_ARGS)
        self.assertEqual(
            (calls[0]["function"]["name"], calls[0]["function"]["arguments"]),
            ("weather", '{"city":"Paris"}'),
        )

    def test_a_value_open_split_across_pieces_is_not_value_text(self):
        policy = tool_policy(SCHEMAS)
        projector = model_output.StreamingToolCallProjector(
            policy, "x", syntax=GLM_ARGS
        )
        events = []
        for piece in (
            "<tool_call>weather<arg_key>city</arg_key><",
            "arg_",
            "value>Pa",
            "ris</arg_value></tool_call>",
        ):
            events += projector.put(piece)
        _, calls, _ = projector.finish(False)
        self.assertEqual(calls[0]["function"]["arguments"], '{"city":"Paris"}')
        self.assertEqual(
            "".join(
                value["function"].get("arguments", "")
                for kind, value in events
                if kind == "tool"
            ),
            '{"city":"Paris"}',
        )

    def test_without_tools_a_call_is_text(self):
        text = "Here: " + flint_call("weather", {"city": "Paris"}) + "</think>"
        projector = model_output.StreamingToolCallProjector(None, "x")
        put(projector, text)
        content, calls, _ = projector.finish(False)
        self.assertEqual((content, calls), (text.removesuffix("</think>"), []))


class ModelOpenedReasoningTests(unittest.TestCase):
    """Flint opens its reasoning itself: an output that starts with <think>
    reasons to </think>, and any other is all answer."""

    def test_reasoning_is_the_think_block_the_output_starts_with(self):
        call = flint_call("weather", {"city": "Paris"})
        for text, expected in (
            (
                "<think>Plan <tool_call> it.</think>\n\nAnswer",
                ("Plan <tool_call> it.", "Answer"),
            ),
            ("<think>Plan.</think>" + call, ("Plan.", call)),
            ("Answer <think>no</think>", ("", "Answer <think>no</think>")),
            (" <think>late</think>", ("", " <think>late</think>")),
            ("<thin", ("", "<thin")),
            ("", ("", "")),
        ):
            for size in range(1, len(text) + 2):
                with self.subTest(text=text, size=size):
                    self.assertEqual(split(text, size, FLINT)[:2], expected)

    def test_output_cut_inside_the_reasoning_ends_in_it(self):
        self.assertEqual(split("<think>Still", 3, FLINT), ("Still", "", True))

    def test_qwen_reasoning_is_unchanged(self):
        # Qwen's generation prompt opens the reasoning, and a call's opening
        # ends it too; a <think> its output starts with is answer text.
        call = qwen_call("weather", {"city": "Paris"})
        self.assertEqual(
            split("Plan." + call, 4, QWEN, thinking=True)[:2], ("Plan.", call)
        )
        self.assertEqual(
            split("<think>x</think>y", 4, QWEN)[:2], ("", "<think>x</think>y")
        )
        # A text completion's output, which has no chat format.
        self.assertEqual(
            split("<think>x", 4, None, tool_calls=False)[:2], ("", "<think>x")
        )

    def test_reasoning_tokens_are_those_inside_the_model_s_think_block(self):
        texts = {1: "a", 2: " b", 6: "<think>", 7: "</think>", 10: "<tool_call>"}
        tokenizer = SimpleNamespace(
            backend_tokenizer=None,
            decode=lambda ids: "".join(texts[token] for token in ids),
        )
        chat_format = replace(FLINT, think_start_id=6, think_end_id=7, call_open_id=10)
        for token_ids, expected in (
            ([6, 1, 2, 7, 1], 2),
            ([6, 1, 10, 2, 7], 3),
            ([6, 1, 2], 2),
            ([1, 6, 2, 7], 0),
            ([], 0),
        ):
            with self.subTest(token_ids=token_ids):
                streamer = backend_api.CallbackStreamer(tokenizer, lambda *_: None)
                streamer.token_ids = token_ids
                self.assertEqual(
                    streamer.count_reasoning_tokens(chat_format, False, True), expected
                )
                self.assertEqual(streamer.count_reasoning_tokens(None, False, True), 0)


class GlmGrammarTests(unittest.TestCase):
    """Flint's constrained output: its optional reasoning, then calls in its
    tags, compiled over a tokenizer that adds the tags as Flint's does."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = flint_tokenizer()
        cls.chat_format = flint_format(cls.tokenizer)
        cls.factory = constraints.ConstraintFactory(cls.tokenizer, cls.chat_format)

    def policy(self, choice="auto", parallel=True, strict=False):
        tools = [
            {
                "type": "function",
                "function": {"name": name, "parameters": schema, "strict": strict},
            }
            for name, schema in SCHEMAS.items()
        ]
        return tool_schema.normalize_tools(tools, choice, parallel)[1]

    def accepts(self, grammar, text):
        constraint = self.factory.create(grammar)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        matcher = constraint.matcher
        return matcher.consume_tokens(tokens) and matcher.is_accepting()

    def test_calls_follow_an_optional_think_block_the_output_starts_with(self):
        grammar = tool_schema.tool_grammar(
            self.policy("required"), False, chat_format=self.chat_format
        )
        call = flint_call("weather", {"city": "Paris", "days": 3})
        for text, accepted in (
            (call, True),
            ("<think>Paris, then.</think>" + call, True),
            ("<think>Paris</think>\n" + call + "\n" + flint_call("time", {}), True),
            ("Sure. " + call, False),
            (call + "<think>x</think>", False),
            ("<think>no call</think>", False),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.accepts(grammar, text), accepted)

    def test_strict_arguments_are_generated_in_the_tags(self):
        grammar = tool_schema.tool_grammar(
            self.policy("required", strict=True), False, chat_format=self.chat_format
        )
        for arguments, accepted in (
            ({"city": "Paris", "days": 3, "units": "metric"}, True),
            ({"city": "Paris"}, True),
            ({"city": "Paris", "days": "three"}, False),
            ({"city": "Paris", "units": "kelvin"}, False),
            ({"days": 3}, False),
            ({"city": "Paris", "other": 1}, False),
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    self.accepts(grammar, flint_call("weather", arguments)), accepted
                )

    def test_what_the_grammar_writes_reads_back_as_written(self):
        arguments = {"city": "New York", "days": 2, "units": "imperial"}
        text = "<think>Check.</think>" + flint_call("weather", arguments)
        grammar = tool_schema.tool_grammar(
            self.policy("required", strict=True), False, chat_format=self.chat_format
        )
        self.assertTrue(self.accepts(grammar, text))
        reasoning, content, _ = split(text, 5, self.chat_format)
        _, calls, _ = project(content, self.policy(), syntax=GLM_ARGS)
        self.assertEqual(reasoning, "Check.")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), arguments)

    def test_a_response_schema_follows_the_optional_reasoning(self):
        grammar = tool_schema.json_grammar(
            {"type": "object", "properties": {"a": {"type": "integer"}}},
            False,
            chat_format=self.chat_format,
        )
        for text, accepted in (
            ('{"a": 1}', True),
            ('<think>One.</think>{"a": 1}', True),
            ('{"a": 1}<think>x</think>', False),
            ('<think>One.</think>{"a": "1"}', False),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.accepts(grammar, text), accepted)

    def test_free_arguments_take_any_name_and_text(self):
        grammar = tool_schema._free_arguments(GLM_ARGS)
        self.assertFalse(LLMatcher.validate_grammar(grammar, self.factory.tokenizer))
        matcher = LLMatcher(self.factory.tokenizer, grammar)
        text = (
            "<arg_key>a b</arg_key><arg_value>any\ntext</arg_value>"
            "<arg_key>c</arg_key><arg_value></arg_value>"
        )
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.assertTrue(matcher.consume_tokens(tokens) and matcher.is_accepting())

    def test_a_strict_tool_s_argument_grammar_writes_flint_s_tags_as_tokens(self):
        # The grammar compiler takes the tags, added tokens, apart from text:
        # a grammar writes them as tokens, and raw text up to the close's.
        grammar = argument_grammar(SCHEMAS["weather"], GLM_ARGS)
        self.assertIn('<arg_key> "city" </arg_key> <arg_value>', grammar)
        self.assertIn("parameter_0_value: RAW_VALUE </arg_value>\n", grammar)
        self.assertIn(
            "RAW_VALUE: /(?s:.*)/ & ~/(?s:.*)<\\/arg_value>(?s:.*)/\n", grammar
        )
        self.assertNotIn('"<arg', grammar)

    def test_the_constraint_ends_at_any_of_flint_s_stop_tokens(self):
        grammar = tool_schema.tool_grammar(
            self.policy(), False, chat_format=self.chat_format
        )
        for stop, token in self.chat_format.stop_tokens:
            with self.subTest(stop=stop):
                constraint = self.factory.create(grammar)
                self.assertEqual(constraint.eos_tokens, self.chat_format.stop_ids)
                constraint.commit(
                    self.tokenizer.encode("Done.", add_special_tokens=False)
                )
                constraint.commit([token])
                constraint.finish()


class QwenGrammarTests(unittest.TestCase):
    def test_qwen_grammars_are_unchanged(self):
        # Qwen's grammars as they were written before the formats were
        # values: a thinking prefix closed by the think-close token, calls in
        # Qwen's XML, the arguments of a tool that is not strict free.
        policy = tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "f"}}], "required", False
        )[1]
        grammars = json.loads(tool_schema.tool_grammar(policy, True, chat_format=QWEN))
        main, free = (grammar["lark_grammar"] for grammar in grammars["grammars"])
        self.assertEqual(
            main,
            "%llguidance {}\n"
            "start: think WS (call_0) WS\n"
            "think: TEXT <[248069]>\n"
            'call_0: <tool_call> "\\n<function=f>\\n" @free_arguments "</function>\\n" '
            "</tool_call>\n" + tool_schema.WHITESPACE_RULE + "\n"
            r"TEXT: /(?s:.*)/ & ~/(?s:.*)(<tool_call>|<\/think>)(?s:.*)/" + "\n",
        )
        self.assertTrue(
            free.startswith(
                '%llguidance {}\nstart: parameter*\nparameter: "<parameter=" NAME '
                '">\\n" value\nvalue[suffix="\\n</parameter>\\n"]: /(?s:.*)/\n'
            )
        )
        self.assertEqual(
            tool_schema.json_grammar({"type": "integer"}, False, chat_format=QWEN),
            "%llguidance {}\nstart: WS %json "
            + json.dumps(
                tool_schema._grammar_compatible_schema({"type": "integer"}),
                separators=(",", ":"),
            )
            + "\n"
            + tool_schema.WHITESPACE_RULE
            + "\n",
        )

    def test_each_syntax_is_a_value(self):
        for syntax, opening in (
            (QWEN_XML, "<tool_call>\n<function=f>\n"),
            (GLM_ARGS, "<tool_call>f"),
        ):
            with self.subTest(opening=opening):
                self.assertIsInstance(syntax, ToolCallSyntax)
                self.assertEqual("<tool_call>" + syntax.function_opening("f"), opening)
        self.assertEqual(
            QWEN_XML.argument_tags, ("<parameter=", "</function>", "</tool_call>")
        )
        self.assertEqual(GLM_ARGS.argument_tags, ("<arg_key>", "</tool_call>"))


class FlintTokenizerContractTests(unittest.TestCase):
    def test_flint_s_tokens_are_checked_against_its_format(self):
        tokenizer = flint_tokenizer()
        chat_format = flint_format(tokenizer)
        constraints.validate_tokenizer(tokenizer, chat_format)
        for field, text in (
            ("think_start_id", "<think>"),
            ("think_end_id", "</think>"),
            ("call_open_id", "<tool_call>"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    fixtures.engine_runtime.EngineUnhealthy,
                    f"tokenizer must encode '{text}' as native token",
                ):
                    constraints.validate_tokenizer(
                        tokenizer,
                        replace(
                            chat_format, **{field: getattr(chat_format, field) + 1}
                        ),
                    )
        with self.assertRaisesRegex(
            fixtures.engine_runtime.EngineUnhealthy, "vocabulary does not fit"
        ):
            constraints.validate_tokenizer(
                tokenizer, replace(chat_format, vocabulary=len(tokenizer) - 1)
            )

    def test_the_real_ids_are_flint_s(self):
        self.assertEqual(
            (
                FLINT.vocabulary,
                FLINT.stop_ids,
                FLINT.think_start_id,
                FLINT.think_end_id,
            ),
            (150144, (150001, 150003, 150005, 150018), 150006, 150007),
        )
        self.assertEqual((FLINT.call_open_id, FLINT.tool_calls), (150010, GLM_ARGS))
        self.assertFalse(FLINT.calls_end_reasoning)
        self.assertTrue(QWEN.calls_end_reasoning)


class FlintPromptTests(unittest.TestCase):
    """Flint's template renders as shipped: its roles, its tools block, the
    reasoning and calls of a history, and a generation prompt that leaves
    reasoning to the model, whatever the request's effort."""

    def frontend(self, **options):
        tokenizer = flint_tokenizer()
        return fixtures.make_frontend(
            tokenizer,
            None,
            "test-model",
            4096,
            10,
            2,
            vision=False,
            chat_format=flint_format(tokenizer),
            **options,
        )

    def test_the_probe_finds_later_system_messages_native_and_one_prompt(self):
        templates = ChatTemplates(flint_tokenizer())
        template = templates.select(None)
        self.assertEqual(template.later_system, NATIVE)
        self.assertEqual(
            {text for text, _ in template.generation_prompts.values()},
            {"<|assistant|>"},
        )

    def test_a_tool_conversation_renders_in_flint_s_markup(self):
        body = {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Look it up.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city": "Paris", "days": 2}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": SCHEMAS["weather"]},
                }
            ],
        }
        prompt = self.frontend().apply_template(body, deadline=fixtures.FOREVER)
        self.assertTrue(prompt.startswith("<|system|>Be brief.<|system|>\n# Tools\n"))
        self.assertIn('<tools>{"name": "weather", "parameters": {', prompt)
        self.assertTrue(
            prompt.endswith(
                "<|user|>Weather in Paris?<|assistant|><think>Look it up.</think>"
                + flint_call("weather", {"city": "Paris", "days": 2})
                + "<|observation|><tool_response>Sunny</tool_response><|assistant|>"
            ),
            prompt,
        )

    def test_reasoning_options_leave_the_prompt_to_the_model(self):
        frontend = self.frontend()
        prompts = set()
        for options in (
            {},
            {"reasoning_effort": "none"},
            {"reasoning_effort": "high"},
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"thinking_token_budget": 3},
        ):
            with self.subTest(**options):
                body = fixtures.chat_body(**options)
                job = frontend.prepare(body, deadline=fixtures.FOREVER)
                self.assertFalse(job.thinking)
                self.assertIsNone(job.thinking_token_budget)
                self.assertIs(job.chat_format, frontend.chat_format)
                prompts.add(frontend.apply_template(body, deadline=fixtures.FOREVER))
        self.assertEqual(prompts, {"<|user|>hello<|assistant|>"})

    def test_scores_are_read_where_the_model_answers_without_reasoning(self):
        # Its effort leaves Flint's model reasoning, which it would open where
        # the generation prompt ends: judgment and System One slots are read
        # after an empty think block, where its answer starts.
        frontend = self.frontend()
        judgment, _ = frontend.prepare_judgment(
            fixtures.judgment_body(),
            deadline=fixtures.FOREVER,
            disconnected=fixtures.stays_connected,
        )
        systemone = frontend.prepare_systemone(
            {"model": "test-model", "state": {}, "questions": {"q": {"type": "noul"}}},
            deadline=fixtures.FOREVER,
            disconnected=fixtures.stays_connected,
        )
        ids = frontend.tokenizer.convert_tokens_to_ids
        for job in (judgment, systemone[0][2]):
            self.assertEqual(
                list(job.prompt_tokens[-3:]),
                [ids("<|assistant|>"), ids("<think>"), ids("</think>")],
            )
            self.assertEqual(len(job.score_tokens), 2)

    def test_prompt_tokens_are_reused_after_the_last_assistant_turn(self):
        # Splits before the byte-level BPE, as Flint's tokenizer has three,
        # and the GGUF tokenizer's byte-level post-processor.
        tokenizer = flint_tokenizer()
        backend = tokenizer.backend_tokenizer
        backend.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(r"\d{1,3}"), behavior="isolated"),
                pre_tokenizers.Split(Regex(" "), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        backend.post_processor = processors.ByteLevel(trim_offsets=False)
        cache = PromptTokenizer(tokenizer, FLINT.message_boundary)
        self.assertTrue(cache.enabled)
        first = "<|user|>" + "Paris 12345 " * 400 + "<|assistant|>"
        later = first + "Sunny.<|user|>And Rome?<|assistant|>"
        for text in (first, later):
            self.assertEqual(
                cache.encode(text),
                tokenizer(text, add_special_tokens=False)["input_ids"],
            )
        self.assertEqual(cache.stats()["hits"], 1)


class FlintServingTests(fixtures.HarnessTestCase):
    """Flint's output through the server: the reasoning it opens and its
    calls, streamed or not, read from the tokens the engine returns."""

    FRAGMENTS = {
        40: "<think>",
        41: "Check the weather.",
        42: "</think>",
        43: "Let me look.",
        44: "<tool_call>wea",
        45: "ther<arg_key>ci",
        46: "ty</arg_key><arg_val",
        47: "ue>Par",
        48: "is</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
    }
    # Flint's format over the fake vocabulary's IDs.
    CHAT_FORMAT = replace(FLINT, think_start_id=40, think_end_id=42, call_open_id=44)

    def harness(self, *plans, **options):
        # Prompts render with Flint's template; the engine's tokens decode
        # to the fragments.
        tokenizer = fixtures.TemplateTokenizer(TEMPLATE)
        tokenizer.fragments = self.FRAGMENTS
        tokenizer.backend_tokenizer = fixtures.byte_backend(self.FRAGMENTS)
        return super().harness(
            fixtures.FakeRuntime(*plans),
            tokenizer=tokenizer,
            chat_format=self.CHAT_FORMAT,
            **options,
        )

    def body(self, **extra):
        return fixtures.chat_body(
            tools=[
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": SCHEMAS["weather"]},
                }
            ],
            **extra,
        )

    def test_reasoning_text_and_a_call_read_from_the_output(self):
        output = [[40], [41], [42], [43], [44], [45], [46], [47], [48]]
        harness = self.harness(fixtures.Plan(output), fixtures.Plan(output))
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body()
        )
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        message = response["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "Check the weather.")
        self.assertEqual(message["content"], "Let me look.")
        self.assertEqual(
            [
                (call["function"]["name"], call["function"]["arguments"])
                for call in message["tool_calls"]
            ],
            [("weather", '{"city":"Paris","days":3}')],
        )
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(
            response["usage"]["completion_tokens_details"]["reasoning_tokens"], 1
        )
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(stream=True)
        )
        self.assertEqual(status, 200, payload)
        deltas = [
            event["choices"][0]["delta"]
            for event in fixtures.stream_events(payload)
            if event.get("choices")
        ]
        self.assertEqual(
            "".join(delta.get("reasoning_content") or "" for delta in deltas),
            "Check the weather.",
        )
        self.assertEqual(
            "".join(delta.get("content") or "" for delta in deltas), "Let me look."
        )
        self.assertEqual(
            "".join(
                call["function"].get("arguments", "")
                for delta in deltas
                for call in delta.get("tool_calls") or ()
            ),
            '{"city":"Paris","days":3}',
        )

    def test_an_answer_without_reasoning_is_all_content(self):
        harness = self.harness(fixtures.Plan([[43]]))
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body()
        )
        self.assertEqual(status, 200, payload)
        message = json.loads(payload)["choices"][0]["message"]
        self.assertEqual(message["content"], "Let me look.")
        self.assertNotIn("reasoning_content", message)

    def test_a_required_call_is_constrained_in_flint_s_tags(self):
        factory = fixtures.FakeConstraintFactory()
        harness = self.harness(
            fixtures.Plan([[44], [45], [46], [47], [48]]), constraint_factory=factory
        )
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(tool_choice="required")
        )
        self.assertEqual(status, 200, payload)
        (grammar,) = factory.grammars
        main = json.loads(grammar)["grammars"][0]["lark_grammar"]
        self.assertIn("start: think? WS (call_0) (TEXT (call_0))* TEXT\n", main)
        self.assertIn("think: <[40]> TEXT <[42]>\n", main)
        self.assertIn(
            'call_0: <tool_call> "weather" @free_arguments </tool_call>\n', main
        )


class ServedFamilyTests(unittest.TestCase):
    """main() reads its output in the chat format of the family the model was
    installed as, which its assembly records, and serves no other."""

    def main(self, family):
        runtime = mock.Mock()
        runtime.readiness = wire.ReadyEvent(4, 131072, False)
        tokenizer = object()
        with (
            mock.patch.object(api, "parse_args", return_value=fixtures.main_args()),
            fixtures.unprepared_hold(family),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(api, "validate_tokenizer") as validate,
            mock.patch.object(api, "ChatTemplates"),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ) as runtime_type,
            mock.patch.object(api, "NativeBackend"),
            mock.patch.object(api, "ConstraintFactory") as factory_type,
            mock.patch.object(api, "Frontend") as app_type,
            mock.patch.object(
                api, "FrontendServer", return_value=mock.Mock(server_port=8000)
            ),
            mock.patch.object(api.signal, "signal"),
            mock.patch.object(api, "print_status") as status,
        ):
            try:
                api.main()
            except SystemExit:
                pass
        return SimpleNamespace(
            tokenizer=tokenizer,
            validate=validate,
            factory_type=factory_type,
            app_type=app_type,
            runtime_type=runtime_type,
            status=status,
        )

    def test_the_family_s_format_reads_and_constrains_its_output(self):
        for family in families.FAMILIES:
            with self.subTest(family=family.name):
                started = self.main(family.name)
                started.validate.assert_called_once_with(
                    started.tokenizer, family.chat_format
                )
                started.factory_type.assert_called_once_with(
                    started.tokenizer, family.chat_format
                )
                self.assertIs(
                    started.app_type.call_args.kwargs["chat_format"], family.chat_format
                )

    def test_a_model_of_no_family_this_release_serves_is_refused(self):
        started = self.main("Unknown-1B")
        started.runtime_type.assert_not_called()
        started.validate.assert_not_called()
        self.assertIn(
            "is not prepared as a model of a family this release serves",
            started.status.call_args_list[-1].args[0],
        )


if __name__ == "__main__":
    unittest.main()
