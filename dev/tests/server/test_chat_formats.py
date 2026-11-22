"""Each family's chat format: Flint's GLM argument tags and the reasoning its
model opens itself, read, constrained and rendered as Qwen's XML and Qwen's
prompted reasoning are, Inkling 2's blocks, which its model opens itself in
special tokens, and the Qwen format unchanged beside them."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import transformers
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
from dev.tests.tool_output import (
    argument_grammar,
    project,
    put,
    streamed_arguments,
    tool_policy,
)
from install import families
from server import backend as backend_api
from server import constraints, tool_schema
from server import output as model_output
from server import protocol as wire
from server import server as api
from server.chat_format import (
    FLINT,
    GLM_ARGS,
    INKLING2,
    INKLING2_JSON,
    QWEN,
    QWEN_XML,
    ToolCallSyntax,
)
from server.chat_templates import NATIVE, ChatTemplateError, ChatTemplates
from server.dialects import scoring
from server.errors import APIError
from server.tokenization import PromptTokenizer

TEMPLATE = (
    Path(__file__).resolve().parents[1] / "fixtures/chat_templates/flint.jinja"
).read_text()
INKLING2_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "fixtures/chat_templates/inkling2.jinja"
).read_text()
# Inkling 2's special tokens its template and tokenizer_config name, every
# one a control token of its GGUF, and its tags by their role here.
INKLING2_SPECIAL = (
    "<|endoftext|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_image|>",
    "<|content_model_end_sampling|>",
    "<|content_thinking|>",
    "<|end_message|>",
    "<|content_tool_error|>",
    "<|content_xml|>",
    "<|begin_of_text|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|unused_200054|>",
)
MODEL, THINKING, TEXT, END = (
    "<|message_model|>",
    "<|content_thinking|>",
    "<|content_text|>",
    "<|end_message|>",
)
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
        think_end_id=ids("</think>"),
        think_start_ids=(ids("<think>"),),
        call_open_id=ids("<tool_call>"),
    )


def block(kind, text):
    """A block of Inkling 2's output: the model's message of `kind`."""
    return MODEL + kind + text + END


def inkling2_call(name, arguments):
    """A call as Inkling 2's template writes it: the model's message named for
    the function, then the call's JSON object, compact, its keys sorted."""
    body = json.dumps(
        {"name": name, "args": dict(sorted(arguments.items()))},
        separators=(",", ":"),
    )
    return MODEL + name + "<|content_invoke_tool_json|>" + body + END


def inkling2_tokenizer():
    """A byte-level BPE tokenizer with Inkling 2's special tokens, as its
    GGUF's tokenizer derives them, and its chat template."""
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Paris weather in metric units\nThinking effort level: 0.9\n"],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    backend.add_special_tokens(
        [AddedToken(text, normalized=False) for text in INKLING2_SPECIAL]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|content_model_end_sampling|>"
    )
    tokenizer.chat_template = INKLING2_TEMPLATE
    return tokenizer


def inkling2_format(tokenizer):
    """Inkling 2's chat format over `tokenizer`'s token IDs."""
    ids = tokenizer.convert_tokens_to_ids
    return replace(
        INKLING2,
        think_end_id=ids(END),
        think_start_ids=(ids(MODEL), ids(THINKING)),
        call_open_id=ids(MODEL),
    )


def flint_ready(tokenizer):
    """What an engine serving Flint over `tokenizer`'s token IDs announces:
    its vocabulary and the IDs of Flint's stop texts."""
    ids = tokenizer.convert_tokens_to_ids
    return wire.ReadyEvent(
        4, 4096, False, len(tokenizer), 9, tuple(map(ids, FLINT.stop_tokens))
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
        chat_format = replace(
            FLINT, think_start_ids=(6,), think_end_id=7, call_open_id=10
        )
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
        cls.ready = flint_ready(cls.tokenizer)
        cls.factory = constraints.ConstraintFactory(
            constraints.ConstraintFactory.table(
                cls.tokenizer, cls.chat_format, cls.ready.vocabulary_size
            ),
            cls.ready,
        )

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
        for stop, token in zip(self.chat_format.stop_tokens, self.ready.stop_token_ids):
            with self.subTest(stop=stop):
                constraint = self.factory.create(grammar)
                self.assertEqual(constraint.eos_tokens, self.ready.stop_token_ids)
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
            "start: think (call_0) WS\n"
            "think: TEXT <[248069]> WS\n"
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
            "%llguidance {}\nstart: %json "
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
        for field, text, changed in (
            ("think_start_ids", "<think>", (chat_format.think_start_ids[0] + 1,)),
            ("think_end_id", "</think>", chat_format.think_end_id + 1),
            ("call_open_id", "<tool_call>", chat_format.call_open_id + 1),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    constraints.TokenizerContractError,
                    f"tokenizer must encode '{text}' as native token",
                ):
                    constraints.validate_tokenizer(
                        tokenizer, replace(chat_format, **{field: changed})
                    )
        with self.assertRaisesRegex(
            constraints.TokenizerContractError, "stop text '<\\|user\\|>x'"
        ):
            constraints.validate_tokenizer(
                tokenizer,
                replace(chat_format, stop_tokens=("<|end_of_text|>", "<|user|>x")),
            )

    def test_flint_s_tokenizer_is_checked_against_the_engines_ready(self):
        """Once the engine is Ready: the tokenizer fits its vocabulary, its
        stop texts encode as exactly the engine's stop tokens, its EOS is one
        of them, and an engine serving images needs the template's one-token
        image placeholder, which Flint's has not."""
        tokenizer = flint_tokenizer()
        chat_format = flint_format(tokenizer)
        ready = flint_ready(tokenizer)
        constraints.check_capabilities(tokenizer, chat_format, ready)
        vision = replace(ready, vision=True)
        for changed, refusal, served_format in (
            (
                replace(ready, vocabulary_size=len(tokenizer) - 1),
                "does not fit",
                chat_format,
            ),
            (
                replace(ready, stop_token_ids=ready.stop_token_ids[1:]),
                "stops at",
                chat_format,
            ),
            (
                replace(ready, stop_token_ids=(*ready.stop_token_ids[:-1], 0)),
                "stops at",
                chat_format,
            ),
            (vision, "no image placeholder", chat_format),
            (
                vision,
                "image placeholder '<\\|user\\|>x'",
                replace(chat_format, image_pad="<|user|>x"),
            ),
        ):
            with self.subTest(refusal=refusal, ready=changed):
                with self.assertRaisesRegex(
                    constraints.TokenizerContractError, refusal
                ):
                    constraints.check_capabilities(tokenizer, served_format, changed)

    def test_the_real_ids_are_flint_s(self):
        self.assertEqual(
            (FLINT.stop_tokens, FLINT.think_start_ids, FLINT.think_end_id),
            (
                ("<|end_of_text|>", "<|user|>", "<|observation|>", "<|developer|>"),
                (150006,),
                150007,
            ),
        )
        self.assertEqual((FLINT.call_open_id, FLINT.call_syntax), (150010, GLM_ARGS))
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

    def test_models_report_no_efforts_for_a_model_that_decides_itself(self):
        self.assertEqual(
            self.frontend(default_reasoning_effort="high").reasoning,
            {"thinks": None, "efforts": [], "default": None},
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

    def test_tokens_are_counted_without_an_image_placeholder(self):
        """Flint's template has no image placeholder and its engine serves no
        images: a count is the prompt's tokens, as generation prepares them,
        and an image is refused as every endpoint refuses it."""
        frontend = self.frontend()
        body = fixtures.chat_body()
        job = frontend.prepare(body, deadline=fixtures.FOREVER)
        self.assertEqual(
            frontend.count_tokens(body, deadline=fixtures.FOREVER),
            len(job.prompt_tokens),
        )
        image = {"type": "image_url", "image_url": {"url": fixtures.png_data_url()}}
        body = fixtures.chat_body(messages=[{"role": "user", "content": [image]}])
        refusals = set()
        for request in (frontend.prepare, frontend.count_tokens):
            with self.subTest(request=request.__name__):
                with self.assertRaises(APIError) as raised:
                    request(body, deadline=fixtures.FOREVER)
                self.assertEqual(raised.exception.status, 400)
                refusals.add(str(raised.exception))
        self.assertEqual(len(refusals), 1, refusals)

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
        judgment, _ = scoring.prepare_judgment(
            frontend,
            fixtures.judgment_body(),
            deadline=fixtures.FOREVER,
            disconnected=fixtures.stays_connected,
        )
        systemone = scoring.prepare_systemone(
            frontend,
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
        cache = PromptTokenizer(
            tokenizer, FLINT.message_boundary, fixtures.NATIVE_CONTEXT
        )
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
    CHAT_FORMAT = replace(
        FLINT, think_start_ids=(40,), think_end_id=42, call_open_id=44
    )

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
        self.assertIn("start: think? (call_0) (TEXT (call_0))* TEXT\n", main)
        self.assertIn("think: <[40]> TEXT <[42]> WS\n", main)
        self.assertIn(
            'call_0: <tool_call> "weather" @free_arguments </tool_call>\n', main
        )


class InklingReadingTests(unittest.TestCase):
    """Inkling 2's output is blocks its model opens itself: the reasoning in
    <|content_thinking|> blocks, which may come again after text, the answer
    in <|content_text|> blocks, each closed by <|end_message|>."""

    def test_reasoning_and_text_are_read_from_their_blocks(self):
        for text, expected in (
            (block(THINKING, "Plan.") + block(TEXT, "Answer"), ("Plan.", "Answer")),
            (block(TEXT, "Just <b>this</b>."), ("", "Just <b>this</b>.")),
            (
                block(THINKING, "a")
                + block(TEXT, "b")
                + block(THINKING, "c")
                + block(TEXT, "d"),
                ("ac", "bd"),
            ),
            (
                block("<|content_tool_error|>", "No such file."),
                ("", "No such file."),
            ),
            ("", ("", "")),
        ):
            for size in range(1, len(text) + 2):
                with self.subTest(text=text, size=size):
                    self.assertEqual(
                        split(text, size, INKLING2, tool_calls=False)[:2], expected
                    )

    def test_reasoning_after_text_is_its_own_part_in_order(self):
        splitter = model_output.ReasoningSplitter(False, False, INKLING2)
        text = block(TEXT, "b") + block(THINKING, "c") + block(TEXT, "d")
        parts = splitter.put(text) + splitter.finish()
        self.assertEqual(
            parts,
            [("content", "b"), ("reasoning_content", "c"), ("content", "d")],
        )

    def test_output_cut_inside_the_reasoning_ends_in_it(self):
        self.assertEqual(
            split(MODEL + THINKING + "Still", 3, INKLING2, tool_calls=False),
            ("Still", "", True),
        )

    def test_reasoning_tokens_are_those_inside_its_blocks(self):
        model, thinking, text, end = 200001, 200008, 200004, 200010
        for token_ids, expected in (
            ([model, thinking, 1, 2, end, model, text, 3, end], 2),
            ([model, text, 3, end, model, thinking, 1, end, model, thinking, 2], 2),
            ([model, text, 1, 2, end], 0),
            ([], 0),
        ):
            with self.subTest(token_ids=token_ids):
                streamer = backend_api.CallbackStreamer(
                    SimpleNamespace(backend_tokenizer=None), lambda *_: None
                )
                streamer.token_ids = token_ids
                self.assertEqual(
                    streamer.count_reasoning_tokens(INKLING2, False, False), expected
                )

    def test_its_tags_decode_as_text_and_its_stop_token_does_not(self):
        tokenizer = inkling2_tokenizer()
        written = block(TEXT, "Paris weather")
        token_ids = tokenizer.encode(written, add_special_tokens=False) + [
            tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
        ]
        for visible, expected in (
            (INKLING2.visible_tags, written),
            ((), "Paris weather"),
        ):
            with self.subTest(visible=visible):
                chunks = []
                streamer = backend_api.CallbackStreamer(
                    tokenizer, chunks.append, visible_tags=visible
                )
                for token_id in token_ids:
                    streamer.put_tokens([token_id])
                streamer.end()
                self.assertEqual("".join(chunks), expected)


class InklingDecodingTests(unittest.TestCase):
    def test_a_tag_after_an_unfinished_character_streams_as_the_output_decodes(
        self,
    ):
        # A token that ends inside a multibyte character (the first byte of
        # "é"), then a visible tag, then text: the stream finishes the run
        # before the tag as the whole output's decode does, its unfinished
        # character as the replacement character, and the two agree.
        tokenizer = inkling2_tokenizer()
        ids = tokenizer.convert_tokens_to_ids
        token_ids = [ids("Ã"), ids(END)] + tokenizer.encode(
            "Paris", add_special_tokens=False
        )
        chunks = []
        streamer = backend_api.CallbackStreamer(
            tokenizer, chunks.append, visible_tags=INKLING2.visible_tags
        )
        for token_id in token_ids:
            streamer.put_tokens([token_id])
        streamer.end()
        self.assertEqual("".join(chunks), "\ufffd" + END + "Paris")
        self.assertEqual("".join(chunks), streamer._decode(token_ids))


class InklingCallReadingTests(unittest.TestCase):
    """Inkling 2's calls, a JSON object in the model's message named for the
    function, read in pieces as they arrive, through the splitter and the
    projector as a request reads them."""

    POLICY = tool_policy(SCHEMAS)

    def read(self, text, size, incomplete=False):
        splitter = model_output.ReasoningSplitter(False, True, INKLING2)
        projector = model_output.StreamingToolCallProjector(
            self.POLICY, "r", syntax=INKLING2_JSON, think_close=END
        )
        reasoning, events = [], []
        pieces = [text[offset : offset + size] for offset in range(0, len(text), size)]
        for kind, value in [p for piece in pieces for p in splitter.put(piece)] + (
            splitter.finish()
        ):
            if kind == "reasoning_content":
                reasoning.append(value)
            else:
                events += projector.put(value)
        content, calls, owed = projector.finish(incomplete)
        return "".join(reasoning), content, calls, events + owed

    def test_calls_text_and_reasoning_read_in_any_pieces(self):
        weather = inkling2_call("weather", {"city": "Paris", "days": 3})
        time = inkling2_call("time", {})
        text = (
            block(THINKING, "Look it up.")
            + block(TEXT, "Let me look.")
            + weather
            + time
        )
        for size in range(1, len(text) + 2):
            with self.subTest(size=size):
                reasoning, content, calls, events = self.read(text, size)
                self.assertEqual((reasoning, content), ("Look it up.", "Let me look."))
                self.assertEqual(
                    [
                        (call["function"]["name"], call["function"]["arguments"])
                        for call in calls
                    ],
                    [("weather", '{"city":"Paris","days":3}'), ("time", "{}")],
                )
                # The arguments streamed are the arguments read.
                self.assertEqual(
                    streamed_arguments(events),
                    '{"city":"Paris","days":3}{}',
                )

    def test_a_cut_call_keeps_the_arguments_it_has(self):
        text = inkling2_call("weather", {"city": "Paris", "days": 3})
        cut = text[: text.index("days")]
        _, _, calls, _ = self.read(cut, 4, incomplete=True)
        self.assertEqual(
            [
                (call["function"]["name"], call["function"]["arguments"])
                for call in calls
            ],
            [("weather", '{"city":"Paris","')],
        )

    def test_an_object_laid_out_otherwise_is_read_whole(self):
        text = (
            MODEL
            + "weather<|content_invoke_tool_json|>"
            + '{ "args": {"city": "Paris"}, "name": "weather" }'
            + END
        )
        for size in (1, 5, len(text)):
            with self.subTest(size=size):
                _, _, calls, _ = self.read(text, size)
                self.assertEqual(
                    json.loads(calls[0]["function"]["arguments"]), {"city": "Paris"}
                )


class InklingGrammarTests(unittest.TestCase):
    """Inkling 2's constrained output: an optional reasoning block, then calls
    in its blocks with their arguments as JSON, or a JSON answer in a text
    block, compiled over a tokenizer of its special tokens."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = inkling2_tokenizer()
        cls.chat_format = inkling2_format(cls.tokenizer)
        ids = cls.tokenizer.convert_tokens_to_ids
        ready = wire.ReadyEvent(
            4,
            4096,
            False,
            len(cls.tokenizer),
            9,
            (ids("<|content_model_end_sampling|>"),),
        )
        cls.factory = constraints.ConstraintFactory(
            constraints.ConstraintFactory.table(
                cls.tokenizer, cls.chat_format, ready.vocabulary_size
            ),
            ready,
        )

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

    def test_calls_follow_an_optional_reasoning_block(self):
        grammar = tool_schema.tool_grammar(
            self.policy("required", strict=True), False, chat_format=self.chat_format
        )
        call = inkling2_call("weather", {"city": "Paris", "days": 3})
        for text, accepted in (
            (call, True),
            (block(THINKING, "Paris, then.") + call, True),
            (call + inkling2_call("time", {}), True),
            # A required call comes first; text may follow it.
            (call + block(TEXT, "Done."), True),
            (block(TEXT, "Sure.") + call, False),
            (block(TEXT, "Sure."), False),
            (inkling2_call("weather", {"days": 3}), False),
            (inkling2_call("weather", {"city": 3}), False),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.accepts(grammar, text), accepted)

    def test_a_response_schema_is_answered_in_a_text_block(self):
        grammar = tool_schema.json_grammar(
            {"type": "object", "properties": {"a": {"type": "integer"}}},
            False,
            chat_format=self.chat_format,
        )
        for text, accepted in (
            (block(TEXT, '{"a":1}'), True),
            (block(THINKING, "Think.") + block(TEXT, '{"a":1}'), True),
            ('{"a":1}', False),
            (block(TEXT, '{"a":"x"}'), False),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.accepts(grammar, text), accepted)

    def test_what_the_grammar_writes_reads_back_as_written(self):
        # A required free call, as the grammar lets the model write it, reads
        # back as the call (InklingCallReadingTests' path).
        grammar = tool_schema.tool_grammar(
            self.policy("required"), False, chat_format=self.chat_format
        )
        call = inkling2_call("weather", {"city": "Paris", "detail": {"a": [1]}})
        self.assertTrue(self.accepts(grammar, call))
        self.assertIn(
            'call_0: <|message_model|> "weather" <|content_invoke_tool_json|> '
            '"{\\"name\\":\\"weather\\",\\"args\\":" @free_arguments "}" '
            "<|end_message|>\n",
            json.loads(grammar)["grammars"][0]["lark_grammar"],
        )


class InklingTokenizerContractTests(unittest.TestCase):
    def test_inkling_2_s_tokens_are_checked_against_its_format(self):
        tokenizer = inkling2_tokenizer()
        chat_format = inkling2_format(tokenizer)
        constraints.validate_tokenizer(tokenizer, chat_format)
        model, thinking = chat_format.think_start_ids
        for changed, refusal in (
            (
                {"think_start_ids": (model, thinking + 1)},
                "tokenizer must encode '<\\|message_model\\|><\\|content_thinking\\|>' "
                "as native tokens",
            ),
            ({"think_end_id": chat_format.think_end_id + 1}, "as native token"),
            ({"visible_tags": ("<|message_nobody|>",)}, "the tag"),
            ({"stop_tokens": ("<|content_model_end_sampling|>x",)}, "stop text"),
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(
                    constraints.TokenizerContractError, refusal
                ):
                    constraints.validate_tokenizer(
                        tokenizer, replace(chat_format, **changed)
                    )
        ids = tokenizer.convert_tokens_to_ids
        constraints.check_capabilities(
            tokenizer,
            chat_format,
            wire.ReadyEvent(
                4,
                4096,
                False,
                len(tokenizer),
                9,
                (ids("<|content_model_end_sampling|>"),),
            ),
        )


class InklingPromptTests(unittest.TestCase):
    """Inkling 2's template renders as shipped: it appends no generation
    prompt, as its model opens its own message, and tells the model the
    request's effort in a system message."""

    def frontend(self, **options):
        tokenizer = inkling2_tokenizer()
        return fixtures.make_frontend(
            tokenizer,
            None,
            "test-model",
            4096,
            10,
            2,
            vision=False,
            chat_format=inkling2_format(tokenizer),
            chat_templates=ChatTemplates(tokenizer, opens_own_message=True),
            **options,
        )

    def test_the_probe_takes_the_empty_generation_prompt_of_its_format(self):
        tokenizer = inkling2_tokenizer()
        template = ChatTemplates(tokenizer, opens_own_message=True).select(None)
        self.assertEqual(template.later_system, NATIVE)
        self.assertEqual(set(template.generation_prompts.values()), {("", ())})
        # A custom template must serve every request: the same one serves
        # where the model opens its own message, and only there.
        ChatTemplates(tokenizer, override=INKLING2_TEMPLATE, opens_own_message=True)
        with self.assertRaisesRegex(ChatTemplateError, "serves no request"):
            ChatTemplates(tokenizer, override=INKLING2_TEMPLATE)

    def test_the_request_s_effort_is_told_to_the_model(self):
        frontend = self.frontend()
        question = "<|message_user|>" + TEXT + "hello" + END
        effort = "<|message_system|>" + TEXT + "Thinking effort level: "
        for options, level in (
            ({}, "0.9"),
            ({"reasoning_effort": "none"}, "0"),
            ({"chat_template_kwargs": {"enable_thinking": False}}, "0"),
            ({"reasoning_effort": "low"}, "0.2"),
            ({"reasoning_effort": "max"}, "0.99"),
        ):
            with self.subTest(**options):
                body = fixtures.chat_body(**options)
                job = frontend.prepare(body, deadline=fixtures.FOREVER)
                self.assertFalse(job.thinking)
                self.assertIsNone(job.thinking_token_budget)
                self.assertEqual(
                    frontend.apply_template(body, deadline=fixtures.FOREVER),
                    effort + level + END + question,
                )

    def test_models_report_the_efforts_its_template_renders(self):
        self.assertEqual(
            self.frontend().reasoning,
            {
                "thinks": True,
                "efforts": ["minimal", "low", "medium", "high", "xhigh"],
                "default": "high",
            },
        )

    def test_tools_render_in_its_tool_declaration(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "weather", "parameters": SCHEMAS["weather"]},
            }
        ]
        prompt = self.frontend().apply_template(
            fixtures.chat_body(tools=tools), deadline=fixtures.FOREVER
        )
        self.assertTrue(
            prompt.startswith('<|message_system|>tool_declare<|content_xml|>[{"desc'),
            prompt,
        )

    def test_scores_are_read_where_the_model_answers(self):
        # After an empty reasoning block and the opening of its answer.
        frontend = self.frontend()
        judgment, _ = scoring.prepare_judgment(
            frontend,
            fixtures.judgment_body(),
            deadline=fixtures.FOREVER,
            disconnected=fixtures.stays_connected,
        )
        ids = frontend.tokenizer.convert_tokens_to_ids
        self.assertEqual(
            list(judgment.prompt_tokens[-5:]),
            [ids(MODEL), ids(THINKING), ids(END), ids(MODEL), ids(TEXT)],
        )

    def test_prompt_tokens_are_reused_after_the_last_message(self):
        tokenizer = inkling2_tokenizer()
        cache = PromptTokenizer(
            tokenizer, INKLING2.message_boundary, fixtures.NATIVE_CONTEXT
        )
        self.assertTrue(cache.enabled)
        first = "<|message_user|>" + TEXT + "Paris 12345 " * 400 + END
        later = first + block(TEXT, "Sunny.")
        for text in (first, later):
            self.assertEqual(
                cache.encode(text),
                tokenizer(text, add_special_tokens=False)["input_ids"],
            )
        self.assertEqual(cache.stats()["hits"], 1)


class InklingServingTests(fixtures.HarnessTestCase):
    """Inkling 2's output through the server, from the tokens the engine
    returns, in every API's shape, streamed or not."""

    TAGS = {
        50: MODEL,
        51: THINKING,
        53: END,
        54: TEXT,
        59: "<|content_invoke_tool_json|>",
    }
    FRAGMENTS = TAGS | {
        52: "Check.",
        55: "Sunny",
        56: " today.",
        57: "weather",
        58: '{"name":"weather","args":{"ci',
        60: 'ty":"Paris"}}',
    }
    CALLED = [[50], [51], [52], [53], [50], [57], [59], [58], [60], [53]]
    # Inkling 2's format over the fake vocabulary's IDs.
    CHAT_FORMAT = replace(
        INKLING2, think_end_id=53, think_start_ids=(50, 51), call_open_id=50
    )
    REASONED = [[50], [51], [52], [53], [50], [54], [55], [56], [53]]
    # A text block, then reasoning, then another text block.
    RETURNING = [[50], [54], [55], [53], [50], [51], [52], [53], [50], [54], [56], [53]]

    def harness(self, *plans, **options):
        tokenizer = fixtures.TemplateTokenizer(INKLING2_TEMPLATE)
        tokenizer.fragments = self.FRAGMENTS
        tokenizer.backend_tokenizer = fixtures.byte_backend(self.FRAGMENTS)
        tags = {text: token for token, text in self.TAGS.items()}
        tokenizer.convert_tokens_to_ids = tags.get
        return super().harness(
            fixtures.FakeRuntime(*plans),
            tokenizer=tokenizer,
            chat_format=self.CHAT_FORMAT,
            chat_templates=ChatTemplates(tokenizer, opens_own_message=True),
            **options,
        )

    def test_chat_completions_read_the_reasoning_and_the_answer(self):
        harness = self.harness(
            fixtures.Plan(self.REASONED), fixtures.Plan(self.REASONED)
        )
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", fixtures.chat_body()
        )
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        message = response["choices"][0]["message"]
        self.assertEqual(
            (message["reasoning_content"], message["content"]),
            ("Check.", "Sunny today."),
        )
        self.assertEqual(
            response["usage"]["completion_tokens_details"]["reasoning_tokens"], 1
        )
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", fixtures.chat_body(stream=True)
        )
        self.assertEqual(status, 200, payload)
        deltas = [
            event["choices"][0]["delta"]
            for event in fixtures.stream_events(payload)
            if event.get("choices")
        ]
        self.assertEqual(
            "".join(delta.get("reasoning_content") or "" for delta in deltas),
            "Check.",
        )
        self.assertEqual(
            "".join(delta.get("content") or "" for delta in deltas), "Sunny today."
        )

    def test_reasoning_after_text_keeps_its_place_where_an_api_has_one(self):
        harness = self.harness(
            fixtures.Plan(self.RETURNING),
            fixtures.Plan(self.RETURNING),
            fixtures.Plan(self.RETURNING),
        )
        # Chat Completions has one reasoning field and one answer.
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", fixtures.chat_body()
        )
        self.assertEqual(status, 200, payload)
        message = json.loads(payload)["choices"][0]["message"]
        self.assertEqual(
            (message["reasoning_content"], message["content"]),
            ("Check.", "Sunny today."),
        )
        # Messages and Responses keep the blocks in order.
        status, _, payload = harness.request(
            "POST", "/v1/messages", fixtures.anthropic_body(max_tokens=64)
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            [
                (block["type"], block.get("text", block.get("thinking")))
                for block in json.loads(payload)["content"]
            ],
            [("text", "Sunny"), ("thinking", "Check."), ("text", " today.")],
        )
        status, _, payload = harness.request(
            "POST", "/v1/responses", fixtures.responses_body()
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            [item["type"] for item in json.loads(payload)["output"]],
            ["message", "reasoning", "message"],
        )

    def test_a_call_is_read_streamed_and_complete(self):
        harness = self.harness(fixtures.Plan(self.CALLED), fixtures.Plan(self.CALLED))
        tools = [
            {
                "type": "function",
                "function": {"name": "weather", "parameters": SCHEMAS["weather"]},
            }
        ]
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", fixtures.chat_body(tools=tools)
        )
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        message = response["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "Check.")
        self.assertEqual(
            [
                (call["function"]["name"], call["function"]["arguments"])
                for call in message["tool_calls"]
            ],
            [("weather", '{"city":"Paris"}')],
        )
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", fixtures.chat_body(tools=tools, stream=True)
        )
        self.assertEqual(status, 200, payload)
        deltas = [
            event["choices"][0]["delta"]
            for event in fixtures.stream_events(payload)
            if event.get("choices")
        ]
        self.assertEqual(
            "".join(
                call["function"].get("arguments", "")
                for delta in deltas
                for call in delta.get("tool_calls") or ()
            ),
            '{"city":"Paris"}',
        )

    def test_a_text_completion_holds_no_tags(self):
        harness = self.harness(fixtures.Plan([[55], [56]]))
        status, _, payload = harness.request(
            "POST",
            "/v1/completions",
            {"model": "test-model", "prompt": "Weather?", "max_tokens": 8},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["choices"][0]["text"], "Sunny today.")


class ServedFamilyTests(unittest.TestCase):
    """main() reads its output in the chat format of the family the model was
    installed as, which its assembly records, and serves no other."""

    def main(self, family):
        runtime = mock.Mock()
        runtime.capabilities = fixtures.engine_ready(131072)
        tokenizer = object()
        with (
            mock.patch.object(api, "parse_args", return_value=fixtures.main_args()),
            fixtures.unprepared_hold(family),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(api, "validate_tokenizer") as validate,
            mock.patch.object(api, "check_capabilities") as check,
            mock.patch.object(api, "thinking_end_tokens", return_value=None),
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
            check=check,
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
                started.check.assert_called_once_with(
                    started.tokenizer, family.chat_format, fixtures.engine_ready(131072)
                )
                # The table, built while the engine loads, covers the
                # installation's vocabulary and stops as the format does.
                started.factory_type.table.assert_called_once_with(
                    started.tokenizer, family.chat_format, 248320
                )
                started.factory_type.assert_called_once_with(
                    started.factory_type.table.return_value,
                    fixtures.engine_ready(131072),
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
