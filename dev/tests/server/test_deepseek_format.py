"""DeepSeek-V4.1-Flash's chat format: its calls in DSML, one block of them
after the text, read, constrained and rendered as Qwen's and Flint's calls
are; tool names that hold none of DSML's markup and that its template takes;
its embedded template, which renders DeepSeek's own encoding cases, text
that quotes its image placeholder and a history call of a name it refuses;
and a generation prompt that opens or closes the think block by the
request's reasoning."""

import json
import unittest
from dataclasses import replace
from pathlib import Path

from jinja2 import TemplateError
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
from dev.tests.tool_output import project, put, streamed_text, tool_policy
from server import constraints, tool_schema
from server import output as model_output
from server.api_shapes import template_messages
from server.chat_format import (
    DEEPSEEK,
    DEEPSEEK_DSML,
    DEEPSEEK_IMAGE,
    DSML,
    GLM_ARGS,
    QWEN,
    QWEN_XML,
)
from server.chat_templates import (
    NATIVE,
    ChatTemplates,
    render_chat_template,
    template_options,
)
from server.errors import APIError
from server.frontend import PLACEHOLDER_RENDER_MARKER
from server.serve_options import REASONING_EFFORTS
from server.tokenization import PromptTokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/chat_templates"
TEMPLATE = (FIXTURES / "deepseek41.jinja").read_text()
# DeepSeek's control tokens, special in its GGUF, and the tags its template
# writes in text, added tokens that are not special.
SPECIAL = (
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜System｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜latest_reminder｜>",
)
TAGS = ("<think>", "</think>", DSML)
SCHEMAS = {
    "weather": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "units": {"enum": ["metric", "imperial"]},
            "detail": {"type": "object"},
            "note": {},
        },
        "required": ["city"],
    },
    "time": {"type": "object", "properties": {}},
}
# The reasoning effort each of Splash's efforts renders as, through the
# aliases of those DeepSeek's template does not know.
BUDGETS = {"minimal": 50, "low": 50, "medium": 75, "high": 75, "xhigh": 100, "max": 100}


def deepseek_tokenizer():
    """A byte-level BPE tokenizer with DeepSeek's added tokens, each kind as
    its GGUF adds it, and its embedded chat template."""
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
    backend.add_special_tokens([AddedToken(text, normalized=False) for text in SPECIAL])
    backend.add_tokens([AddedToken(text, normalized=False) for text in TAGS])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token=SPECIAL[0], eos_token=SPECIAL[1]
    )
    tokenizer.chat_template = TEMPLATE
    return tokenizer


def deepseek_format(tokenizer):
    """DeepSeek's chat format over `tokenizer`'s token IDs."""
    ids = tokenizer.convert_tokens_to_ids
    return replace(
        DEEPSEEK,
        vocabulary=len(tokenizer),
        stop_tokens=tuple((text, ids(text)) for text, _ in DEEPSEEK.stop_tokens),
        think_end_id=ids("</think>"),
        call_open_id=ids(DSML),
    )


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def qwen_call(name, arguments):
    """A call as Qwen's template writes it."""
    return (
        f"<tool_call>\n<function={name}>\n"
        + "".join(
            f"<parameter={key}>\n{_text(value)}\n</parameter>\n"
            for key, value in arguments.items()
        )
        + "</function>\n</tool_call>"
    )


def dsml_call(name, arguments):
    """A call as DeepSeek's template writes it: each value a string as its
    text, marked so, and any other as JSON."""
    parameters = "\n".join(
        f'<{DSML} parameter name="{key}" '
        f'string="{"true" if isinstance(value, str) else "false"}">'
        f"{_text(value)}</{DSML} parameter>"
        for key, value in arguments.items()
    )
    return f'<{DSML} invoke name="{name}">\n{parameters}\n</{DSML} invoke>'


def dsml_calls(*calls):
    """The block DeepSeek's template writes a turn's calls in, after its
    text: each call a (name, arguments) pair."""
    return (
        f"\n\n<{DSML} calls>\n"
        + "\n".join(dsml_call(*call) for call in calls)
        + f"\n</{DSML} calls>"
    )


class DsmlReadingTests(unittest.TestCase):
    """DeepSeek's calls read as Qwen's read: the same content and calls from
    the same calls in either layout, however the output is cut into pieces,
    without the space that frames DeepSeek's block."""

    def assert_reads_alike(self, text, *calls):
        policy = tool_policy(SCHEMAS)
        expected = project(
            text + "".join(qwen_call(*call) for call in calls), policy, "x"
        )[:2]
        for size in (None, *range(1, 12)):
            with self.subTest(size=size):
                content, read, events = project(
                    text + dsml_calls(*calls),
                    policy,
                    "x",
                    size=size,
                    syntax=DEEPSEEK_DSML,
                )
                self.assertEqual((content, read), expected)
                self.assertEqual(streamed_text(events), content)
        return expected

    def test_values_convert_by_the_types_their_tool_declares(self):
        content, calls = self.assert_reads_alike(
            "Let me check.",
            (
                "weather",
                {"city": "São Paulo", "days": 3, "units": "metric", "detail": {}},
            ),
        )
        self.assertEqual(content, "Let me check.")
        self.assertEqual(
            calls[0]["function"]["arguments"],
            '{"city":"São Paulo","days":3,"units":"metric","detail":{}}',
        )

    def test_parallel_calls_share_one_block(self):
        content, calls = self.assert_reads_alike(
            "", ("weather", {"city": "Paris"}), ("time", {}), ("time", {})
        )
        self.assertEqual(content, "")
        self.assertEqual(
            [call["function"]["arguments"] for call in calls],
            ['{"city":"Paris"}', "{}", "{}"],
        )

    def test_a_value_marked_as_json_reads_as_it_where_its_schema_gives_no_type(self):
        # A declared type decides how a value reads, whatever its mark; the
        # mark decides it where the schema gives none.
        text = (
            f'<{DSML} calls>\n<{DSML} invoke name="weather">\n'
            f'<{DSML} parameter name="days" string="true">3</{DSML} parameter>\n'
            f'<{DSML} parameter name="note" string="false">{{"a": [1]}}'
            f"</{DSML} parameter>\n"
            f'<{DSML} parameter name="other" string="true">{{"a": [1]}}'
            f"</{DSML} parameter>\n"
            f"</{DSML} invoke>\n</{DSML} calls>"
        )
        for size in (None, 3):
            with self.subTest(size=size):
                _, calls, _ = project(
                    text, tool_policy(SCHEMAS), size=size, syntax=DEEPSEEK_DSML
                )
                self.assertEqual(
                    json.loads(calls[0]["function"]["arguments"]),
                    {"days": 3, "note": {"a": [1]}, "other": '{"a": [1]}'},
                )

    def test_the_space_around_the_block_frames_the_calls(self):
        call = ("time", {})
        for text in (
            "Sure." + dsml_calls(call) + "\n",
            f'Sure.<{DSML} calls><{DSML} invoke name="time"></{DSML} invoke>'
            f"</{DSML} calls>",
        ):
            with self.subTest(text=text):
                content, calls, events = project(
                    text, tool_policy(SCHEMAS), syntax=DEEPSEEK_DSML
                )
                self.assertEqual((content, streamed_text(events)), ("Sure.", "Sure."))
                self.assertEqual([c["function"]["name"] for c in calls], ["time"])

    def test_a_cut_call_keeps_the_arguments_it_has(self):
        text = dsml_calls(("weather", {"city": "Paris", "days": 3}))
        # From the end of the function's name on.
        for cut in range(text.index('">\n') + 1, len(text)):
            with self.subTest(cut=cut):
                _, calls, _ = project(
                    text[:cut],
                    tool_policy(SCHEMAS),
                    incomplete=True,
                    syntax=DEEPSEEK_DSML,
                )
                self.assertEqual(
                    [call["function"]["name"] for call in calls], ["weather"]
                )

    def test_an_invoke_that_names_no_function_reads_as_its_text(self):
        # As in Qwen's layout: an invoke without a name, or with one too long
        # to be a name, is the text the model wrote, and a call after it in
        # the block still reads.
        for name in ("", "  ", "x" * (model_output.MAX_NAME_LENGTH + 1)):
            text = dsml_calls((name, {"item": "x"}), ("time", {}))
            for size in (None, 1, 5):
                with self.subTest(name=name[:8], size=size):
                    content, calls, events = project(
                        text, tool_policy(SCHEMAS), size=size, syntax=DEEPSEEK_DSML
                    )
                    self.assertEqual(
                        (content.strip(), streamed_text(events)),
                        (dsml_call(name, {"item": "x"}), content),
                    )
                    self.assertEqual(
                        [call["function"]["name"] for call in calls], ["time"]
                    )

    def test_without_tools_a_block_is_text(self):
        text = "Here:" + dsml_calls(("weather", {"city": "Paris"}))
        projector = model_output.StreamingToolCallProjector(None, "x")
        put(projector, text + "</think>")
        self.assertEqual(projector.finish(False)[:2], (text, []))


class DsmlGrammarTests(unittest.TestCase):
    """DeepSeek's constrained output: its reasoning where the generation
    prompt opened it, then its text and one block of calls, compiled over a
    tokenizer that adds DeepSeek's tokens as its GGUF does."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = deepseek_tokenizer()
        cls.chat_format = deepseek_format(cls.tokenizer)
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

    def grammar(self, policy, thinking=False):
        return tool_schema.tool_grammar(policy, thinking, chat_format=self.chat_format)

    def accepts(self, grammar, text):
        matcher = self.factory.create(grammar).matcher
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return matcher.consume_tokens(tokens) and matcher.is_accepting()

    def assert_accepts(self, grammar, cases):
        for text, accepted in cases:
            with self.subTest(text=text):
                self.assertEqual(self.accepts(grammar, text), accepted)

    def test_the_calls_follow_the_text_in_one_block(self):
        weather, time = ("weather", {"city": "Paris", "days": 3}), ("time", {})
        self.assert_accepts(
            self.grammar(self.policy()),
            (
                ("Paris it is.", True),
                ("Let me <b>check</b>." + dsml_calls(weather), True),
                (dsml_calls(weather, time, time), True),
                ("Done." + dsml_calls(time) + "\n", True),
                ("Done." + dsml_calls(time) + " More text.", False),
                ("Done." + dsml_calls(time) + dsml_calls(time), False),
                (f"A stray {DSML} tag.", False),
                ("Closed again.</think>", False),
            ),
        )
        self.assert_accepts(
            self.grammar(self.policy(parallel=False)),
            (
                ("Check." + dsml_calls(weather), True),
                ("Check." + dsml_calls(weather, time), False),
            ),
        )

    def test_a_required_call_comes_first_after_the_reasoning(self):
        block = dsml_calls(("weather", {"city": "Paris"}), ("time", {}))
        self.assert_accepts(
            self.grammar(self.policy("required"), thinking=True),
            (
                ("Paris, then.</think>" + block, True),
                ("</think>" + block, True),
                ("No close." + block, False),
                ("Paris.</think>Sure." + block, False),
                ("Paris.</think>No call.", False),
            ),
        )

    def test_strict_arguments_are_generated_with_their_marks(self):
        grammar = self.grammar(self.policy("required", strict=True))
        cases = [
            ({"city": "Paris", "days": 3, "units": "metric"}, True),
            ({"city": "Paris", "detail": {}, "note": [1, "a"]}, True),
            ({"city": "Paris", "note": "text"}, True),
            ({"city": "Paris", "days": "three"}, False),
            ({"city": "Paris", "units": "kelvin"}, False),
            ({"city": "Paris", "detail": {"extra": 1}}, False),
            ({"days": 3}, False),
            ({"city": "Paris", "other": 1}, False),
        ]
        self.assert_accepts(
            grammar,
            [(dsml_calls(("weather", arguments)), ok) for arguments, ok in cases]
            + [
                # A number written as text, or a string as JSON.
                (dsml_calls(("weather", {"city": "Paris", "days": "3"})), False),
                (
                    dsml_calls(("weather", {"city": "Paris"})).replace(
                        'string="true">Paris', 'string="false">"Paris"'
                    ),
                    False,
                ),
                # The template's empty line where a call has no arguments.
                (dsml_calls(("time", {})), True),
            ],
        )

    def test_what_the_grammar_writes_reads_back_as_written(self):
        arguments = {"city": "New York", "days": 2, "units": "imperial"}
        text = "Check.</think>" + dsml_calls(("weather", arguments))
        self.assertTrue(
            self.accepts(self.grammar(self.policy("required", strict=True), True), text)
        )
        splitter = model_output.ReasoningSplitter(True, True, self.chat_format)
        parts = splitter.put(text) + splitter.finish()
        reasoning = "".join(v for kind, v in parts if kind == "reasoning_content")
        content = "".join(v for kind, v in parts if kind == "content")
        _, calls, _ = project(content, self.policy(), syntax=DEEPSEEK_DSML)
        self.assertEqual(reasoning, "Check.")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), arguments)

    def test_free_arguments_take_any_name_text_and_mark(self):
        grammar = tool_schema._free_arguments(DEEPSEEK_DSML)
        self.assertFalse(LLMatcher.validate_grammar(grammar, self.factory.tokenizer))
        matcher = LLMatcher(self.factory.tokenizer, grammar)
        text = (
            f'<{DSML} parameter name="a b" string="true">any\ntext</{DSML} parameter>\n'
            f'<{DSML} parameter name="c" string="false">[1]</{DSML} parameter>\n'
        )
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.assertTrue(matcher.consume_tokens(tokens) and matcher.is_accepting())

    def test_the_grammars_write_dsml_as_text(self):
        grammars = json.loads(self.grammar(self.policy("required", parallel=False)))
        main = grammars["grammars"][0]["lark_grammar"]
        calls_open = json.dumps(f"<{DSML} calls>\n")
        calls_close = json.dumps(f"</{DSML} calls>")
        opening = json.dumps(f'<{DSML} invoke name="time">\n')
        invoke_close = json.dumps(f"</{DSML} invoke>")
        newline = json.dumps("\n")
        self.assertIn(
            f"start: WS {calls_open} (call_0 | call_1) {newline} {calls_close} WS\n",
            main,
        )
        self.assertIn(
            f"call_1: {opening} @free_arguments {newline}? {invoke_close}\n", main
        )
        self.assertIn(
            "TEXT: /(?s:.*)/ & ~/(?s:.*)(｜DSML｜|<\\/think>)(?s:.*)/\n", main
        )


class DsmlToolNameTests(unittest.TestCase):
    """A tool's name is written inside its calls' DSML tags, and a strict
    tool's parameter names inside theirs: a name that holds DSML's markup,
    the marker its tags open with or the quote a function's name ends at, is
    refused, naming the tool, and so is one that holds the separator of a
    namespace more than once, which the template refuses. A history call of
    such a name stays text. Qwen's and Flint's tools keep every name they
    took."""

    def policy(self, name, syntax=DEEPSEEK_DSML, **function):
        tools = [{"type": "function", "function": {"name": name, **function}}]
        return tool_schema.normalize_tools(tools, "auto", True, syntax=syntax)[1]

    def test_a_name_holding_dsml_markup_is_refused(self):
        for name, markup in (
            (f"read{DSML}file", DSML),
            (DSML, DSML),
            ('bash"x', '"'),
            ('say "hi"', '"'),
        ):
            with self.subTest(name=name):
                with self.assertRaises(APIError) as caught:
                    self.policy(name)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(
                    caught.exception.message,
                    f"tool name {name} must not contain {markup!r}, "
                    "the markup of this model's tool calls",
                )
        for name in (
            "weather",
            "ns::lookup",
            "mcp__github__list_issues",
            "web search",
            "files:read/v2",
            "天气",
            "｜DSML",
            "a'b",
        ):
            with self.subTest(name=name):
                self.assertEqual(list(self.policy(name).schemas), [name])

    def test_reading_ends_a_call_s_function_name_at_the_quote(self):
        # Why the quote is refused: reading ends a function's name there.
        text = dsml_calls(("bash", {})).replace('"bash"', '"bash"x"')
        _, calls, _ = project(text, tool_policy(SCHEMAS), syntax=DEEPSEEK_DSML)
        self.assertEqual([call["function"]["name"] for call in calls], ["bash"])

    def test_a_name_holding_the_namespace_separator_twice_is_refused(self):
        # The template reads a name's text before its first :: as the tool's
        # namespace and refuses the name if the rest holds another, naming
        # only the rest: every request that offered the tool failed.
        for name in ("a::b::c", "a::::b", "::::", "ns::lookup::"):
            with self.subTest(name=name):
                with self.assertRaises(APIError) as caught:
                    self.policy(name)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(
                    caught.exception.message,
                    f"tool name {name} must not contain '::' more than once, "
                    "the separator of this model's tool namespaces",
                )
        for name in ("ns::lookup", "a:::b", "::", "lookup::", "::lookup"):
            with self.subTest(name=name):
                self.assertEqual(list(self.policy(name).schemas), [name])

    def test_a_history_call_of_a_refused_name_stays_text_in_its_place(self):
        # As a call whose arguments are text: the call of the name the
        # template refuses becomes the text of a turn of its own, between the
        # calls before and after it.
        def call(name, arguments):
            return {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }

        before, after = call("weather", {"city": "Paris"}), call("time", {})
        made_up = call("a::b::c", {"city": "Paris"})
        message = {
            "role": "assistant",
            "content": "Let me look.",
            "reasoning_content": "Both.",
            "tool_calls": [before, made_up, after],
        }
        self.assertEqual(
            template_messages([message], syntax=DEEPSEEK_DSML),
            [
                {**message, "tool_calls": [before]},
                {
                    "role": "assistant",
                    "content": 'Incomplete tool call:\n{"name": "a::b::c", '
                    '"arguments": {"city": "Paris"}}',
                },
                {"role": "assistant", "content": "", "tool_calls": [after]},
            ],
        )
        # Qwen's and Flint's templates write any name.
        for syntax in (None, QWEN_XML, GLM_ARGS):
            with self.subTest(syntax=syntax and syntax.key_open):
                self.assertEqual(template_messages([message], syntax=syntax), [message])

    def test_other_families_take_the_names_they_took(self):
        # Their markup is or holds < or >, which no tool name holds, and their
        # templates read no namespace from a name.
        for syntax in (QWEN_XML, GLM_ARGS):
            for markup in (syntax.marker, syntax.name_end):
                self.assertTrue({"<", ">"} & set(markup), markup)
            self.assertIsNone(syntax.namespace_separator)
            for name in (f"read{DSML}file", 'bash"x', "a::b::c", "weather"):
                with self.subTest(syntax=syntax.key_open, name=name):
                    self.assertEqual(list(self.policy(name, syntax).schemas), [name])

    def test_a_strict_parameter_name_holding_the_marker_is_refused(self):
        schema = {"type": "object", "properties": {f"a{DSML}b": {"type": "string"}}}
        for chat_format, refused in ((DEEPSEEK, True), (QWEN, False)):
            policy = self.policy(
                "tool",
                chat_format.call_syntax,
                parameters=schema,
                strict=True,
            )
            with self.subTest(syntax=chat_format.call_syntax.key_open):
                if not refused:
                    tool_schema.tool_grammar(policy, False, chat_format=chat_format)
                    continue
                with self.assertRaisesRegex(APIError, "invalid tool parameter name"):
                    tool_schema.tool_grammar(policy, False, chat_format=chat_format)


class DeepSeekTokenizerContractTests(unittest.TestCase):
    def test_deepseek_s_tokens_are_checked_against_its_format(self):
        tokenizer = deepseek_tokenizer()
        chat_format = deepseek_format(tokenizer)
        constraints.validate_tokenizer(tokenizer, chat_format)
        for field, text in (("think_end_id", "</think>"), ("call_open_id", DSML)):
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

    def test_the_real_ids_are_deepseek_s(self):
        # Those of the GGUF's vocabulary (dev/tests/install/test_gguf_deepseek41.py).
        self.assertEqual(
            (
                DEEPSEEK.vocabulary,
                DEEPSEEK.stop_tokens,
                DEEPSEEK.think_start_id,
                DEEPSEEK.think_end_id,
                DEEPSEEK.call_open_id,
            ),
            (129280, (("<｜end▁of▁sentence｜>", 1),), None, 128822, 128825),
        )
        self.assertIs(DEEPSEEK.call_syntax, DEEPSEEK_DSML)
        self.assertTrue(DEEPSEEK.reasoning_switch)
        self.assertFalse(DEEPSEEK.calls_end_reasoning)
        # The image placeholder its template writes and refuses in text.
        self.assertEqual(DEEPSEEK.refused_placeholder, "<｜deepseek_image｜>")
        self.assertIn(f'set IMG = "{DEEPSEEK_IMAGE}"', TEMPLATE)


class DeepSeekPromptTests(unittest.TestCase):
    """DeepSeek's embedded template renders as its reference encoder does,
    through the options a request passes it, also where a message's text
    quotes the image placeholder the template refuses there or a history
    call names a function by a name it refuses, and its generation prompt
    opens the think block where the request thinks."""

    def frontend(self, **changes):
        """A frontend serving DeepSeek's format, with `changes` to it."""
        tokenizer = deepseek_tokenizer()
        return fixtures.make_frontend(
            tokenizer,
            None,
            "test-model",
            8192,
            10,
            2,
            vision=False,
            chat_format=replace(deepseek_format(tokenizer), **changes),
        )

    def test_the_template_renders_deepseek_s_own_encoding_cases(self):
        tokenizer = deepseek_tokenizer()
        frontend = self.frontend()
        cases = sorted((FIXTURES / "deepseek41-encoding").glob("test_input_*.json"))
        self.assertEqual(len(cases), 5)
        for case in cases:
            with self.subTest(case=case.name):
                data = json.loads(case.read_text())
                if isinstance(data, list):
                    data = {"messages": data}
                # A history call's arguments are an object, as the server
                # reads them from JSON text.
                for message in data["messages"]:
                    for call in message.get("tool_calls") or ():
                        function = call["function"]
                        function["arguments"] = json.loads(function["arguments"])
                thinking = data.get("thinking_mode") == "thinking"
                options = template_options(
                    reasoning_effort=data.get("reasoning_effort")
                    if thinking
                    else "none",
                    preserve_thinking=None,
                    tools=data.get("tools"),
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
                expected = case.with_name(
                    case.name.replace("input", "output").replace(".json", ".txt")
                ).read_text()
                self.assertEqual(
                    render_chat_template(
                        tokenizer, data["messages"], options | {"tokenize": False}
                    ),
                    expected,
                )
                # As requests render it, with a marker for its image
                # placeholder, which the image blocks of case 5 write, and
                # their calls as calls.
                self.assertEqual(
                    frontend._apply_chat_template(
                        template_messages(data["messages"], syntax=DEEPSEEK_DSML),
                        options | {"chat_template": TEMPLATE, "tokenize": False},
                    ),
                    expected,
                )
                # Cases 1 and 2 hold only what a request carries, the others
                # DeepSeek's own roles and fields or images by path: requests
                # of them render alike, through every step of preparation.
                # Each ends with an answer, after which the template sets no
                # generation prompt.
                if case.name in ("test_input_1.json", "test_input_2.json"):
                    body = fixtures.chat_body(
                        messages=data["messages"],
                        tools=data.get("tools"),
                        reasoning_effort=data.get("reasoning_effort")
                        if thinking
                        else "none",
                        add_generation_prompt=False,
                    )
                    self.assertEqual(
                        frontend.apply_template(body, deadline=fixtures.FOREVER),
                        expected,
                    )

    def test_the_probe_finds_later_system_messages_native_and_two_prompts(self):
        template = ChatTemplates(deepseek_tokenizer()).select(None)
        self.assertEqual(template.later_system, NATIVE)
        self.assertEqual(
            {text for text, _ in template.generation_prompts.values()},
            {"<｜Assistant｜><think>", "<｜Assistant｜></think>"},
        )

    def test_each_effort_opens_the_think_block_but_none(self):
        frontend = self.frontend()
        # Its template knows low, high (its default) and max, which /v1/models
        # names the others by, as they render as their aliases.
        self.assertEqual(
            frontend.reasoning,
            {"thinks": True, "efforts": ["low", "high", "max"], "default": "high"},
        )
        for effort in REASONING_EFFORTS:
            with self.subTest(effort=effort):
                body = fixtures.chat_body(reasoning_effort=effort)
                job = frontend.prepare(body, deadline=fixtures.FOREVER)
                prompt = frontend.apply_template(body, deadline=fixtures.FOREVER)
                self.assertEqual(job.thinking, effort != "none")
                if effort == "none":
                    self.assertEqual(
                        prompt,
                        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>",
                    )
                else:
                    self.assertEqual(
                        prompt,
                        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: "
                        f"{BUDGETS[effort]} (range 1-100, the higher the value, "
                        "the more thorough the reasoning)\n\n"
                        "<｜User｜>hello<｜Assistant｜><think>",
                    )

    def test_a_tool_conversation_renders_in_dsml(self):
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
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜><｜System｜>"))
        self.assertIn("Be brief.\n\n## Tools\n", prompt)
        # With tools, the history keeps its reasoning.
        self.assertTrue(
            prompt.endswith(
                "<｜User｜>Weather in Paris?<｜Assistant｜><think>Look it up.</think>"
                + dsml_calls(("weather", {"city": "Paris", "days": 2}))
                + "<｜end▁of▁sentence｜><｜User｜><tool_result>Sunny</tool_result>"
                "<｜Assistant｜><think>"
            ),
            prompt,
        )

    def test_the_template_refuses_the_names_its_syntax_refuses(self):
        # In a tool's declaration and in a call of it alike: a name it takes
        # it writes as it is, its namespace's :: too.
        tokenizer = deepseek_tokenizer()
        ask = {"role": "user", "content": "Go."}
        for name in (
            "lookup",
            "ns::lookup",
            "a:::b",
            "::",
            ":::",
            "lookup::",
            "::lookup",
            "a::b::c",
            "a::::b",
            "::::",
            "a::b:::c",
            "ns::lookup::",
        ):
            call = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": {}}}
                ],
            }
            for where, messages, options, written in (
                (
                    "declaration",
                    [ask],
                    {"tools": [{"type": "function", "function": {"name": name}}]},
                    f'{{"name": {json.dumps(name)}}}',
                ),
                ("call", [ask, call, ask], {}, f'invoke name="{name}">'),
            ):
                with self.subTest(name=name, where=where):
                    options = options | {
                        "add_generation_prompt": True,
                        "tokenize": False,
                    }
                    if not DEEPSEEK_DSML.refuses_name(name):
                        self.assertIn(
                            written, render_chat_template(tokenizer, messages, options)
                        )
                        continue
                    with self.assertRaisesRegex(
                        TemplateError, "Tool name must not contain '::'"
                    ):
                        render_chat_template(tokenizer, messages, options)

    def test_a_history_call_of_a_name_the_template_refuses_renders_as_text(self):
        # A model that calls tools freely may call one by any name, made up or
        # read in a page, and the server returns the call as it reads it; the
        # client sends it back in every later request of the conversation.
        def call(call_id, name, arguments):
            return {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }

        def result(call_id, content):
            return {"role": "tool", "tool_call_id": call_id, "content": content}

        ask = {"role": "user", "content": "Weather in Paris?"}
        made_up = call("call_2", "a::b::c", '{"city": "Paris"}')
        alone = [
            ask,
            {"role": "assistant", "content": "", "tool_calls": [made_up]},
            result("call_2", "unknown tool"),
        ]
        between = [
            ask,
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Look it up.",
                "tool_calls": [
                    call("call_1", "weather", '{"city": "Paris"}'),
                    made_up,
                    call("call_3", "time", "{}"),
                ],
            },
            result("call_1", "Sunny"),
            result("call_2", "unknown tool"),
            result("call_3", "noon"),
        ]
        later = [
            *between,
            {"role": "assistant", "content": "Sunny at noon."},
            {"role": "user", "content": "And Rome?"},
        ]
        tools = [
            {
                "type": "function",
                "function": {"name": name, "parameters": SCHEMAS[name]},
            }
            for name in SCHEMAS
        ]
        tokenizer, frontend = deepseek_tokenizer(), self.frontend()
        text = (
            'Incomplete tool call:\n{"name": "a::b::c", "arguments": {"city": "Paris"}}'
        )
        for name, messages in (
            ("alone", alone),
            ("between calls", between),
            ("later turn", later),
        ):
            body = fixtures.chat_body(messages=messages, tools=tools)
            with self.subTest(conversation=name):
                # The template's own refusal.
                with self.assertRaisesRegex(
                    TemplateError, "Tool name must not contain '::': b::c"
                ):
                    render_chat_template(
                        tokenizer,
                        messages,
                        {
                            "tools": tools,
                            "add_generation_prompt": True,
                            "tokenize": False,
                        },
                    )
                prompt = frontend.apply_template(body, deadline=fixtures.FOREVER)
                self.assertEqual(prompt.count(text), 1, prompt)
                self.assertNotIn('invoke name="a::b::c"', prompt)
                frontend.prepare(body, deadline=fixtures.FOREVER)
                frontend.count_tokens(body, deadline=fixtures.FOREVER)
        # The turn's other calls stay calls, in their order around it.
        prompt = frontend.apply_template(
            fixtures.chat_body(messages=between, tools=tools),
            deadline=fixtures.FOREVER,
        )
        self.assertTrue(
            prompt.endswith(
                "<｜User｜>Weather in Paris?<｜Assistant｜><think>Look it up.</think>"
                + dsml_calls(("weather", {"city": "Paris"}))
                + f"<｜end▁of▁sentence｜></think>{text}<｜end▁of▁sentence｜></think>"
                + dsml_calls(("time", {}))
                + "<｜end▁of▁sentence｜><｜User｜><tool_result>Sunny</tool_result>\n\n"
                "<tool_result>unknown tool</tool_result>\n\n"
                "<tool_result>noon</tool_result><｜Assistant｜><think>"
            ),
            prompt,
        )

    def test_ordinary_requests_render_and_encode_as_without_the_name_rule(self):
        # The names the template takes, a namespace's among them, in tools
        # and in calls of them.
        names = ("weather", "ns::weather", "a:::b", "lookup::")
        tools = [
            {
                "type": "function",
                "function": {"name": name, "parameters": SCHEMAS["weather"]},
            }
            for name in names
        ]
        calls = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": '{"city": "Paris"}'},
            }
            for index, name in enumerate(names)
        ]
        body = fixtures.chat_body(
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Look it up.",
                    "tool_calls": calls,
                },
                *(
                    {"role": "tool", "tool_call_id": call["id"], "content": "Sunny"}
                    for call in calls
                ),
                {"role": "assistant", "content": "Sunny.", "reasoning_content": "Hm"},
                {"role": "user", "content": "And Rome?"},
            ],
            tools=tools,
        )

        def prompt(app, request):
            """The request's prompt text and tokens."""
            return (
                app.apply_template(request, deadline=fixtures.FOREVER),
                app.prepare(request, deadline=fixtures.FOREVER).prompt_tokens,
            )

        frontend = self.frontend()
        unruled = self.frontend(
            call_syntax=replace(DEEPSEEK_DSML, namespace_separator=None)
        )
        for effort in (None, *REASONING_EFFORTS):
            request = body | {"reasoning_effort": effort}
            with self.subTest(effort=effort):
                self.assertEqual(prompt(frontend, request), prompt(unruled, request))

    def test_text_quoting_the_image_placeholder_renders_as_it_is(self):
        # The template refuses a message whose text holds its image
        # placeholder. A page, file or tool result an agent read may quote
        # it, and the model may quote it back in its answer or its reasoning;
        # every later request of the conversation renders that text again.
        quote = f"The page reads {DEEPSEEK_IMAGE} here."
        ask = {"role": "user", "content": "Read the page."}
        then = {"role": "user", "content": "Go on."}
        call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": {"city": "Paris"}},
                }
            ],
        }
        result = {"role": "tool", "tool_call_id": "call_1", "content": quote}
        done = {"role": "assistant", "content": "Done."}
        conversations = {
            "system": [{"role": "system", "content": quote}, ask],
            "user": [{"role": "user", "content": quote}],
            "tool": [ask, call, result],
            "later turn": [ask, call, result, done, then],
            "assistant": [ask, {**done, "content": quote}, then],
            "reasoning": [ask, {**done, "reasoning_content": quote}, then],
        }
        # Tools keep the history's reasoning.
        tools = [
            {
                "type": "function",
                "function": {"name": "weather", "parameters": SCHEMAS["weather"]},
            }
        ]
        tokenizer, frontend = deepseek_tokenizer(), self.frontend()
        for name, messages in conversations.items():
            body = fixtures.chat_body(messages=messages, tools=tools)
            with self.subTest(conversation=name):
                # The template's own refusal.
                with self.assertRaisesRegex(TemplateError, "image special token"):
                    render_chat_template(
                        tokenizer,
                        messages,
                        {
                            "tools": tools,
                            "add_generation_prompt": True,
                            "tokenize": False,
                        },
                    )
                prompt = frontend.apply_template(body, deadline=fixtures.FOREVER)
                self.assertEqual(prompt.count(quote), 1, prompt)
                self.assertNotIn(PLACEHOLDER_RENDER_MARKER, prompt)
                frontend.prepare(body, deadline=fixtures.FOREVER)

    def test_ordinary_requests_render_and_encode_as_without_the_marker(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "weather", "parameters": SCHEMAS["weather"]},
            }
        ]
        bodies = (
            fixtures.chat_body(),
            fixtures.chat_body(
                messages=[
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
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
                    {
                        "role": "assistant",
                        "content": "Sunny.",
                        "reasoning_content": "Hm",
                    },
                    {"role": "system", "content": "Answer in French."},
                    {"role": "user", "content": "And Rome?"},
                ],
                tools=tools,
            ),
        )

        def prompt(app, request):
            """The request's prompt text and tokens."""
            return (
                app.apply_template(request, deadline=fixtures.FOREVER),
                app.prepare(request, deadline=fixtures.FOREVER).prompt_tokens,
            )

        frontend, unmarked = self.frontend(), self.frontend(refused_placeholder=None)
        for index, body in enumerate(bodies):
            for effort in (None, *REASONING_EFFORTS):
                request = body | {"reasoning_effort": effort}
                with self.subTest(body=index, effort=effort):
                    self.assertEqual(
                        prompt(frontend, request), prompt(unmarked, request)
                    )

    def test_a_judgment_of_text_quoting_the_image_placeholder_is_scored(self):
        frontend, unmarked = self.frontend(), self.frontend(refused_placeholder=None)

        def prompt(app, state):
            job, _ = app.prepare_judgment(
                fixtures.judgment_body(state=state),
                deadline=fixtures.FOREVER,
                disconnected=fixtures.stays_connected,
            )
            return job.prompt_tokens, job.prompt_sha256

        ordinary = {"evidence": "the sky is blue"}
        self.assertEqual(prompt(frontend, ordinary), prompt(unmarked, ordinary))
        tokens, _ = prompt(frontend, {"evidence": f"a page reads {DEEPSEEK_IMAGE}"})
        self.assertTrue(tokens)

    def test_prompt_tokens_are_reused_after_the_last_assistant_turn(self):
        # Splits before the byte-level BPE, as DeepSeek's tokenizer has, and
        # the GGUF tokenizer's byte-level post-processor.
        tokenizer = deepseek_tokenizer()
        backend = tokenizer.backend_tokenizer
        backend.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(r"\d{1,3}"), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        backend.post_processor = processors.ByteLevel(trim_offsets=False)
        cache = PromptTokenizer(tokenizer, DEEPSEEK.message_boundary)
        self.assertTrue(cache.enabled)
        first = "<｜User｜>" + "Paris 12345 " * 400 + "<｜Assistant｜></think>"
        later = first + "Sunny.<｜end▁of▁sentence｜><｜User｜>Rome?<｜Assistant｜>"
        for text in (first, later):
            self.assertEqual(
                cache.encode(text),
                tokenizer(text, add_special_tokens=False)["input_ids"],
            )
        self.assertEqual(cache.stats()["hits"], 1)


class DeepSeekServingTests(fixtures.HarnessTestCase):
    """DeepSeek's output through the server: the reasoning its generation
    prompt opened and its block of calls, streamed or not, read from the
    tokens the engine returns."""

    FRAGMENTS = {
        41: "Check the weather.",
        42: "</think>",
        43: "Let me look.",
        44: f'\n\n<{DSML} calls>\n<{DSML} invoke name="wea',
        45: f'ther">\n<{DSML} parameter name="ci',
        46: 'ty" string="true">Par',
        47: (
            f'is</{DSML} parameter>\n<{DSML} parameter name="days" string="false">'
            f"3</{DSML} parameter>\n</{DSML} invoke>\n</{DSML} calls>"
        ),
        # A call of a name the template refuses, which no tool may hold.
        49: dsml_calls(("a::b::c", {"city": "Paris"})),
    }
    # DeepSeek's format over the fake vocabulary's IDs.
    CHAT_FORMAT = replace(DEEPSEEK, think_end_id=42, call_open_id=48)

    def harness(self, *plans, **options):
        # Prompts render with DeepSeek's template; the engine's tokens decode
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
        output = [[41], [42], [43], [44], [45], [46], [47]]
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

    def test_a_required_call_is_constrained_in_dsml(self):
        factory = fixtures.FakeConstraintFactory()
        harness = self.harness(
            fixtures.Plan([[41], [42], [44], [45], [46], [47]]),
            constraint_factory=factory,
        )
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(tool_choice="required")
        )
        self.assertEqual(status, 200, payload)
        (grammar,) = factory.grammars
        main = json.loads(grammar)["grammars"][0]["lark_grammar"]
        calls_open = json.dumps(f"<{DSML} calls>\n")
        calls_close = json.dumps(f"</{DSML} calls>")
        self.assertIn(
            f'start: think WS {calls_open} ((call_0) "\\n")+ {calls_close} WS\n', main
        )
        self.assertIn("think: TEXT <[42]>\n", main)

    def test_a_tool_result_quoting_the_image_placeholder_is_served(self):
        harness = self.harness(fixtures.Plan([[41], [42], [43]]))
        messages = [
            {"role": "user", "content": "Read the page."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": f"The page reads {DEEPSEEK_IMAGE} here.",
            },
        ]
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(messages=messages)
        )
        self.assertEqual(status, 200, payload)
        message = json.loads(payload)["choices"][0]["message"]
        self.assertEqual(message["content"], "Let me look.")

    def test_a_tool_whose_name_holds_dsml_markup_is_refused(self):
        harness = self.harness()
        name = f"read{DSML}file"
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            fixtures.chat_body(
                tools=[{"type": "function", "function": {"name": name}}]
            ),
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(
            json.loads(payload)["error"]["message"],
            f"tool name {name} must not contain '{DSML}', "
            "the markup of this model's tool calls",
        )

    def test_a_call_of_a_name_the_template_refuses_is_returned_and_sent_back(self):
        # Without a grammar the model calls a tool by any name: the server
        # returns the call as the model wrote it, and the conversation that
        # sends it back goes on, in each API, its call kept as text.
        harness = self.harness(
            fixtures.Plan([[41], [42], [43], [49]]),
            # Each later request but count_tokens generates.
            *(fixtures.Plan([[41], [42], [43]]) for _ in range(3)),
        )
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body()
        )
        self.assertEqual(status, 200, payload)
        message = json.loads(payload)["choices"][0]["message"]
        (call,) = message["tool_calls"]
        self.assertEqual(
            (call["function"]["name"], call["function"]["arguments"]),
            ("a::b::c", '{"city":"Paris"}'),
        )
        ask, failed = "Weather in Paris?", "unknown tool a::b::c"
        chat = [
            {"role": "user", "content": ask},
            message,
            {"role": "tool", "tool_call_id": call["id"], "content": failed},
        ]
        responses = [
            {"type": "message", "role": "user", "content": ask},
            {
                "type": "function_call",
                "call_id": call["id"],
                "name": "a::b::c",
                "arguments": call["function"]["arguments"],
            },
            {"type": "function_call_output", "call_id": call["id"], "output": failed},
        ]
        messages = [
            {"role": "user", "content": ask},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": "a::b::c",
                        "input": {"city": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": failed,
                    }
                ],
            },
        ]
        function = {
            "type": "function",
            "name": "weather",
            "parameters": SCHEMAS["weather"],
        }
        tool = {"name": "weather", "input_schema": SCHEMAS["weather"]}
        for path, body in (
            ("/v1/chat/completions", self.body(messages=chat)),
            (
                "/v1/responses",
                fixtures.responses_body(input=responses, tools=[function]),
            ),
            ("/v1/messages", fixtures.anthropic_body(messages=messages, tools=[tool])),
            (
                "/v1/messages/count_tokens",
                fixtures.anthropic_body(messages=messages, tools=[tool]),
            ),
        ):
            with self.subTest(path=path):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 200, payload)
                rendered, _ = harness.tokenizer.templates[-1]
                self.assertIn(
                    {
                        "role": "assistant",
                        "content": 'Incomplete tool call:\n{"name": "a::b::c", '
                        '"arguments": {"city": "Paris"}}',
                    },
                    rendered,
                )

    def test_a_tool_whose_name_holds_the_namespace_separator_twice_is_refused(self):
        # Before any template renders, in each API, naming the whole name.
        harness = self.harness()
        tool = {"name": "a::b::c", "input_schema": {"type": "object"}}
        namespace = {
            "type": "namespace",
            "name": "a::b",
            "tools": [{"type": "function", "name": "c::d"}],
        }
        for path, body, name in (
            (
                "/v1/chat/completions",
                fixtures.chat_body(
                    tools=[{"type": "function", "function": {"name": "a::b::c"}}]
                ),
                "a::b::c",
            ),
            (
                "/v1/responses",
                fixtures.responses_body(
                    tools=[{"type": "function", "name": "a::b::c"}]
                ),
                "a::b::c",
            ),
            # A namespace's function goes by namespace__name.
            ("/v1/responses", fixtures.responses_body(tools=[namespace]), "a::b__c::d"),
            ("/v1/messages", fixtures.anthropic_body(tools=[tool]), "a::b::c"),
            (
                "/v1/messages/count_tokens",
                fixtures.anthropic_body(tools=[tool]),
                "a::b::c",
            ),
        ):
            with self.subTest(path=path, name=name):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 400, payload)
                self.assertEqual(
                    json.loads(payload)["error"]["message"],
                    f"tool name {name} must not contain '::' more than once, "
                    "the separator of this model's tool namespaces",
                )
        self.assertEqual(harness.tokenizer.templates, [])


if __name__ == "__main__":
    unittest.main()
