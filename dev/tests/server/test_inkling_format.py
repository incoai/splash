"""Inkling 2's chat format: prompts rendered as its template renders them,
with no generation prompt and the reasoning effort as a level; its output's
blocks read into reasoning, content and calls, streamed or not; its calls
constrained in their JSON frames; and scoring after a text block's
opening."""

import json
import unittest
from dataclasses import replace

from llguidance import LLMatcher, LLTokenizer
from llguidance.hf import from_tokenizer

from dev.tests import server_fixtures as fixtures
from dev.tests.inkling_fixtures import TEMPLATE, inkling_tokenizer
from dev.tests.tool_output import streamed_arguments, streamed_text, tool_policy
from install import families
from server import backend as backend_api
from server import constraints, tool_schema
from server import output as model_output
from server import protocol as wire
from server.chat_format import INKLING, INKLING_FRAMES, INKLING_JSON, REASONING_EFFORTS
from server.chat_templates import NATIVE, ChatTemplateError, ChatTemplates
from server.dialects import scoring
from server.errors import APIError
from server.tokenization import PromptTokenizer

HEADER = "<|message_model|>"
THINK = HEADER + "<|content_thinking|>"
TEXT = HEADER + "<|content_text|>"
CALL = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
TURN_END = "<|content_model_end_sampling|>"
WEATHER = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
        "units": {"enum": ["metric", "imperial"]},
    },
    "required": ["city"],
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Weather of a city.",
            "parameters": WEATHER,
        },
    },
    {"type": "function", "function": {"name": "time"}},
]


def call(name, arguments, *, spelled=None):
    """A call block as Inkling's template writes it: the name, its JSON
    frame, and the object of the name and the arguments, compact; `spelled`
    writes the object's text instead."""
    if spelled is None:
        spelled = json.dumps(
            {"name": name, "args": arguments},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return f"{HEADER}{name}{CALL}{spelled}{END}"


def read(text, policy=None, *, size=None, structured=False, incomplete=False):
    """Read `text`, Inkling's output, as a chat job reads it, in pieces of
    `size` characters: its reasoning, the content and calls the projector
    finishes with, every event it published, and whether it ended inside its
    reasoning."""
    splitter = model_output.reasoning_splitter(False, policy is not None, INKLING)
    projector = model_output.tool_call_projector(policy, "r", structured, INKLING)
    reasoning, events = [], []

    def take(pairs):
        for field, piece in pairs:
            if field == "content":
                events.extend(projector.put(piece))
            else:
                reasoning.append(piece)

    size = size or len(text) or 1
    for offset in range(0, len(text), size):
        take(splitter.put(text[offset : offset + size]))
    take(splitter.finish())
    content, calls, owed = projector.finish(incomplete)
    return "".join(reasoning), content, calls, events + owed, splitter.reasoning


def frontend(tokenizer=None, **options):
    tokenizer = tokenizer or inkling_tokenizer()
    return fixtures.make_frontend(
        tokenizer,
        None,
        "test-model",
        4096,
        10,
        2,
        vision=False,
        chat_format=INKLING,
        **options,
    )


class InklingPromptTests(unittest.TestCase):
    """A request renders byte for byte as the template renders the same
    conversation for a client that hands it the messages as the template
    takes them (a call's arguments an object), with the request's effort."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = inkling_tokenizer()
        cls.frontend = frontend(cls.tokenizer)

    def render(self, body):
        return self.frontend.apply_template(
            {"model": "test-model"} | body, deadline=fixtures.FOREVER
        )

    def reference(self, messages, **options):
        """The template's own rendering, as transformers renders it."""
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **options)

    def test_the_layout_is_the_template_s(self):
        # Tools come first, sorted and compact; the effort's system block
        # follows the leading system message; each content part, reasoning
        # and call is a block of its own, a call's arguments sorted; an
        # assistant turn ends with the stop token; a tool result names its
        # tool by its call; and the prompt ends with the last block.
        body = {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "reasoning_content": "Look it up.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"units": "metric", "city": "Paris"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            ],
            "tools": TOOLS,
        }
        declared = json.dumps(
            [
                {
                    "description": "Weather of a city.",
                    "name": "weather",
                    "parameters": WEATHER,
                    "type": "function",
                },
                {
                    "description": "",
                    "name": "time",
                    "parameters": {},
                    "type": "function",
                },
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(
            self.render(body),
            f"<|message_system|>tool_declare<|content_xml|>{declared}{END}"
            f"<|message_system|><|content_text|>Be brief.{END}"
            f"<|message_system|><|content_text|>Thinking effort level: 0.9{END}"
            f"<|message_user|><|content_text|>Weather in Paris?{END}"
            f"{THINK}Look it up.{END}{TEXT}Checking.{END}"
            + call("weather", {"city": "Paris", "units": "metric"})
            + f"{TURN_END}<|message_tool|>weather<|content_text|>Sunny{END}",
        )

    def test_requests_render_as_the_template_renders_them(self):
        history = [
            {"role": "user", "content": "Weather in Paris and Brno?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Two cities: two calls.",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"city": "Paris", "days": 2},
                        },
                    },
                    {
                        "id": "b",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"city": "Brno ěščř 😀", "days": 1},
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "Sunny"},
            {"role": "tool", "tool_call_id": "b", "content": "Rain"},
            {"role": "assistant", "content": "Sunny in Paris, rain in Brno."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And tomorrow?"},
                    {"type": "text", "text": " Briefly."},
                ],
            },
        ]

        def sent(messages):
            # As a client sends them: a call's arguments as JSON text.
            return [
                message
                | {
                    "tool_calls": [
                        tool_call
                        | {
                            "function": tool_call["function"]
                            | {
                                "arguments": json.dumps(
                                    tool_call["function"]["arguments"]
                                )
                            }
                        }
                        for tool_call in message["tool_calls"]
                    ]
                }
                if "tool_calls" in message
                else message
                for message in messages
            ]

        conversations = (
            ([{"role": "user", "content": "hello"}], {}),
            ([{"role": "system", "content": "Be kind."}, *history], {"tools": TOOLS}),
            (history, {"tools": TOOLS, "reasoning_effort": "low"}),
            (history, {"reasoning_effort": "none"}),
            (history[:1], {"reasoning_effort": 0.35}),
        )
        for messages, options in conversations:
            with self.subTest(messages=len(messages), **options):
                self.assertEqual(
                    self.render({"messages": sent(messages), **options}),
                    self.reference(messages, **options),
                )

    def test_efforts_render_as_the_template_s_levels(self):
        levels = [
            ("none", "0"),
            ("minimal", "0.1"),
            ("low", "0.2"),
            ("medium", "0.7"),
            ("high", "0.9"),
            ("xhigh", "0.99"),
            ("max", "0.99"),
            (0, "0"),
            (0.0, "0"),
            (0.5, "0.5"),
            (0.99, "0.99"),
        ]
        self.assertLessEqual(set(REASONING_EFFORTS), {effort for effort, _ in levels})
        messages = [{"role": "user", "content": "hello"}]
        for effort, level in levels:
            with self.subTest(effort=effort):
                self.assertIn(
                    f"Thinking effort level: {level}{END}",
                    self.render({"messages": messages, "reasoning_effort": effort}),
                )
        # The template's own default, high's; thinking off is effort none.
        self.assertIn("level: 0.9<", self.render({"messages": messages}))
        self.assertIn(
            "level: 0<",
            self.render(
                {
                    "messages": messages,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            ),
        )
        for effort in (1.0, -0.1, True, "huge", [0.5]):
            with self.subTest(effort=effort):
                with self.assertRaises(APIError) as raised:
                    self.render({"messages": messages, "reasoning_effort": effort})
                self.assertEqual(raised.exception.status, 400)

    def test_a_server_s_effort_and_default_render_as_levels(self):
        messages = [{"role": "user", "content": "hello"}]
        default = frontend(self.tokenizer, default_reasoning_effort="low")
        self.assertIn(
            "level: 0.2<",
            default.apply_template(
                {"model": "test-model", "messages": messages},
                deadline=fixtures.FOREVER,
            ),
        )
        every = frontend(self.tokenizer, reasoning_effort="medium")
        self.assertIn(
            "level: 0.7<",
            every.apply_template(
                {"model": "test-model", "messages": messages, "reasoning_effort": 0.1},
                deadline=fixtures.FOREVER,
            ),
        )

    def test_models_report_the_template_s_efforts(self):
        self.assertEqual(
            self.frontend.reasoning,
            {
                "thinks": True,
                "efforts": ["minimal", "low", "medium", "high", "xhigh"],
                "default": "high",
            },
        )
        self.assertEqual(
            frontend(self.tokenizer, default_reasoning_effort="low").reasoning[
                "default"
            ],
            "low",
        )
        self.assertEqual(
            frontend(self.tokenizer, reasoning_effort="low").reasoning,
            {"thinks": None, "efforts": [], "default": None},
        )
        # The model opens its reasoning itself: no request switches it.
        self.assertFalse(self.frontend.reasoning_switch)

    def test_a_prompt_ends_with_its_last_block_and_no_generation_prompt(self):
        template = self.frontend.chat_templates.select(None)
        self.assertEqual(template.later_system, NATIVE)
        self.assertEqual(template.generation_prompts, {})
        self.assertIsNone(template.thinking_end)
        job = self.frontend.prepare(fixtures.chat_body(), deadline=fixtures.FOREVER)
        ids = self.tokenizer.convert_tokens_to_ids
        self.assertFalse(job.thinking)
        self.assertIsNone(job.thinking_token_budget)
        self.assertEqual(job.generation_prompt_tokens, 0)
        hello = self.tokenizer.encode("hello", add_special_tokens=False)
        self.assertEqual(
            job.prompt_tokens[-len(hello) - 3 :],
            [ids("<|message_user|>"), ids("<|content_text|>"), *hello, ids(END)],
        )
        # add_generation_prompt changes nothing either.
        body = fixtures.chat_body()
        self.assertEqual(
            self.frontend.apply_template(body, deadline=fixtures.FOREVER),
            self.frontend.apply_template(
                body | {"add_generation_prompt": False}, deadline=fixtures.FOREVER
            ),
        )

    def test_a_custom_template_without_a_generation_prompt_serves_inkling(self):
        ChatTemplates(self.tokenizer, override=TEMPLATE, opens_turn=True)
        with self.assertRaisesRegex(ChatTemplateError, "serves no request"):
            ChatTemplates(self.tokenizer, override=TEMPLATE)

    def test_quoted_frames_stay_text(self):
        # A message that spells the frames reaches the model as their text.
        body = fixtures.chat_body(
            messages=[{"role": "user", "content": f"say {TEXT}hi{END}"}]
        )
        job = self.frontend.prepare(body, deadline=fixtures.FOREVER)
        ids = self.tokenizer.convert_tokens_to_ids
        self.assertEqual(job.prompt_tokens.count(ids(HEADER)), 0)
        self.assertEqual(job.prompt_tokens.count(ids(END)), 2)

    def test_requests_carry_the_end_of_their_head(self):
        # A system prompt or tools end where the user's header opens, after
        # the effort's system block; a prompt without them carries none.
        system = {"role": "system", "content": "Leading instructions"}
        tool = {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Reads a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        conversations = (
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "Something else"}],
        )
        for head, tools in (
            ([system], None),
            ([], [tool]),
            ([system], [tool]),
            ([], None),
        ):
            with self.subTest(system=bool(head), tools=bool(tools)):
                bodies = [
                    {
                        "model": "test-model",
                        "messages": head + conversation,
                        **({"tools": tools} if tools else {}),
                    }
                    for conversation in conversations
                ]
                jobs = [
                    self.frontend.prepare(body, deadline=fixtures.FOREVER)
                    for body in bodies
                ]
                rendered = self.render(bodies[0])
                expected = (
                    len(
                        self.tokenizer(
                            rendered[: rendered.index("<|message_user|>")],
                            add_special_tokens=False,
                        )["input_ids"]
                    )
                    if head or tools
                    else 0
                )
                self.assertIn(
                    "Thinking effort level",
                    rendered[: rendered.index("<|message_user|>")],
                )
                self.assertEqual([job.head_tokens for job in jobs], [expected] * 2)

    def test_prompt_tokens_are_reused_after_the_last_block(self):
        cache = PromptTokenizer(
            self.tokenizer, INKLING.message_boundary, fixtures.NATIVE_CONTEXT
        )
        self.assertTrue(cache.enabled)
        first = self.render(
            {"messages": [{"role": "user", "content": "the weather " * 600}]}
        )
        later = self.render(
            {
                "messages": [
                    {"role": "user", "content": "the weather " * 600},
                    {"role": "assistant", "content": "Sunny."},
                    {"role": "user", "content": "And Paris?"},
                ]
            }
        )
        for text in (first, later):
            self.assertEqual(
                cache.encode(text),
                self.tokenizer(text, add_special_tokens=False)["input_ids"],
            )
        self.assertEqual(cache.stats()["hits"], 1)


class InklingScoringTests(unittest.TestCase):
    """Scores are read where Inkling answers without reasoning: effort none,
    then its turn's text block opened, as its template renders a turn
    without reasoning, the answer's first token next."""

    @classmethod
    def setUpClass(cls):
        cls.frontend = frontend()
        cls.ids = cls.frontend.tokenizer.convert_tokens_to_ids

    def test_judgments_and_system_one_read_after_a_text_block_s_opening(self):
        judgment, _ = scoring.prepare_judgment(
            self.frontend,
            fixtures.judgment_body(),
            deadline=fixtures.FOREVER,
            disconnected=fixtures.stays_connected,
        )
        systemone = scoring.prepare_systemone(
            self.frontend,
            {"model": "test-model", "state": {}, "questions": {"q": {"type": "noul"}}},
            deadline=fixtures.FOREVER,
            disconnected=fixtures.stays_connected,
        )
        for job in (judgment, systemone[0][2]):
            self.assertEqual(
                list(job.prompt_tokens[-3:]),
                [self.ids(END), self.ids(HEADER), self.ids("<|content_text|>")],
            )
            self.assertEqual(len(job.score_tokens), 2)
            text = self.frontend.tokenizer.decode(job.prompt_tokens)
            self.assertIn(f"Thinking effort level: 0{END}", text)
            self.assertNotIn("<|content_thinking|>", text)


class InklingOutputTests(unittest.TestCase):
    """The blocks of Inkling's output read into reasoning, content and calls,
    alike in any pieces, and what streams adds up to what the answer
    holds."""

    SIZES = (1, 2, 3, 5, 8, 13, None)

    def setUp(self):
        self.policy = tool_policy({"weather": WEATHER, "time": {}})

    def assert_reads_alike(self, text, expected, policy="tools", **options):
        policy = self.policy if policy == "tools" else policy
        for size in self.SIZES:
            with self.subTest(size=size):
                reasoning, content, calls, events, _ = read(
                    text, policy, size=size, **options
                )
                self.assertEqual(
                    (
                        reasoning,
                        content,
                        [
                            (c["function"]["name"], c["function"]["arguments"])
                            for c in calls
                        ],
                    ),
                    expected,
                )
                self.assertEqual(streamed_text(events), content)
                self.assertEqual(
                    streamed_arguments(events),
                    "".join(c["function"]["arguments"] for c in calls),
                )
        return events

    def test_reasoning_text_and_parallel_calls(self):
        text = (
            f"{THINK}Two <cities>.{END}{TEXT}Checking both.{END}"
            + call("weather", {"city": "Paris", "days": 2})
            + call("weather", {"city": "Brno ěščř 😀"})
        )
        events = self.assert_reads_alike(
            text,
            (
                "Two <cities>.",
                "Checking both.",
                [
                    ("weather", '{"city":"Paris","days":2}'),
                    ("weather", '{"city":"Brno ěščř 😀"}'),
                ],
            ),
        )
        headers = [value for kind, value in events if kind == "tool" and "id" in value]
        self.assertEqual(
            [(h["index"], h["id"], h["function"]["name"]) for h in headers],
            [(0, "call_r_0", "weather"), (1, "call_r_1", "weather")],
        )
        for _, _, calls, _, _ in (read(text, self.policy),):
            for tool_call in calls:
                json.loads(tool_call["function"]["arguments"])

    def test_reasoning_may_come_again_and_frames_may_carry_names(self):
        # The model's own reader joins each kind; a name between a header and
        # a content token is dropped, a content token opens its block without
        # a header, and a tool error or a text-form call is text.
        text = (
            f"{THINK}first{END}{TEXT}One.{END}"
            f"{HEADER}model<|content_thinking|> second{END}"
            f"<|content_text|> Two.{END}"
            f"{HEADER}<|content_tool_error|> Three.{END}"
        )
        self.assert_reads_alike(text, ("first second", "One. Two. Three.", []))

    def test_an_answer_without_reasoning_and_without_tools(self):
        self.assert_reads_alike(f"{TEXT}Just text.{END}", ("", "Just text.", []))
        # Without tools a call is text, as the model wrote it.
        self.assert_reads_alike(
            call("time", {}),
            ("", call("time", {}), []),
            policy=None,
        )

    def test_a_call_whose_frame_names_none_is_named_by_its_object(self):
        self.assert_reads_alike(
            f'{HEADER}{CALL}{{"name":"weather","args":{{"city":"Rome"}}}}{END}'
            f'{CALL}{{"args":{{"city":"Oslo"}},"name":"weather"}}{END}',
            ("", "", [("weather", '{"city":"Rome"}'), ("weather", '{"city":"Oslo"}')]),
        )
        # A call that names no function is left out.
        self.assert_reads_alike(
            f'{HEADER}{CALL}{{"args":{{"city":"Rome"}}}}{END}{TEXT}ok{END}',
            ("", "ok", []),
        )

    def test_arguments_are_written_compactly_and_as_valid_json(self):
        spelled = (
            '{ "name" : "weather" , "args" : { "city" : "Pa\\u0072is\\n\\ud83d\\ude00",'
            ' "days" : 3 , "extra" : [ true , null , -1.5e2 , { } ] } }'
        )
        self.assert_reads_alike(
            call("weather", None, spelled=spelled),
            (
                "",
                "",
                [
                    (
                        "weather",
                        '{"city":"Pa\\u0072is\\n\\ud83d\\ude00","days":3,'
                        '"extra":[true,null,-1.5e2,{}]}',
                    )
                ],
            ),
        )
        # A lone surrogate is U+FFFD, a raw control character escaped.
        self.assert_reads_alike(
            call(
                "weather",
                None,
                spelled='{"name":"weather","args":{"city":"a\\udc00\tb"}}',
            ),
            ("", "", [("weather", '{"city":"a\\ufffd\\tb"}')]),
        )

    def test_arguments_that_are_no_json_object_end_where_they_fail(self):
        for spelled, arguments in (
            ('{"name":"weather","args":[1,2]}', "{}"),
            ('{"name":"weather","args":"city"}', "{}"),
            (
                '{"name":"weather","args":{"city":"Paris","days":tomorrow}}',
                '{"city":"Paris"}',
            ),
            ('{"name":"weather","args":{"city":"Paris"} trailing', '{"city":"Paris"}'),
            ("not json at all", "{}"),
        ):
            with self.subTest(spelled=spelled):
                _, _, calls, _, _ = read(
                    call("weather", None, spelled=spelled), self.policy
                )
                self.assertEqual(calls[0]["function"]["arguments"], arguments)

    def test_a_cut_call_keeps_the_arguments_it_has(self):
        text = f'{TEXT}Wait.{END}{HEADER}weather{CALL}{{"name":"weather","args":{{"city":"Par'
        for size in self.SIZES:
            with self.subTest(size=size):
                _, content, calls, events, _ = read(
                    text, self.policy, size=size, incomplete=True
                )
                self.assertEqual(content, "Wait.")
                self.assertEqual(calls[0]["function"]["arguments"], '{"city":"Par')
                self.assertEqual(streamed_arguments(events), '{"city":"Par')
                # Output the model ended there closes the call validly.
                _, _, calls, _, _ = read(text, self.policy, size=size)
                self.assertEqual(calls[0]["function"]["arguments"], '{"city":"Par"}')
        # A call cut while it names its function is none.
        for cut in (f"{HEADER}wea", HEADER):
            _, content, calls, _, _ = read(
                f"{TEXT}a{END}{cut}", self.policy, incomplete=True
            )
            self.assertEqual((content, calls), ("a", []))

    def test_output_cut_inside_its_reasoning_ends_in_it(self):
        reasoning, content, _, _, inside = read(f"{THINK}still thinking", None)
        self.assertEqual((reasoning, content, inside), ("still thinking", "", True))
        _, _, _, _, inside = read(f"{THINK}done{END}{TEXT}ok", None)
        self.assertFalse(inside)

    def test_a_structured_answer_is_its_json(self):
        self.assert_reads_alike(
            f'{THINK}hm{END}{TEXT}{{"a": [1, "<|x|>"]}}{END}',
            ("hm", '{"a": [1, "<|x|>"]}', []),
            policy=None,
            structured=True,
        )

    def test_reasoning_tokens_are_those_inside_reasoning_blocks(self):
        tokenizer = inkling_tokenizer()
        streamer = backend_api.CallbackStreamer(tokenizer, lambda _text: None)
        text = f"{THINK}the abc{END}{TEXT}ab{END}{THINK}the{END}" + call("time", {})
        streamer.put_tokens(tokenizer.encode(text, add_special_tokens=False))
        thinking = len(tokenizer.encode(" the abc", add_special_tokens=False)) - 1
        expected = len(tokenizer.encode("the abc", add_special_tokens=False)) + len(
            tokenizer.encode("the", add_special_tokens=False)
        )
        self.assertGreater(thinking, 0)
        self.assertEqual(
            streamer.count_reasoning_tokens(INKLING, False, True), expected
        )
        streamer.end()
        self.assertEqual("".join(streamer.emitted), text)


class InklingGrammarTests(unittest.TestCase):
    """Constrained output is written as Inkling frames it: optional
    reasoning blocks, then a call's JSON frame or the answer's text block,
    each frame its token."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = inkling_tokenizer()
        cls.guidance = from_tokenizer(
            cls.tokenizer,
            n_vocab=201024,
            eos_token=[200006],
            slices=LLTokenizer.json_slices(),
        )

    def accepts(self, grammar, text):
        error = LLMatcher.validate_grammar(grammar, self.guidance)
        self.assertFalse(error, error)
        matcher = LLMatcher(self.guidance, grammar, log_level=0)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return (
            matcher.consume_tokens(tokens)
            and matcher.is_accepting()
            and matcher.consume_token(200006)
        )

    def policy(self, choice="auto", parallel=True, strict=True):
        tools = [
            {
                "type": "function",
                "function": {"name": "weather", "parameters": WEATHER}
                | ({"strict": True} if strict else {}),
            },
            {"type": "function", "function": {"name": "time"}},
        ]
        return tool_schema.normalize_tools(tools, choice, parallel)[1]

    def test_calls_follow_optional_reasoning_as_the_choice_says(self):
        thought = f"{THINK}hm{END}"
        text = f"{TEXT}Let me see.{END}"
        paris = call("weather", {"city": "Paris", "days": 2})
        named = {"type": "function", "function": {"name": "weather"}}
        cases = {
            # (choice, parallel): the outputs it takes, then those it refuses.
            ("required", True): (
                [thought + paris, paris + call("time", {})],
                [thought + text, thought + text + paris, ""],
            ),
            ("required", False): ([paris, thought + paris], [paris + paris]),
            ("auto", False): ([thought + text, text + paris, ""], [paris + paris]),
            ("auto", True): ([thought + text + paris + paris], [paris + text]),
            ("none", True): ([thought + text], [paris]),
        }
        for (choice, parallel), (taken, refused) in cases.items():
            grammar = tool_schema.tool_grammar(
                self.policy(choice, parallel), False, chat_format=INKLING
            )
            for output in taken:
                with self.subTest(choice=choice, parallel=parallel, output=output):
                    self.assertTrue(self.accepts(grammar, output))
            for output in refused:
                with self.subTest(choice=choice, parallel=parallel, refused=output):
                    self.assertFalse(self.accepts(grammar, output))
        grammar = tool_schema.tool_grammar(
            tool_schema.normalize_tools(TOOLS, named, True)[1],
            False,
            chat_format=INKLING,
        )
        self.assertTrue(self.accepts(grammar, paris))
        self.assertFalse(self.accepts(grammar, call("time", {})))

    def test_a_strict_tool_s_arguments_follow_its_schema(self):
        grammar = tool_schema.tool_grammar(
            self.policy("required"), False, chat_format=INKLING
        )
        for arguments, taken in (
            ({"city": "Paris"}, True),
            ({"city": "Paris", "units": "metric"}, True),
            ({"days": 2}, False),
            ({"city": "Paris", "units": "kelvin"}, False),
            ({"city": "Paris", "other": 1}, False),
        ):
            with self.subTest(arguments=arguments):
                self.assertIs(self.accepts(grammar, call("weather", arguments)), taken)
        # A tool that is not strict takes any object.
        grammar = tool_schema.tool_grammar(
            self.policy("required", strict=False), False, chat_format=INKLING
        )
        self.assertTrue(self.accepts(grammar, call("weather", {"other": [1]})))
        self.assertFalse(
            self.accepts(
                grammar, call("weather", None, spelled='{"name":"weather","args":1}')
            )
        )

    def test_a_response_schema_is_the_text_block_after_the_reasoning(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        }
        grammar = tool_schema.json_grammar(schema, False, chat_format=INKLING)
        self.assertTrue(self.accepts(grammar, f'{THINK}hm{END}{TEXT}{{"a":1}}{END}'))
        self.assertTrue(self.accepts(grammar, f'{TEXT}{{"a": 2}}{END}'))
        self.assertFalse(self.accepts(grammar, f'{TEXT}{{"a":"x"}}{END}'))
        self.assertFalse(self.accepts(grammar, '{"a":1}'))
        # Beside tools, the answer stands instead of the calls.
        grammar = tool_schema.tool_grammar(
            self.policy(), False, schema, chat_format=INKLING
        )
        self.assertTrue(self.accepts(grammar, f'{TEXT}{{"a":1}}{END}'))
        self.assertTrue(self.accepts(grammar, call("weather", {"city": "Rome"})))

    def test_what_the_grammar_writes_reads_back_as_written(self):
        grammar = tool_schema.tool_grammar(
            self.policy("required"), False, chat_format=INKLING
        )
        output = f"{THINK}hm{END}" + call("weather", {"city": "Paris", "days": 2})
        self.assertTrue(self.accepts(grammar, output))
        decoded = self.tokenizer.decode(
            self.tokenizer.encode(output, add_special_tokens=False),
            skip_special_tokens=True,
        )
        self.assertEqual(decoded, output)
        _, _, calls, _, _ = read(decoded, self.policy("required"))
        self.assertEqual(
            [(c["function"]["name"], c["function"]["arguments"]) for c in calls],
            [("weather", '{"city":"Paris","days":2}')],
        )

    def test_the_constraint_takes_each_strict_call_s_opening_and_ends_at_eos(self):
        ready = wire.ReadyEvent(4, 4096, False, 201024, 9, (200006,))
        factory = constraints.ConstraintFactory(
            constraints.ConstraintFactory.table(self.tokenizer, INKLING, 201024), ready
        )
        app = frontend(self.tokenizer, constraint_factory=factory)
        policy = self.policy("required")
        constraint = factory.create(
            tool_schema.tool_grammar(policy, False, chat_format=INKLING),
            prefixes=lambda: app._call_openings(policy, False),
        )
        output = call("weather", {"city": "Paris"})
        tokens = self.tokenizer.encode(output, add_special_tokens=False)
        constraint.commit(tokens)
        constraint.commit([200006])
        constraint.finish()


class InklingTokenizerContractTests(unittest.TestCase):
    def test_the_tokenizer_is_checked_against_inkling_s_format(self):
        tokenizer = inkling_tokenizer()
        constraints.validate_tokenizer(tokenizer, INKLING)
        constraints.check_capabilities(
            tokenizer, INKLING, wire.ReadyEvent(4, 4096, False, 201024, 9, (200006,))
        )
        # A frame the tokenizer decodes as nothing could not be read.
        backend = tokenizer.backend_tokenizer
        from tokenizers import AddedToken

        backend.add_special_tokens([AddedToken(HEADER, normalized=False, special=True)])
        with self.assertRaisesRegex(
            constraints.TokenizerContractError, "decodes as text"
        ):
            constraints.validate_tokenizer(tokenizer, INKLING)

    def test_the_family_serves_inkling_s_format(self):
        family = families.named("Inkling-2-Mini")
        self.assertEqual(
            (family.chat_format, family.draft_repo, family.vision),
            (INKLING, "unsloth/Inkling-2-Mini-GGUF", False),
        )
        self.assertEqual(
            family.sampling,
            families.SamplingDefaults(temperature=1.0, top_p=1.0, top_k=0),
        )

    def test_a_request_s_effort_reaches_the_model_and_switches_nothing(self):
        self.assertTrue(INKLING.reasoning_effort)
        self.assertFalse(INKLING.reasoning_switch)
        self.assertEqual(INKLING.call_syntax, INKLING_JSON)
        self.assertEqual(INKLING.frames, INKLING_FRAMES)


class InklingServingTests(fixtures.HarnessTestCase):
    """Inkling's output through the server, streamed and not, read from the
    tokens the engine returns."""

    FRAGMENTS = {
        40: HEADER,
        41: "<|content_thinking|>",
        42: "Check the weather.",
        43: END,
        44: "<|content_text|>",
        45: "Let me look.",
        46: "weat",
        47: "her" + CALL + '{"name":"wea',
        48: 'ther","args":{"ci',
        49: 'ty":"Par',
        50: 'is","days":3}}' + END,
    }
    # Inkling's format over the fake vocabulary's IDs.
    CHAT_FORMAT = replace(
        INKLING,
        frames=replace(
            INKLING_FRAMES, header_id=40, reasoning_id=41, close_id=43, text_id=44
        ),
        think_start_id=41,
        think_end_id=43,
        call_open_id=40,
    )
    OUTPUT = [
        [40],
        [41],
        [42],
        [43],
        [40],
        [44],
        [45],
        [43],
        [40],
        [46],
        [47],
        [48],
        [49],
        [50],
    ]

    def harness(self, *plans, **options):
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
        return fixtures.chat_body(tools=TOOLS[:1], **extra)

    def test_reasoning_text_and_a_call_read_from_the_output(self):
        harness = self.harness(fixtures.Plan(self.OUTPUT), fixtures.Plan(self.OUTPUT))
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
                (c["function"]["name"], c["function"]["arguments"])
                for c in message["tool_calls"]
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
        calls = [c for delta in deltas for c in delta.get("tool_calls") or ()]
        self.assertEqual(calls[0]["function"]["name"], "weather")
        self.assertEqual(
            "".join(c["function"].get("arguments", "") for c in calls),
            '{"city":"Paris","days":3}',
        )
        # The arguments streamed as the model wrote them, before the close.
        self.assertGreater(len([c for c in calls if c["function"].get("arguments")]), 2)

    def test_messages_read_the_same_blocks(self):
        harness = self.harness(fixtures.Plan(self.OUTPUT))
        body = fixtures.anthropic_body(
            tools=[{"name": "weather", "input_schema": WEATHER}],
            max_tokens=64,
        )
        status, _, payload = harness.request("POST", "/v1/messages", body)
        self.assertEqual(status, 200, payload)
        content = json.loads(payload)["content"]
        self.assertEqual(
            [block["type"] for block in content], ["thinking", "text", "tool_use"]
        )
        self.assertEqual(content[2]["input"], {"city": "Paris", "days": 3})

    def test_a_required_call_is_constrained_in_its_json_frame(self):
        factory = fixtures.FakeConstraintFactory()
        harness = self.harness(
            fixtures.Plan([[40], [46], [47], [48], [49], [50]]),
            constraint_factory=factory,
        )
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(tool_choice="required")
        )
        self.assertEqual(status, 200, payload)
        (grammar,) = factory.grammars
        main = json.loads(grammar)["grammars"][0]["lark_grammar"]
        self.assertIn("start: think* (call_0) (call_0)*\n", main)
        self.assertIn("think: <[40]> <[41]> TEXT <[43]>\n", main)
        self.assertIn(
            'call_0: <[40]> "weather" <[200049]> "{\\"name\\":\\"weather\\",\\"args\\":"'
            ' arguments_0 "}" <[43]>\n',
            main,
        )


if __name__ == "__main__":
    unittest.main()
