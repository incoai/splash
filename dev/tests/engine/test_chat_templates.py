import copy
import json
import unittest
from pathlib import Path

from jinja2 import TemplateError
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from dev.tests import test_server as fixtures
from server import api_shapes, chat_templates
from server import frontend as request_frontend
from server.chat_templates import (
    LATER_SYSTEM_UNSUPPORTED,
    NATIVE,
    PATCHED,
    REASONING_EFFORTS,
    UNSUPPORTED,
    ChatTemplateError,
    ChatTemplates,
    template_options,
)
from server.errors import APIError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/chat_templates"
UPSTREAM = ("qwen36", "qwen38", "qwen36_gguf", "qwen38_gguf")
# How each unmodified upstream template treats a later system message.
ORIGINAL = {
    "qwen36": chat_templates.REJECTS,
    "qwen38": chat_templates.REJECTS,
    "qwen36_gguf": chat_templates.DROPS,
    "qwen38_gguf": chat_templates.REJECTS,
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {"type": "function", "function": {"name": "list", "parameters": {}}},
]
# Normalized as request preparation leaves them: one leading system message.
AGENT_TURNS = [
    {"role": "system", "content": "You are a coding agent.\n\nWork in /repo."},
    {"role": "user", "content": "Fix the failing test"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Read the test first.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "t.py"}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "def test(): ..."},
    {"role": "system", "content": "Approval mode changed: never ask."},
    {"role": "user", "content": "Continue"},
]


def source(name):
    return (FIXTURES / f"{name}.jinja").read_text()


def tokenizer(template):
    result = PreTrainedTokenizerFast(
        tokenizer_object=fixtures._byte_backend({0: "hello"})
    )
    result.chat_template = template
    return result


def chat_tokenizer(template, *special):
    """A byte-level BPE tokenizer for `template`, with `special` and the think
    tags as added tokens. Like Qwen's, it encodes `assistant` and a blank line
    as one token each."""
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["assistant\n\n"],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    backend.add_special_tokens([AddedToken(text, normalized=False) for text in special])
    backend.add_tokens(
        [AddedToken(text, normalized=False) for text in ("<think>", "</think>")]
    )
    result = PreTrainedTokenizerFast(tokenizer_object=backend)
    result.chat_template = template
    return result


def render(template_tokenizer, messages, template=None, **options):
    return template_tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
        **options,
    )


def variants(name):
    """Changes to an upstream template that keep what it renders, at and
    around the construct it is patched at: formatting, and tags from
    transformers' Jinja extensions."""
    text = source(name)
    yield "space after endmacro", text.replace("{%- endmacro %}", "{%- endmacro %} ", 1)
    yield "trailing newlines", text + "\n\n"
    yield "CRLF line endings", text.replace("\n", "\r\n")
    yield "single line", "".join(line.strip() for line in text.split("\n"))
    # A dict literal ends in "}}" inside an expression, and a string after it
    # holds a statement tag.
    loop = "{%- for message in messages %}"
    yield (
        "nested dict literal",
        text.replace(
            loop, loop + "{{- {'a': {'b': ''}}['a']['b'] ~ '{% if x %}' if false }}", 1
        ),
    )
    raise_tag = "{{- raise_exception('System message must be at the beginning.') }}"
    if raise_tag in text:
        yield (
            "reformatted raise",
            text.replace(
                raise_tag,
                '{{-   raise_exception( "System message must be at the beginning." )   -}}',
            ),
        )
        yield (
            "reworded raise",
            text.replace(
                raise_tag, "{{- raise_exception('Put system messages first.') }}"
            ),
        )
    end, tool = "{{- '<|im_end|>\\n' }}", '\n    {%- elif message.role == "tool" %}'
    assistant = '{%- elif message.role == "assistant" %}'
    assert end + tool in text and assistant in text, name
    yield (
        "generation block",
        text.replace(
            end + tool, "{%- generation %}" + end + "{%- endgeneration %}" + tool, 1
        ),
    )
    yield (
        "continue statement",
        text.replace(
            assistant,
            '{%- elif message.role == "ignored" %}{%- continue %}' + assistant,
            1,
        ),
    )
    skip = '{%- if loop.index0 >= num_sys and message.role != "system" and message.role != "developer" %}'
    if skip in text:
        yield (
            "reformatted skip",
            text.replace(
                skip,
                "{%-if loop.index0>=num_sys and message['role']!='system' "
                "and message.role != 'developer'-%}",
            ),
        )


class ChatTemplateProbeTests(unittest.TestCase):
    def test_upstream_templates_are_classified_and_patched(self):
        for name in UPSTREAM:
            with self.subTest(name=name):
                templates = ChatTemplates(tokenizer(source(name)))
                chosen = templates.select(None)
                self.assertEqual(chosen.original, ORIGINAL[name])
                self.assertEqual(chosen.later_system, PATCHED)
                self.assertEqual(templates.status(), {"later_system": PATCHED})
                # This tokenizer reads any text as one token.
                self.assertEqual(
                    templates.describe(),
                    "patched to render later system messages in place "
                    f"(the original template {ORIGINAL[name]} them), "
                    "generation prompt 1 token",
                )

    def test_original_templates_reject_or_drop_what_the_patch_renders(self):
        for name in UPSTREAM:
            with self.subTest(name=name):
                upstream = tokenizer(source(name))
                if ORIGINAL[name] == chat_templates.REJECTS:
                    with self.assertRaises(TemplateError):
                        render(upstream, AGENT_TURNS)
                else:
                    self.assertNotIn("Approval mode", render(upstream, AGENT_TURNS))

    def test_later_system_message_renders_in_place_as_a_system_turn(self):
        for name in UPSTREAM:
            upstream = tokenizer(source(name))
            patched = ChatTemplates(upstream).select(TOOLS).source
            for options in (
                {},
                {"tools": TOOLS},
                {"enable_thinking": False},
                {"enable_thinking": True, "reasoning_effort": "low"},
                {"preserve_thinking": True},
            ):
                with self.subTest(name=name, options=options):
                    messages = copy.deepcopy(AGENT_TURNS)
                    rendered = render(upstream, messages, patched, **options)
                    self.assertIn(
                        "</tool_response><|im_end|>\n"
                        "<|im_start|>system\nApproval mode changed: never ask."
                        "<|im_end|>\n<|im_start|>user\nContinue<|im_end|>\n",
                        rendered,
                    )
                    self.assertEqual(rendered.count("Approval mode"), 1)
                    self.assertEqual(messages, AGENT_TURNS)
            self.assertEqual(upstream.chat_template, source(name))

    def test_consecutive_later_system_messages_each_render_in_place(self):
        messages = [
            {"role": "system", "content": "Original instructions"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "system", "content": "New instructions"},
            {"role": "system", "content": "More instructions"},
            {"role": "user", "content": "Next question"},
        ]
        for name in UPSTREAM:
            with self.subTest(name=name):
                upstream = tokenizer(source(name))
                patched = ChatTemplates(upstream).select(None).source
                self.assertIn(
                    "First answer<|im_end|>\n"
                    "<|im_start|>system\nNew instructions<|im_end|>\n"
                    "<|im_start|>system\nMore instructions<|im_end|>\n"
                    "<|im_start|>user\nNext question",
                    render(upstream, messages, patched),
                )

    def test_ordinary_conversations_render_byte_identically(self):
        image = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
            ],
        }
        two_calls = copy.deepcopy(AGENT_TURNS[2])
        two_calls["tool_calls"].append(
            {"type": "function", "function": {"name": "list", "arguments": {}}}
        )
        conversations = (
            AGENT_TURNS[:2],
            AGENT_TURNS[:4],
            [*AGENT_TURNS[:4], {"role": "assistant", "content": "Done"}, image],
            [AGENT_TURNS[1], two_calls, AGENT_TURNS[3], AGENT_TURNS[3]],
            [image, {"role": "assistant", "content": "<think>\nx\n</think>\n\nSame"}],
        )
        for name in UPSTREAM:
            upstream = tokenizer(source(name))
            patched = ChatTemplates(upstream).select(None).source
            for messages in conversations:
                for options in (
                    {},
                    {"tools": TOOLS},
                    {"enable_thinking": False},
                    {"enable_thinking": True, "reasoning_effort": "medium"},
                    {"preserve_thinking": False},
                    {"add_generation_prompt": False},
                ):
                    with self.subTest(name=name, messages=messages, options=options):
                        expected = upstream.apply_chat_template(
                            messages, tokenize=False, **options
                        )
                        self.assertEqual(
                            upstream.apply_chat_template(
                                messages,
                                chat_template=patched,
                                tokenize=False,
                                **options,
                            ),
                            expected,
                        )

    def test_template_variants_are_patched_at_the_same_construct(self):
        for name in UPSTREAM:
            for variant, text in variants(name):
                with self.subTest(name=name, variant=variant):
                    chosen = ChatTemplates(tokenizer(text)).select(None)
                    self.assertEqual(chosen.later_system, PATCHED)
                    upstream = tokenizer(text)
                    self.assertIn(
                        "<|im_start|>system\nApproval mode changed: never ask."
                        "<|im_end|>\n<|im_start|>user\nContinue",
                        render(upstream, AGENT_TURNS, chosen.source),
                    )
                    self.assertEqual(
                        render(upstream, AGENT_TURNS[:4], chosen.source),
                        render(upstream, AGENT_TURNS[:4]),
                    )

    def test_native_template_is_used_unchanged(self):
        native = (
            "{%- for message in messages %}"
            "{{- '<|im_start|>' + message.role + '\\n' + message.content"
            " + '<|im_end|>\\n' }}{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
            "{%- endif %}"
        )
        chosen = ChatTemplates(tokenizer(native)).select(None)
        self.assertEqual((chosen.later_system, chosen.source), (NATIVE, native))

    def test_templates_without_the_construct_are_unsupported(self):
        chatml = (
            "{{- '<|im_start|>' + message.role + '\\n' + message.content"
            " + '<|im_end|>\\n' }}"
        )
        cases = {
            # The check lives outside the message loop.
            "rejects before the loop": (
                "{%- for message in messages[1:] %}{%- if message.role == 'system' %}"
                "{{- raise_exception('System message must be at the beginning.') }}"
                "{%- endif %}{%- endfor %}"
                "{%- for message in messages %}" + chatml + "{%- endfor %}"
            ),
            # A filtered loop drops system messages, with no condition to patch.
            "drops by loop filter": (
                "{%- if messages[0].role == 'system' %}"
                "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
                "{%- endif %}"
                "{%- for message in messages if message.role != 'system' %}"
                + chatml
                + "{%- endfor %}"
            ),
            # Moves every system message to the front.
            "misplaces": (
                "{%- for message in messages if message.role == 'system' %}"
                + chatml
                + "{%- endfor %}"
                "{%- for message in messages if message.role != 'system' %}"
                + chatml
                + "{%- endfor %}"
            ),
            "unbalanced": source("qwen36").replace("{%- endmacro %}", "", 1),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                chosen = ChatTemplates(tokenizer(text)).select(None)
                self.assertEqual(
                    (chosen.later_system, chosen.source), (UNSUPPORTED, text)
                )

    def test_patch_that_changes_ordinary_conversations_is_not_kept(self):
        # Tool results share the rejecting branch: the construct is found, but
        # replacing its raise would render them as system turns.
        text = source("qwen36").replace(
            '{%- if message.role == "system" %}',
            '{%- if message.role == "system" or message.role == "tool" %}',
            1,
        )
        upstream = tokenizer(text)
        with self.assertRaises(TemplateError):
            render(upstream, AGENT_TURNS[:4])
        patch = chat_templates._patch(
            chat_templates._renderer(upstream), text, chat_templates.REJECTS
        )
        # The patch alone renders the later system message...
        self.assertIn(
            "<|im_start|>system\nApproval",
            render(upstream, [*AGENT_TURNS[:2], *AGENT_TURNS[4:]], patch),
        )
        # ...but verification sees the tool result it would change.
        chosen = ChatTemplates(upstream).select(None)
        self.assertEqual((chosen.later_system, chosen.source), (UNSUPPORTED, text))

    def test_patch_is_verified_under_every_reasoning_effort(self):
        # The guard also raises for the leading system message under one
        # effort, so there the patch would render that message twice.
        guard = "{%- if not loop.first %}"
        for effort in chat_templates.REASONING_EFFORTS:
            if effort == "none":  # reaches the template as enable_thinking only
                continue
            text = source("qwen36").replace(
                guard, f"{{%- if not loop.first or reasoning_effort == '{effort}' %}}"
            )
            with self.subTest(effort=effort):
                chosen = ChatTemplates(tokenizer(text)).select(None)
                self.assertEqual(
                    (chosen.later_system, chosen.source), (UNSUPPORTED, text)
                )

    def test_named_templates_are_probed_and_selected_like_the_tokenizer(self):
        native = (
            "{%- for message in messages %}{{- message.role + ': ' + message.content"
            " + '\\n' }}{%- endfor %}"
        )
        upstream = tokenizer({"default": native, "tool_use": source("qwen36")})
        templates = ChatTemplates(upstream)
        self.assertEqual(templates.select(None).later_system, NATIVE)
        self.assertEqual(templates.select(TOOLS).later_system, PATCHED)
        self.assertEqual(
            templates.status(),
            {"later_system": {"default": NATIVE, "tool_use": PATCHED}},
        )
        self.assertEqual(
            templates.describe(),
            "default renders later system messages in place, generation prompt "
            "unknown · tool_use patched to render later system messages in place "
            "(the original template rejects them), generation prompt 1 token",
        )
        for defined, message in (
            (None, "defines no chat template"),
            ("", "defines no chat template"),
            ({"tool_use": native}, "have no default"),
        ):
            with (
                self.subTest(defined=defined),
                self.assertRaisesRegex(ChatTemplateError, message),
            ):
                ChatTemplates(tokenizer(defined))


class ChatTemplateFrontendTests(unittest.TestCase):
    class OffsetTokenizer(fixtures.TemplateTokenizer):
        def __call__(self, text, **kwargs):
            return self.renderer(text, **kwargs)

    def harness(self, template, runtime=None):
        harness = fixtures.Harness(
            runtime or fixtures.FakeRuntime(), tokenizer=self.OffsetTokenizer(template)
        )
        self.addCleanup(harness.close)
        return harness

    def post(self, harness, path, body, status=200):
        code, _, payload = harness.request(
            "POST", path, {"model": "test-model", **body}
        )
        self.assertEqual(code, status, payload)
        return json.loads(payload)

    def test_every_request_path_uses_the_template_chosen_at_startup(self):
        harness = self.harness(
            source("qwen36"), fixtures.FakeRuntime(fixtures.Plan([[1]]))
        )
        chosen = harness.app.chat_templates.select(None)
        status, _, payload = harness.request("GET", "/status")
        self.assertEqual(
            json.loads(payload)["chat_template"], {"later_system": PATCHED}
        )
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Later"},
            {"role": "user", "content": "Again"},
        ]
        image = {
            "role": "user",
            "content": [
                {"type": "image", "image": "unused"},
                {"type": "text", "text": "Describe"},
            ],
        }
        # Image rendering swaps the pad token for a marker in that template.
        chosen_sources = {
            chosen.source,
            chosen.source.replace(
                api_shapes.IMAGE_PAD_TOKEN, request_frontend.IMAGE_RENDER_MARKER
            ),
        }
        for path, render in (
            (
                "/apply-template",
                lambda: self.post(harness, "/apply-template", {"messages": messages}),
            ),
            (
                "/v1/chat/completions",
                lambda: self.post(
                    harness,
                    "/v1/chat/completions",
                    {"messages": messages, "max_tokens": 1},
                ),
            ),
            (
                "/v1/messages/count_tokens",
                lambda: self.post(
                    harness, "/v1/messages/count_tokens", {"messages": messages}
                ),
            ),
            (
                "image render",
                lambda: harness.app._render_image_tokens(
                    [*messages, image], {"chat_template": chosen.source}
                ),
            ),
        ):
            with self.subTest(path=path):
                harness.tokenizer.templates.clear()
                render()
                used = {
                    kwargs["chat_template"] for _, kwargs in harness.tokenizer.templates
                }
                self.assertTrue(used)
                self.assertLessEqual(used, chosen_sources)
        prompt = self.post(harness, "/apply-template", {"messages": messages})["prompt"]
        self.assertIn(
            "<|im_start|>system\nLater<|im_end|>\n<|im_start|>user\nAgain", prompt
        )
        _, _, rendered = harness.app._render_image_tokens(
            [*messages, image], {"chat_template": chosen.source}
        )
        self.assertIn("<|im_start|>system\nLater<|im_end|>", rendered)
        self.assertIn("<|image_pad|>", rendered)
        self.assertEqual(harness.tokenizer.renderer.chat_template, source("qwen36"))

    def test_requests_carry_the_length_of_their_generation_prompt(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Again"},
        ]
        image = {"url": fixtures.ServerTest._png_data_url()}
        with_image = [
            *messages[:-1],
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": "Again"},
                ],
            },
        ]

        def frontend(name, *special):
            return fixtures.make_frontend(
                chat_tokenizer(source(name), *special),
                None,
                "test-model",
                4096,
                10,
                2,
                vision=True,
            )

        for name in ("qwen36_gguf", "qwen38_gguf"):
            app = frontend(
                name, "<|im_start|>", "<|im_end|>", api_shapes.IMAGE_PAD_TOKEN
            )
            self.assertTrue(
                app.chat_templates.describe().endswith("generation prompt 5-7 tokens")
            )
            # <|im_start|> assistant \n <think> \n\n </think> \n\n; with
            # thinking, the first four and \n.
            for effort, expected in (("none", 7), ("high", 5)):
                with self.subTest(name=name, effort=effort):
                    body = {
                        "model": "test-model",
                        "messages": messages,
                        "reasoning_effort": effort,
                    }
                    job = app.prepare(body, deadline=fixtures.FOREVER)
                    self.assertEqual(job.generation_prompt_tokens, expected)
                    history = app.apply_template(
                        {**body, "add_generation_prompt": False},
                        deadline=fixtures.FOREVER,
                    )
                    self.assertEqual(
                        job.prompt_tokens[:-expected],
                        app.tokenizer(history, add_special_tokens=False)["input_ids"],
                    )
                    # Images precede it, so expanding them keeps its length.
                    image_job = app.prepare(
                        {**body, "messages": with_image}, deadline=fixtures.FOREVER
                    )
                    self.assertEqual(image_job.generation_prompt_tokens, expected)
                    self.assertEqual(
                        image_job.prompt_tokens[-expected:],
                        job.prompt_tokens[-expected:],
                    )
        # The template says where its generation prompt starts, so a tokenizer
        # without a turn-start token finds the same boundary.
        app = frontend("qwen36_gguf", "<|im_end|>")
        body = {"model": "test-model", "messages": messages}
        job = app.prepare(body, deadline=fixtures.FOREVER)
        history = app.apply_template(
            {**body, "add_generation_prompt": False}, deadline=fixtures.FOREVER
        )
        self.assertGreater(job.generation_prompt_tokens, 0)
        self.assertEqual(
            job.prompt_tokens[: -job.generation_prompt_tokens],
            app.tokenizer(history, add_special_tokens=False)["input_ids"],
        )

    def test_generation_prompts_need_no_turn_marker(self):
        """Whatever a chat format's markers, the generation prompt is what its
        template appends. A template that appends nothing, or renders the
        conversation itself differently, has none, and its prompts are not
        served."""
        gemma = (
            "{% for m in messages %}<start_of_turn>"
            "{{ 'model' if m.role == 'assistant' else m.role }}\n"
            "{{ m.content }}<end_of_turn>\n{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        )
        appends_nothing = (
            "{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}"
        )
        rewrites_last_turn = (
            "{% for m in messages %}{{ m.content }}"
            "{% if add_generation_prompt and loop.last %}?{% endif %} {% endfor %}"
            "{% if add_generation_prompt %}>{% endif %}"
        )
        gemma_tokenizer = chat_tokenizer(gemma, "<start_of_turn>", "<end_of_turn>")
        chosen = ChatTemplates(gemma_tokenizer).select(None)
        text = "<start_of_turn>model\n"
        ids = tuple(gemma_tokenizer(text, add_special_tokens=False)["input_ids"])
        for effort in (None, *REASONING_EFFORTS):
            options = template_options(
                reasoning_effort=effort,
                preserve_thinking=None,
                tools=None,
                add_generation_prompt=True,
            )
            self.assertEqual(
                chosen.generation_prompts[chat_templates._generation_key(options)],
                (text, ids),
            )
        body = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Again"},
            ],
        }

        def frontend(template_tokenizer):
            return fixtures.make_frontend(
                template_tokenizer, None, "test-model", 4096, 10, 2, vision=False
            )

        job = frontend(gemma_tokenizer).prepare(body, deadline=fixtures.FOREVER)
        self.assertEqual(
            (job.generation_prompt_tokens, job.thinking), (len(ids), False)
        )
        self.assertEqual(tuple(job.prompt_tokens[-len(ids) :]), ids)
        for template in (appends_nothing, rewrites_last_turn):
            with self.subTest(template=template):
                template_tokenizer = chat_tokenizer(template)
                chosen = ChatTemplates(template_tokenizer).select(None)
                self.assertEqual(chosen.generation_prompts, {})
                with self.assertRaisesRegex(APIError, "assistant generation prefix"):
                    frontend(template_tokenizer).prepare(
                        body, deadline=fixtures.FOREVER
                    )

    def test_template_kwargs_are_template_variables(self):
        """A request's chat_template_kwargs reach the template: its own
        enable_thinking outranks the reasoning effort and the server default,
        and a variable only another template reads is probed when first set."""
        messages = [{"role": "user", "content": "Hi"}]

        def frontend(template, **options):
            return fixtures.make_frontend(
                chat_tokenizer(template, "<|im_start|>", "<|im_end|>"),
                None,
                "test-model",
                4096,
                10,
                2,
                vision=False,
                **options,
            )

        qwen = source("qwen36_gguf")
        off, on = {"enable_thinking": False}, {"enable_thinking": True}
        for options, extra, kwargs, thinking, tokens in (
            ({}, {}, off, False, 7),
            ({"default_reasoning_effort": "high"}, {}, off, False, 7),
            ({}, {"reasoning_effort": "high"}, off, False, 7),
            ({}, {"reasoning_effort": "none"}, on, True, 5),
        ):
            with self.subTest(options=options, extra=extra, kwargs=kwargs):
                job = frontend(qwen, **options).prepare(
                    {
                        "model": "test-model",
                        "messages": messages,
                        "chat_template_kwargs": kwargs,
                        **extra,
                    },
                    deadline=fixtures.FOREVER,
                )
                self.assertEqual(
                    (job.thinking, job.generation_prompt_tokens), (thinking, tokens)
                )
        # A switch Splash does not set, as DeepSeek's templates name it.
        switch = (
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>"
            "{% if thinking %}<think>{% else %}</think>{% endif %}{% endif %}"
        )
        app = frontend(switch)
        body = {"model": "test-model", "messages": messages}
        for kwargs, expected in ((None, False), ({"thinking": True}, True)) * 2:
            job = app.prepare(
                {**body, "chat_template_kwargs": kwargs}, deadline=fixtures.FOREVER
            )
            self.assertEqual(job.thinking, expected)
        self.assertEqual(app.chat_templates.select(None).probe.cache_info().misses, 1)
        for kwargs, error in (
            ("on", "must be an object"),
            ({"tools": []}, "cannot set tools"),
            ({"add_generation_prompt": False}, "cannot set add_generation_prompt"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(APIError, error):
                app.prepare(
                    {**body, "chat_template_kwargs": kwargs}, deadline=fixtures.FOREVER
                )

    class ScoringTokenizer(
        fixtures.TemplateTokenizer, fixtures.ServerTest.CharTokenizer
    ):
        """Renders the real template; one token per character for answer slots."""

    def test_scoring_prompts_use_the_template_chosen_at_startup(self):
        tokenizer = self.ScoringTokenizer(source("qwen36"))
        app = fixtures.make_frontend(
            tokenizer, None, "test-model", 8192, 10, 2, vision=True
        )
        tokenizer.templates.clear()
        app.prepare_judgment(
            fixtures.ServerTest.judgment_body(), deadline=fixtures.FOREVER
        )
        app.prepare_systemone(
            {"model": "test-model", "state": {}, "questions": {"q": {"type": "noul"}}},
            deadline=fixtures.FOREVER,
        )
        self.assertEqual(
            [kwargs.get("chat_template") for _, kwargs in tokenizer.templates],
            [app.chat_templates.select(None).source] * 2,
        )

    def test_codex_shaped_responses_render_instructions_first_and_later_in_place(self):
        harness = self.harness(source("qwen36_gguf"), fixtures.FakeRuntime())
        body = {
            "instructions": "Base instructions",
            "input": [
                {"type": "message", "role": "developer", "content": "Permissions"},
                {"type": "message", "role": "developer", "content": "Environment"},
                {"type": "message", "role": "user", "content": "Start"},
                {"type": "message", "role": "developer", "content": "Mode changed"},
                {"type": "message", "role": "user", "content": "Continue"},
            ],
        }
        chat, _ = api_shapes.responses_to_chat_body(body, body["input"])
        prompt = harness.app._render_prompt(
            harness.app._prepare_prompt(chat, None, deadline=fixtures.FOREVER),
            float("inf"),
            check_context=False,
        ).text
        self.assertTrue(
            prompt.startswith(
                "<|im_start|>system\nBase instructions\n\nPermissions\n\nEnvironment"
                "<|im_end|>\n<|im_start|>user\nStart<|im_end|>\n"
                "<|im_start|>system\nMode changed<|im_end|>\n"
                "<|im_start|>user\nContinue<|im_end|>\n"
            ),
            prompt,
        )

    def test_unsupported_template_rejects_later_system_messages(self):
        text = source("qwen36").replace(
            "{{- raise_exception('System message must be at the beginning.') }}",
            "{{- '' }}",
        )
        runtime = fixtures.FakeRuntime()
        harness = self.harness(text, runtime)
        self.assertEqual(
            harness.app.chat_templates.select(None).later_system, UNSUPPORTED
        )
        later = [
            {"role": "user", "content": "Hi"},
            {"role": "developer", "content": "Later"},
            {"role": "user", "content": "Again"},
        ]
        for path, body in (
            ("/v1/chat/completions", {"messages": later}),
            ("/apply-template", {"messages": later}),
            (
                "/v1/responses",
                {
                    "input": [
                        {"role": message["role"], "content": message["content"]}
                        for message in later
                    ]
                },
            ),
            (
                "/v1/messages/count_tokens",
                {"messages": [later[0], {"role": "system", "content": "x"}, later[2]]},
            ),
        ):
            with self.subTest(path=path):
                harness.tokenizer.templates.clear()
                error = self.post(harness, path, body, 400)["error"]
                self.assertEqual(error["message"], LATER_SYSTEM_UNSUPPORTED)
                self.assertEqual(harness.tokenizer.templates, [])
        self.assertEqual(runtime.requests, [])
        # A leading system message is not a later one.
        prompt = self.post(
            harness,
            "/apply-template",
            {"messages": [{"role": "system", "content": "First"}, later[0]]},
        )["prompt"]
        self.assertTrue(prompt.startswith("<|im_start|>system\nFirst<|im_end|>"))


class LeadingSystemMergeTests(unittest.TestCase):
    def test_leading_system_and_developer_messages_merge_into_one(self):
        merged = api_shapes.normalize_messages(
            [
                {"role": "system", "content": "One"},
                {"role": "developer", "content": [{"type": "text", "text": "Two"}]},
                {"role": "system", "content": ""},
                {"role": "developer", "content": "Three"},
                {"role": "user", "content": "Ask"},
                {"role": "developer", "content": "Later"},
                {"role": "system", "content": "Later still"},
            ],
            vision=True,
            deadline=fixtures.FOREVER,
        )
        self.assertEqual(
            merged,
            [
                {"role": "system", "content": "One\n\nTwo\n\nThree"},
                {"role": "user", "content": "Ask"},
                {"role": "system", "content": "Later"},
                {"role": "system", "content": "Later still"},
            ],
        )
        self.assertEqual(
            api_shapes.normalize_messages(
                [
                    {"role": "system", "content": "  Only  "},
                    {"role": "user", "content": "x"},
                ],
                vision=True,
                deadline=fixtures.FOREVER,
            )[0],
            {"role": "system", "content": "  Only  "},
        )

    def test_every_api_shape_leads_with_one_system_message(self):
        responses = api_shapes.responses_to_chat_body(
            {"instructions": "Base"},
            [
                {"role": "developer", "content": "Developer"},
                {"role": "user", "content": "Ask"},
            ],
        )[0]["messages"]
        anthropic = api_shapes.anthropic_to_chat_prompt(
            {
                "model": "m",
                "system": "Base",
                "messages": [
                    {"role": "system", "content": "Developer"},
                    {"role": "user", "content": "Ask"},
                ],
            },
            thinking_resolver=fixtures.no_signed_thinking,
        )["messages"]
        for messages in (responses, anthropic):
            with self.subTest(messages=messages):
                self.assertEqual(
                    api_shapes.normalize_messages(
                        messages, vision=True, deadline=fixtures.FOREVER
                    ),
                    [
                        {"role": "system", "content": "Base\n\nDeveloper"},
                        {"role": "user", "content": "Ask"},
                    ],
                )


if __name__ == "__main__":
    unittest.main()
