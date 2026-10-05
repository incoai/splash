"""Reasoning effort and thinking mode as a request sets them and the chat
template renders them; the server's default effort is in
test_default_reasoning.py."""

import json
import unittest
from unittest import mock

from dev.tests.server_fixtures import (
    FOREVER,
    FakeConstraintFactory,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    Plan,
    TemplateTokenizer,
    chat_body,
    make_frontend,
    reasoning_template,
    responses_body,
    rich_weather_tool,
)
from server import backend as backend_api
from server import frontend as request_frontend
from server import server as api


class ReasoningTests(HarnessTestCase):
    def test_qwen_reasoning_efforts(self):
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(
            FakeRuntime(), tokenizer, lambda _record: None
        )
        self.addCleanup(backend.close)
        app = make_frontend(tokenizer, backend, "test-model", 128, 1, 2, vision=True)
        for effort in ("xhigh", "medium", "low"):
            app.prepare(chat_body(reasoning_effort=effort), deadline=FOREVER)
            template = tokenizer.templates[-1][1]
            self.assertTrue(template["enable_thinking"])
            self.assertEqual(template["reasoning_effort"], effort)
        app.prepare(chat_body(reasoning_effort="none"), deadline=FOREVER)
        template = tokenizer.templates[-1][1]
        self.assertFalse(template["enable_thinking"])
        self.assertEqual(template["reasoning_effort"], "none")

    def test_native_thinking_default_controls_parser_and_grammar(self):
        schema = {"type": "object", "properties": {"x": {"const": 3}}}
        for default in (False, True):
            for tools in (False, True):
                with self.subTest(default=default, tools=tools):
                    tokenizer = TemplateTokenizer(reasoning_template(default=default))
                    factory = FakeConstraintFactory()
                    batches = [[1], [26], [10]] if default else [[10]]
                    harness = self.harness(
                        FakeRuntime(Plan(batches), Plan(batches)),
                        tokenizer=tokenizer,
                        constraint_factory=factory,
                    )
                    body = chat_body(
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"schema": schema},
                        }
                    )
                    if tools:
                        body["tools"] = [rich_weather_tool()]
                    for stream in (False, True):
                        status, _, payload = harness.request(
                            "POST", "/v1/chat/completions", {**body, "stream": stream}
                        )
                        self.assertEqual(status, 200, payload)
                        self.assertEqual("think:" in factory.grammars[-1], default)
                        if stream:
                            self.assertEqual(b'"reasoning_content"' in payload, default)
                            self.assertIn(b"[DONE]", payload)
                        else:
                            message = json.loads(payload)["choices"][0]["message"]
                            self.assertEqual(message["content"], '{"x":3}')
                            self.assertEqual("reasoning_content" in message, default)
                    for _, kwargs in tokenizer.templates:
                        self.assertNotIn("enable_thinking", kwargs)
                        self.assertNotIn("reasoning_effort", kwargs)

    def test_dictionary_template_uses_the_selected_generation_prefix(self):
        tokenizer = TemplateTokenizer(
            {
                "default": reasoning_template(default=False),
                "tool_use": reasoning_template(default=True),
            }
        )
        factory = FakeConstraintFactory()
        app = make_frontend(
            tokenizer,
            None,
            "test-model",
            128,
            1,
            2,
            constraint_factory=factory,
            vision=True,
        )
        for tools, expected in (([], False), ([rich_weather_tool()], True)):
            body = chat_body(tools=tools, tool_choice="required" if tools else "auto")
            body["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": "A historical <think> must not open thinking.",
                },
            )
            job = app.prepare(body, deadline=FOREVER)
            self.assertEqual(job.thinking, expected)
            self.assertEqual(
                app.count_tokens(body, deadline=FOREVER), len(job.prompt_tokens)
            )
        self.assertIn("think:", factory.grammars[-1])

    def test_responses_omitted_or_null_effort_preserves_native_thinking_off(self):
        tokenizer = TemplateTokenizer(reasoning_template(default=False))
        harness = self.harness(
            FakeRuntime(Plan([[27]]), Plan([[27]]), Plan([[27]])), tokenizer=tokenizer
        )
        for extra in ({}, {"reasoning": None}, {"reasoning": {"effort": None}}):
            with self.subTest(extra=extra):
                status, _, payload = harness.request(
                    "POST", "/v1/responses", responses_body(**extra)
                )
                self.assertEqual(status, 200, payload)
                response = json.loads(payload)
                self.assertEqual(response["status"], "completed")
                self.assertEqual(
                    [item["type"] for item in response["output"]], ["message"]
                )
                self.assertEqual(
                    response["output"][0]["content"][0]["text"], "answer\n"
                )
                kwargs = tokenizer.templates[-1][1]
                self.assertNotIn("enable_thinking", kwargs)
                self.assertNotIn("reasoning_effort", kwargs)

    def test_reasoning_aliases_only_retry_when_the_template_rejects_native_values(self):
        for accepted, aliases in (
            (("minimal", "low", "medium", "high", "xhigh", "max"), {}),
            (
                ("low", "medium", "xhigh"),
                {"high": "xhigh", "max": "xhigh", "minimal": "low"},
            ),
        ):
            tokenizer = TemplateTokenizer(reasoning_template(efforts=accepted))
            app = make_frontend(tokenizer, None, "test-model", 128, 1, 2, vision=True)
            for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
                with self.subTest(accepted=accepted, effort=effort):
                    tokenizer.templates.clear()
                    job = app.prepare(
                        chat_body(reasoning_effort=effort), deadline=FOREVER
                    )
                    self.assertTrue(job.thinking)
                    self.assertEqual(
                        [
                            kwargs["reasoning_effort"]
                            for _, kwargs in tokenizer.templates
                        ],
                        [effort, aliases[effort]] if effort in aliases else [effort],
                    )

    def test_reasoning_template_errors_do_not_silently_drop_effort(self):
        tokenizer = TemplateTokenizer(reasoning_template(efforts=("medium",)))
        app = make_frontend(tokenizer, None, "test-model", 128, 1, 2, vision=True)
        tokenizer.templates.clear()
        with self.assertRaises(api.APIError):
            app.prepare(chat_body(reasoning_effort="high"), deadline=FOREVER)
        self.assertEqual(
            [kwargs["reasoning_effort"] for _, kwargs in tokenizer.templates],
            ["high", "xhigh"],
        )
        with mock.patch.object(
            tokenizer, "apply_chat_template", side_effect=ValueError("bad content")
        ) as render:
            with self.assertRaises(api.APIError) as caught:
                app.prepare(chat_body(reasoning_effort="high"), deadline=FOREVER)
            render.assert_called_once()
            self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_reasoning_effort_accepts_only_standard_protocol_values(self):
        tokenizer = FakeTokenizer()
        app = make_frontend(tokenizer, None, "test-model", 128, 1, 2, vision=True)
        tokenizer.templates.clear()
        for effort in ("", "on", "off", "ultra", True, 1, [], {}):
            with (
                self.subTest(effort=effort),
                self.assertRaisesRegex(api.APIError, "invalid reasoning_effort"),
            ):
                app.prepare(chat_body(reasoning_effort=effort), deadline=FOREVER)
        self.assertEqual(tokenizer.templates, [])
        job = app.prepare(chat_body(reasoning_effort=None), deadline=FOREVER)
        self.assertTrue(job.thinking)
        self.assertNotIn("enable_thinking", tokenizer.templates[-1][1])

    def test_explicit_thinking_mode_must_be_honored(self):
        for prefix, effort in (
            ("<think>\\n", "none"),
            ("<think>\\n\\n</think>\\n\\n", "low"),
        ):
            tokenizer = TemplateTokenizer(
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\\n" + prefix + "' }}{% endif %}"
            )
            app = make_frontend(tokenizer, None, "test-model", 128, 1, 2, vision=True)
            with (
                self.subTest(effort=effort),
                self.assertRaisesRegex(api.APIError, "requested thinking mode"),
            ):
                app.prepare(chat_body(reasoning_effort=effort), deadline=FOREVER)

    def test_generation_prompt_decides_thinking_and_its_token_count(self):
        history = "<|im_start|>assistant\n<think>old</think>answer<|im_end|>\n"
        for suffix, thinking in (
            ("", False),
            ("<think>\n", True),
            ("<think>\n\n</think>\n\n", False),
        ):
            text = "<|im_start|>assistant\n" + suffix
            # Tokens that end otherwise, or are the generation prompt alone,
            # are not counted.
            for tokens, count in (
                ([1, 2, 7, 8, 9], 3),
                ([1, 2, 7, 8, 8], 0),
                ([7, 8, 9], 0),
            ):
                with self.subTest(suffix=suffix, tokens=tokens):
                    self.assertEqual(
                        request_frontend._generation_prompt(
                            (text, (7, 8, 9)), history + text, tokens
                        ),
                        (thinking, count),
                    )
        # A prompt must end with the generation prompt the probe found.
        assistant = ("<|im_start|>assistant\n", (7, 8))
        for probed, rendered in (
            (assistant, ""),
            (assistant, "<|im_start|>assistant\n<think>old<|im_end|>\n"),
            (("", ()), history),
        ):
            with self.subTest(rendered=rendered), self.assertRaises(api.APIError):
                request_frontend._generation_prompt(probed, rendered, [1, 2, 7, 8])

    def test_reasoning_effort_is_preserved_until_the_template_renders(self):
        harness = self.harness(FakeRuntime())
        prompt = harness.app._prepare_prompt(
            chat_body(reasoning_effort="minimal"), None, deadline=FOREVER
        )
        self.assertEqual(prompt.reasoning_effort, "minimal")
        job = harness.app.prepare(
            chat_body(reasoning_effort="minimal"), deadline=FOREVER
        )
        self.assertTrue(job.thinking)


if __name__ == "__main__":
    unittest.main()
