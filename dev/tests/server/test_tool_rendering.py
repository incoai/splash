"""The same tool reaches the chat template as the same object through Chat,
Messages and Responses, so a template that prints a tool's JSON as it is,
as Qwen's does, renders the same prompt and one dialect's cached prompt
serves the others (serving-probe-cache.md, group I)."""

import json
import unittest
from pathlib import Path

from dev.tests import server_fixtures as fixtures

QWEN_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "fixtures/chat_templates/qwen36_gguf.jinja"
).read_text()
SYSTEM = "You answer briefly."
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def bodies(description=True, strict=None):
    """A Chat, a Messages and a Responses request of one conversation and
    one tool, each as its clients write it."""
    chat_function = {"name": "weather"}
    messages_tool = {"name": "weather"}
    responses_tool = {"type": "function", "name": "weather"}
    if description:
        for tool in (chat_function, messages_tool, responses_tool):
            tool["description"] = "The weather in a city."
    chat_function["parameters"] = SCHEMA
    messages_tool["input_schema"] = SCHEMA
    responses_tool["parameters"] = SCHEMA
    if strict is not None:
        for tool in (chat_function, messages_tool, responses_tool):
            tool["strict"] = strict
    user = "Weather in Paris?"
    return {
        "/v1/chat/completions": {
            "model": "test-model",
            "max_tokens": 4,
            "reasoning_effort": "none",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "tools": [{"type": "function", "function": chat_function}],
        },
        "/v1/messages": {
            "model": "test-model",
            "max_tokens": 4,
            "thinking": {"type": "disabled"},
            "system": SYSTEM,
            "messages": [{"role": "user", "content": user}],
            "tools": [messages_tool],
        },
        "/v1/responses": {
            "model": "test-model",
            "max_output_tokens": 4,
            "reasoning": {"effort": "none"},
            "input": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "tools": [responses_tool],
        },
    }


class ToolRenderingTests(unittest.TestCase):
    def rendered(self, path, body):
        """The tools the template received, as JSON in their key order, and
        the prompt it rendered."""
        tokenizer = fixtures.TemplateTokenizer(QWEN_TEMPLATE)
        harness = fixtures.Harness(
            fixtures.FakeRuntime(fixtures.Plan([[4]])),
            tokenizer=tokenizer,
            max_context=8192,
        )
        self.addCleanup(harness.close)
        status, _, payload = harness.request("POST", path, body)
        self.assertEqual(status, 200, payload)
        messages, options = tokenizer.templates[-1]
        prompt = tokenizer.renderer.apply_chat_template(
            messages, **(options | {"tokenize": False})
        )
        return json.dumps(options.get("tools")), prompt

    def test_every_dialect_hands_the_template_the_same_tool(self):
        for description, strict in ((True, None), (False, None), (True, True)):
            with self.subTest(description=description, strict=strict):
                rendered = {
                    path: self.rendered(path, body)
                    for path, body in bodies(description, strict).items()
                }
                chat_tools, chat_prompt = rendered["/v1/chat/completions"]
                for path, (tools, prompt) in rendered.items():
                    with self.subTest(path=path):
                        self.assertEqual(tools, chat_tools)
                        self.assertEqual(prompt, chat_prompt)
                self.assertIn(
                    '{"name": "weather", "description": "The weather in a city.", '
                    '"parameters": '
                    if description
                    else '{"name": "weather", "parameters": ',
                    chat_prompt,
                )


if __name__ == "__main__":
    unittest.main()
