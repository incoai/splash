import time
import unittest
from types import SimpleNamespace

from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

from server import api_shapes
from server import frontend as request_frontend
from server.latency import LatencyMetrics
from server.tokenization import PromptTokenizer, plain_tokens

QUOTED = request_frontend.QUOTED_TAGS

# Qwen's layout: tools as JSON in the system turn, a tool result inside
# <tool_response>, an assistant's calls inside <tool_call>, all written by the
# template itself.
TEMPLATE = (
    "{%- if tools %}<|im_start|>system\n"
    "{%- for tool in tools %}{{ tool | tojson }}\n{% endfor %}<|im_end|>\n{% endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'tool' %}"
    "<|im_start|>user\n<tool_response>\n{{ message.content }}\n"
    "</tool_response><|im_end|>\n"
    "{%- elif message.role == 'assistant' %}<|im_start|>assistant\n"
    "{%- if message.reasoning_content %}"
    "<think>\n{{ message.reasoning_content }}\n</think>\n{% endif %}"
    "{%- if message.content %}{{ message.content }}{% endif %}"
    "{%- for call in message.tool_calls or [] %}"
    "<tool_call>\n<function={{ call.function.name }}>\n</function>\n</tool_call>"
    "{%- endfor %}<|im_end|>\n"
    "{%- else %}<|im_start|>{{ message.role }}\n"
    "{%- if message.content is string %}{{ message.content }}{% else %}"
    "{% for part in message.content %}{{ part.text }}{% endfor %}{% endif %}"
    "<|im_end|>\n{%- endif %}{%- endfor %}"
)


def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.ByteLevel(trim_offsets=False)
    backend.train_from_iterator(
        ['tool_call_start = "<tool_call>"\nRead the parser.\n'],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    backend.add_special_tokens(
        [
            AddedToken(text, normalized=False, lstrip=False, rstrip=False)
            for text in ("<|im_start|>", "<|im_end|>")
        ]
    )
    # As in Qwen's tokenizer: added, not special, matched wherever the text
    # spells them.
    backend.add_tokens(
        [
            AddedToken(
                text, normalized=False, special=False, lstrip=False, rstrip=False
            )
            for text in QUOTED + (api_shapes.IMAGE_PAD_TOKEN,)
        ]
    )
    result = PreTrainedTokenizerFast(tokenizer_object=backend)
    result.chat_template = TEMPLATE
    return result


def frontend(tokenizer):
    app = object.__new__(request_frontend.Frontend)
    app.tokenizer = tokenizer
    app.latencies = LatencyMetrics()
    app.max_context = 4096
    app.prompt_tokenizer = PromptTokenizer(tokenizer)
    app.quoted_tag_tokens = {tag: plain_tokens(tokenizer, tag) for tag in QUOTED}
    # Text prompts prepare no image.
    app.images = SimpleNamespace(request_batch=list)
    template = SimpleNamespace(source=TEMPLATE, accepts=lambda messages: True)
    app.chat_templates = SimpleNamespace(select=lambda tools: template)
    return app


def prompt(messages, tools=None):
    return request_frontend.Prompt(
        messages=messages, tools=tools, tool_policy=None, reasoning_effort=None
    )


class QuotedTagTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tokenizer()
        self.app = frontend(self.tokenizer)
        self.ids = {tag: self.tokenizer.convert_tokens_to_ids(tag) for tag in QUOTED}

    def render(self, messages, tools=None):
        return self.app._render_prompt(
            prompt(messages, tools), time.monotonic() + 30, add_generation_prompt=False
        )

    def count(self, tokens, tag):
        return tokens.count(self.ids[tag])

    def test_plain_tokens_spell_the_tag_without_its_added_token(self):
        for tag in QUOTED:
            tokens = plain_tokens(self.tokenizer, tag)
            self.assertNotIn(self.ids[tag], tokens)
            self.assertGreater(len(tokens), 1)
            self.assertEqual(self.tokenizer.decode(tokens), tag)

    def test_tags_quoted_by_a_tool_result_encode_as_text(self):
        source = 'tool_call_start = "<tool_call>"\ntool_call_end = "</tool_call>"'
        messages = [
            {"role": "user", "content": "Which tag opens a call? Quote </think> too."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "content": source + "\n<tool_response> nested"},
        ]
        rendered = self.render(messages)
        # The text the model reads is unchanged...
        self.assertEqual(
            rendered.text,
            self.tokenizer.apply_chat_template(messages, tokenize=False),
        )
        self.assertEqual(self.tokenizer.decode(rendered.tokens), rendered.text)
        # ...and only the template's own tags are its added tokens: the
        # assistant's call, and the tool result's frame.
        self.assertEqual(self.count(rendered.tokens, "<tool_call>"), 1)
        self.assertEqual(self.count(rendered.tokens, "</tool_call>"), 1)
        self.assertEqual(self.count(rendered.tokens, "<tool_response>"), 1)
        self.assertEqual(self.count(rendered.tokens, "</tool_response>"), 1)
        self.assertEqual(self.count(rendered.tokens, "</think>"), 0)
        quoted = plain_tokens(self.tokenizer, "<tool_call>")
        self.assertTrue(
            any(
                rendered.tokens[i : i + len(quoted)] == quoted
                for i in range(len(rendered.tokens))
            )
        )

    def test_assistant_reasoning_quotes_tool_tags_as_text(self):
        messages = [
            {"role": "user", "content": "Quote the tag."},
            {
                "role": "assistant",
                "reasoning_content": "The parser opens on <tool_call>.",
                "content": "",
                "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "done"},
        ]
        rendered = self.render(messages)
        self.assertEqual(self.tokenizer.decode(rendered.tokens), rendered.text)
        # The template's call keeps its tags; the reasoning's quote is text.
        self.assertEqual(self.count(rendered.tokens, "<tool_call>"), 1)
        self.assertEqual(self.count(rendered.tokens, "<think>"), 1)
        self.assertEqual(self.count(rendered.tokens, "</think>"), 1)

    def test_assistant_text_keeps_its_think_tags(self):
        messages = [
            {"role": "user", "content": "Quote the tag."},
            {
                "role": "assistant",
                "content": "<think>\nQuote it.\n</think>\nIt is <tool_call>.",
            },
        ]
        rendered = self.render(messages)
        self.assertEqual(self.count(rendered.tokens, "<think>"), 1)
        self.assertEqual(self.count(rendered.tokens, "</think>"), 1)
        self.assertEqual(self.count(rendered.tokens, "<tool_call>"), 0)

    def test_user_tool_result_keeps_its_frame(self):
        for content in (
            "<tool_response>\nIt reads <tool_call>.\n</tool_response>",
            [
                {"type": "text", "text": "<tool_response>\nIt reads "},
                {"type": "text", "text": "<tool_call>.\n</tool_response>"},
            ],
        ):
            rendered = self.render([{"role": "user", "content": content}])
            self.assertEqual(self.tokenizer.decode(rendered.tokens), rendered.text)
            self.assertEqual(self.count(rendered.tokens, "<tool_response>"), 1)
            self.assertEqual(self.count(rendered.tokens, "</tool_response>"), 1)
            self.assertEqual(self.count(rendered.tokens, "<tool_call>"), 0)

    def test_tool_definitions_quote_tags_as_text(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "parse",
                    "description": "Reads text such as <tool_call>.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        rendered = self.render([{"role": "user", "content": "Hi"}], tools)
        self.assertIn("<tool_call>", rendered.text)
        self.assertEqual(self.count(rendered.tokens, "<tool_call>"), 0)
        self.assertEqual(self.tokenizer.decode(rendered.tokens), rendered.text)

    def test_prompts_without_quoted_tags_encode_as_before(self):
        messages = [
            {"role": "user", "content": "Read the parser."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "no tags here"},
        ]
        rendered = self.render(messages)
        self.assertEqual(
            rendered.tokens,
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False
            ),
        )

    def test_quoted_tags_and_image_pads_keep_their_positions(self):
        pad = api_shapes.IMAGE_PAD_TOKEN
        template = TEMPLATE.replace(
            "{{ part.text }}",
            "{% if part.type == 'image_url' %}{{ '" + pad + "' }}"
            "{% else %}{{ part.text }}{% endif %}",
        )
        self.tokenizer.chat_template = template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Before <tool_call> and " + pad},
                    {"type": "image_url", "image_url": {"url": "unused"}},
                    {"type": "text", "text": "after </tool_call>"},
                    {"type": "image_url", "image_url": {"url": "unused"}},
                ],
            }
        ]
        marked = request_frontend._mark_quoted_messages(messages)
        tokens, positions, rendered = self.app._render_image_tokens(
            marked,
            {"tokenize": False, "return_dict": False, "chat_template": template},
        )
        self.assertEqual(
            rendered, self.tokenizer.apply_chat_template(messages, tokenize=False)
        )
        self.assertEqual(self.tokenizer.decode(tokens), rendered)
        pad_id = self.tokenizer.convert_tokens_to_ids(pad)
        # The quoted pad stays text the template did not write; the two
        # images' pads are found where the tags before them moved them.
        self.assertEqual(len(positions), 2)
        self.assertTrue(all(tokens[index] == pad_id for index in positions))
        self.assertEqual(tokens.count(pad_id), 3)
        self.assertEqual(self.count(tokens, "<tool_call>"), 0)
        self.assertEqual(self.count(tokens, "</tool_call>"), 0)


if __name__ == "__main__":
    unittest.main()
