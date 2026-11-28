import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dev.benchmarks import chat_prompts


class FakeTokenizer:
    """A chat template of Qwen's shape, thinking off by an empty thinking
    block, and a tokenizer of one id a character and one past 1000 for each
    special token."""

    SPECIAL = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")

    def __init__(self, repeat=1):
        self.repeat = repeat

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking=True, **options
    ):
        text = "".join(
            f"<|im_start|>{message['role']}\n{message['content'] * self.repeat}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + (
                "" if enable_thinking else "<think>\n\n</think>\n\n"
            )
        return text

    def __call__(self, text, add_special_tokens):
        ids = []
        while text:
            special = next(
                (token for token in self.SPECIAL if text.startswith(token)), None
            )
            ids.append(1000 + self.SPECIAL.index(special) if special else ord(text[0]))
            text = text[len(special) if special else 1 :]
        return {"input_ids": ids}


class ChatPromptsTests(unittest.TestCase):
    def test_decode_profiles_built_in_tokens_are_one_template_each(self):
        # The Qwen families' tokens and Flint's, as decode-profile holds them:
        # each counting request opens with its user header and closes with
        # its assistant header, and Qwen's is the backend benchmark's decode
        # prompt, the same request.
        qwen, flint = chat_prompts.profile_builtin()
        for chat in (qwen, flint):
            header, sentence, assistant, counting = chat
            self.assertTrue(all(chat))
            self.assertEqual(counting[: len(header)], header)
            self.assertEqual(counting[-len(assistant) :], assistant)
        self.assertEqual(qwen[3], chat_prompts.builtin()[0])
        self.assertEqual(len(qwen[1]), len(flint[1]))

    def test_a_models_chat_tokens_are_its_rendering_around_the_users_text(self):
        # The chat request decode-profile builds of them, the header, the
        # sentence repeated and the assistant header, is the rendering of the
        # repeated sentence, and the counting request that of its text.
        tokenizer = FakeTokenizer()
        header, sentence, assistant, counting = chat_prompts.profile_tokens(tokenizer)
        self.assertEqual(header, [1000, *map(ord, "user\n")])
        self.assertEqual(sentence, [*map(ord, chat_prompts.PROFILE_SENTENCE)])
        self.assertEqual(
            assistant,
            [1001, 10, 1000, *map(ord, "assistant\n"), 1002, 10, 10, 1003, 10, 10],
        )
        self.assertEqual(
            header + sentence * 3 + assistant,
            chat_prompts.render(tokenizer, chat_prompts.PROFILE_SENTENCE * 3),
        )
        self.assertEqual(
            counting, chat_prompts.render(tokenizer, chat_prompts.TEXTS[0])
        )

    def test_a_template_that_does_not_render_the_users_text_once_is_refused(self):
        with self.assertRaisesRegex(ValueError, "the user's text once"):
            chat_prompts.profile_tokens(FakeTokenizer(repeat=2))

    def test_decode_profile_takes_chat_tokens_where_they_are_not_built_in(self):
        qwen, flint = chat_prompts.profile_builtin()
        own = [[1, 2], [3], [4, 5], [1, 2, 6, 4, 5]]
        for tokens, options in ((qwen, []), (flint, []), (own, ["--chat-tokens"])):
            with (
                self.subTest(tokens=tokens[0]),
                TemporaryDirectory() as directory,
                mock.patch.object(
                    chat_prompts, "model_profile_tokens", return_value=tokens
                ),
            ):
                path = Path(directory) / "chat-tokens.txt"
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    self.assertEqual(
                        chat_prompts.main(["decode-profile", directory, str(path)]), 0
                    )
                if options:
                    self.assertEqual(printed.getvalue(), f"--chat-tokens {path}\n")
                    self.assertEqual(path.read_text(), "1 2\n3\n4 5\n1 2 6 4 5\n")
                else:
                    self.assertEqual(printed.getvalue(), "\n")
                    self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
