import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

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

from dev.tests.server_fixtures import NATIVE_CONTEXT
from server.chat_format import QWEN
from server.dialects import scoring
from server.tokenization import PromptTokenizer, tokenizable


def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.ByteLevel(trim_offsets=False)
    backend.train_from_iterator(
        ["Hello world!\n", "Release notes 中文 e\u0301 👨‍👩‍👧‍👦\n"],
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
    return PreTrainedTokenizerFast(tokenizer_object=backend)


class PromptTokenizationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tokenizer()
        self.cache = PromptTokenizer(
            self.tokenizer, QWEN.message_boundary, NATIVE_CONTEXT
        )
        self.assertTrue(self.cache.enabled)

    def assert_encoding(self, text, cache=None):
        actual = (cache or self.cache).encode(text)
        expected = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        self.assertEqual(actual, expected)
        return actual

    def test_extensions_unicode_and_changed_history_match_full_encoding(self):
        randomizer = random.Random(4321)
        fragments = [
            "e\u0301",
            "中文",
            " 👨‍👩‍👧‍👦",
            "\n\r\t ",
            "<|im_end|>",
            "<|im_end",
            "'s",
            "12345",
        ]
        history = "<|im_start|>system\n" + "Release notes. " * 400 + "<|im_end|>"
        for turn in range(30):
            text = "".join(randomizer.choices(fragments, k=12))
            history += "\n<|im_start|>user\n" + text + "<|im_end|>"
            prompt = history + "\n<|im_start|>assistant\n<think>\n"
            ids = self.assert_encoding(prompt)
            # Returned token lists are caller-owned, never the cache storage.
            ids[0] = 999999
            self.assert_encoding(prompt)
            if turn % 4 == 0:
                self.assert_encoding(
                    prompt.replace("Release notes.", "Other notes.", 1)
                )
        self.assertGreater(self.cache.stats()["reused_tokens"], 10000)

    def test_text_after_the_split_encodes_on_its_own_with_any_label(self):
        # Encoding text and a label equals the split's tokens followed by the
        # encoded end and label, so a label extends the whole text's tokens
        # by its token exactly when it so extends the end's: a slot check
        # need encode only the end (scoring.encode_prompt).
        randomizer = random.Random(1234)
        fragments = [
            "e\u0301",
            "中文",
            " 👨‍👩‍👧‍👦",
            "\n\r\t ",
            "<|im_end|>",
            "<|im_end",
            "<|im_start|>",
            "'s",
            "12345",
            "Hello",
            " world!",
            "AB",
            " ",
        ]
        labels = ("", "A", "Q", "Z", "AB", "ZZ")
        split = 0
        for _ in range(300):
            text = "".join(randomizer.choices(fragments, k=randomizer.randint(0, 12)))
            head, tail = self.cache.split(text)
            if head:
                split += 1
                self.assertNotIn("<|im_end|>", tail)
                self.assertEqual(head[-1], self.cache.marker_id)
            else:
                self.assertEqual(tail, text)
            for label in labels:
                self.assertEqual(
                    head
                    + self.tokenizer(tail + label, add_special_tokens=False)[
                        "input_ids"
                    ],
                    self.tokenizer(text + label, add_special_tokens=False)["input_ids"],
                )
        self.assertGreater(split, 50)
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_text_is_not_split_where_the_tokenizer_may_not_split_it(self):
        consumed = tokenizer()
        consumed.backend_tokenizer.add_special_tokens(
            [AddedToken("hello<|im_end", normalized=False)]
        )
        for name, t, text in (
            ("no marker", self.tokenizer, "Hello world!"),
            ("consumed marker", consumed, "hello<|im_end|>suffix"),
        ):
            with self.subTest(name):
                self.assertEqual(
                    PromptTokenizer(t, QWEN.message_boundary, NATIVE_CONTEXT).split(
                        text
                    ),
                    ([], text),
                )

    def test_slot_checks_encode_only_the_prompt_end(self):
        template = (
            "{% for message in messages %}<|im_start|>{{ message.role }}\n"
            "{{ message.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        messages = [
            {"role": "system", "content": "Choose one."},
            {"role": "user", "content": "Release notes 中文 e\u0301. " * 2000},
        ]
        labels = scoring.slot_labels(self.tokenizer)[:26]
        encode = self.tokenizer.encode
        tokens = self.cache.tokens
        encoded = []

        def counted(text):
            encoded.append(text)
            return tokens(text)

        with mock.patch.object(self.cache, "tokens", counted):
            ids, slots, prompt = scoring.encode_prompt(
                self.cache,
                template,
                messages,
                labels,
                chat_format=QWEN,
                admit=lambda tokens: None,
                checkpoint=lambda: None,
            )
        self.assertEqual(
            ids, self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        self.assertEqual(
            slots, [encode(label, add_special_tokens=False)[0] for label in labels]
        )
        end = "\n<|im_start|>assistant\n"
        self.assertTrue(prompt.endswith("<|im_end|>" + end))
        # The prompt up to its end once, the end, and each label after the
        # end.
        self.assertEqual(
            sorted(encoded),
            sorted([prompt[: -len(end)], end, *(end + label for label in labels)]),
        )

    def test_storage_holds_four_prompts_of_the_served_context(self):
        # Even text CPython stores at 4 bytes a character, as it does once
        # the text holds an emoji: each prompt stays and is read again.
        prompts = [
            f"<|im_start|>user\n{i} " + "Hello world!" * 400 + " \U0001f680<|im_end|>"
            for i in range(4)
        ]
        context = max(
            len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for prompt in prompts
        )
        cache = PromptTokenizer(self.tokenizer, QWEN.message_boundary, context)
        for prompt in prompts:
            self.assert_encoding(prompt, cache)
        for prompt in prompts:
            self.assert_encoding(prompt + "\n<|im_start|>assistant\n", cache)
        self.assertEqual(cache.stats()["entries"], 4)
        self.assertEqual(cache.stats()["hits"], 4)

    def test_interleaved_conversations_each_reuse_their_newest_prompt(self):
        # Agents interleave more conversations than there are lanes, each
        # waiting on its tool between turns: every next turn finds its own.
        prompts = [
            f"<|im_start|>user\n{i} " + "Hello world!" * 400 + "<|im_end|>\n"
            for i in range(8)
        ]
        for prompt in prompts:
            self.assert_encoding(prompt + "<|im_start|>assistant\n")
        for i, prompt in enumerate(prompts):
            self.assert_encoding(
                prompt
                + f"<|im_start|>assistant\n{i}<|im_end|>\n"
                + "<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n"
            )
        stats = self.cache.stats()
        self.assertEqual(
            (stats["hits"], stats["misses"], stats["evictions"], stats["entries"]),
            (8, 8, 0, 8),
        )

    def test_concurrent_requests_eviction_and_oversized_prefix(self):
        self.enterContext(mock.patch.object(PromptTokenizer, "CAPACITY", 2))
        self.enterContext(mock.patch.object(PromptTokenizer, "BUDGET_PROMPTS", 2))
        # Two prompts of 833 tokens: 39,984 bytes.
        cache = PromptTokenizer(self.tokenizer, QWEN.message_boundary, 833)
        budget = cache.stats()["budget_bytes"]
        prompts = [
            "<|im_start|>user\n" + str(i) + " Hello world!" * 400 + "<|im_end|>\n"
            for i in range(8)
        ]
        expected = [
            self.tokenizer(text, add_special_tokens=False)["input_ids"]
            for text in prompts
        ]
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(cache.encode, prompts * 3))
        self.assertEqual(results, expected * 3)
        stats = cache.stats()
        self.assertLessEqual(stats["entries"], 2)
        self.assertLessEqual(stats["bytes"], budget)
        # Each encode looked up its prompt, and each of the 8 kept in turn
        # but no longer kept was evicted.
        self.assertEqual(stats["hits"] + stats["misses"], 24)
        self.assertGreaterEqual(stats["evictions"], 8 - stats["entries"])
        before = stats["bytes"]
        self.assert_encoding("x" * 50000 + "<|im_end|>tail", cache)
        self.assertEqual(cache.stats()["bytes"], before)

    def test_lone_surrogates_encode_as_replacement_characters(self):
        # JSON spells lone surrogates, such as half of an emoji, which the
        # tokenizer refuses: text encodes with U+FFFD in their place, one for
        # one, before and after a cached boundary.
        prefix = "<|im_start|>user\n" + "Hello world!\n" * 400 + "<|im_end|>"
        for text, replaced in (
            ("Hello \ud83d world", "Hello \ufffd world"),
            ("\udfff\ud800", "\ufffd\ufffd"),
            (prefix + "\ud83d!", prefix + "\ufffd!"),
            ("a\ud83d" + prefix + "b", "a\ufffd" + prefix + "b"),
        ):
            with self.subTest(text=ascii(text)):
                self.assertEqual(tokenizable(text), replaced)
                self.assertEqual(len(tokenizable(text)), len(text))
                self.assertEqual(
                    self.cache.encode(text),
                    self.tokenizer(replaced, add_special_tokens=False)["input_ids"],
                )
        with self.assertRaises(TypeError):
            self.tokenizer("Hello \ud83d world", add_special_tokens=False)

    def test_short_prompts_and_missing_boundary_bypass_cache(self):
        for text in (
            "",
            "small<|im_end|>tail",
            "text " * 2000,
            "x" * 5000 + "<|im_end",
        ):
            self.assert_encoding(text)
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_a_marker_inside_another_added_token_is_no_boundary(self):
        t = tokenizer()
        t.backend_tokenizer.add_special_tokens(
            [AddedToken("<|im_end|>suffix", normalized=False)]
        )
        cache = PromptTokenizer(t, QWEN.message_boundary, NATIVE_CONTEXT)
        self.assertFalse(cache.enabled)
        text = "Hello world!" * 500 + "<|im_end|>suffix e\u0301"
        self.assertEqual(
            cache.encode(text), t(text, add_special_tokens=False)["input_ids"]
        )
        self.assertEqual(cache.stats()["entries"], 0)

    def test_partial_marker_consumed_by_another_added_token_is_not_cached(self):
        self.tokenizer.backend_tokenizer.add_special_tokens(
            [AddedToken("hello<|im_end", normalized=False)]
        )
        cache = PromptTokenizer(self.tokenizer, QWEN.message_boundary, NATIVE_CONTEXT)
        self.assertTrue(cache.enabled)
        text = "Hello world!" * 500 + "hello<|im_end|>suffix"
        self.assert_encoding(text, cache)
        self.assertEqual(cache.stats()["entries"], 0)

    def test_a_tokenizer_whose_added_tokens_cannot_be_quoted_is_refused(self):
        # A tokenizer that could not keep an added token out of the text a
        # client sent is refused, not served with the token left in: one
        # without a fast backend, one that finds an added token other than
        # as its literal text, or one whose pipeline reads a segment by
        # where it stands in the whole text.
        metaspace = Tokenizer(models.BPE())
        metaspace.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="first")
        metaspace.train_from_iterator(
            ["hello world tail"],
            trainers.BpeTrainer(vocab_size=50, show_progress=False),
        )
        metaspace.add_special_tokens([AddedToken("<|im_end|>", normalized=False)])
        metaspace = PreTrainedTokenizerFast(tokenizer_object=metaspace)
        # The text after the marker does not encode there as it does alone.
        whole = metaspace("hello<|im_end|>tail", add_special_tokens=False)
        self.assertNotEqual(
            whole["input_ids"][-1:],
            metaspace("tail", add_special_tokens=False)["input_ids"][-1:],
        )

        def changed(change):
            t = tokenizer()
            change(t.backend_tokenizer)
            return t

        def added(text, **options):
            return lambda backend: backend.add_special_tokens(
                [AddedToken(text, **options)]
            )

        for name, t, reason in (
            ("slow", SimpleNamespace(), "fast tokenizer"),
            ("pre-tokenizer", metaspace, "on its own"),
            (
                "normalizer",
                changed(
                    lambda backend: setattr(
                        backend, "normalizer", normalizers.Prepend(" ")
                    )
                ),
                "on its own",
            ),
            (
                "dropout",
                changed(lambda backend: setattr(backend.model, "dropout", 0.1)),
                "on its own",
            ),
            (
                "stripping",
                changed(added("<|im_end|>", normalized=False, rstrip=True)),
                "literal text",
            ),
            ("normalized", changed(added("<tag>", normalized=True)), "literal text"),
            ("one character", changed(added("\u00a7", normalized=False)), "quoted"),
        ):
            with self.subTest(name), self.assertRaisesRegex(ValueError, reason):
                PromptTokenizer(t, QWEN.message_boundary, NATIVE_CONTEXT)
