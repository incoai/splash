import random
import unittest
from concurrent.futures import ThreadPoolExecutor

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

from server.tokenization import PromptTokenizer


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
        self.cache = PromptTokenizer(self.tokenizer)
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

    def test_concurrent_requests_eviction_and_oversized_prefix(self):
        cache = PromptTokenizer(self.tokenizer, capacity=2, budget_bytes=40000)
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
        self.assertLessEqual(stats["bytes"], 40000)
        before = stats["bytes"]
        self.assert_encoding("x" * 50000 + "<|im_end|>tail", cache)
        self.assertEqual(cache.stats()["bytes"], before)

    def test_short_prompts_and_missing_boundary_bypass_cache(self):
        for text in (
            "",
            "small<|im_end|>tail",
            "text " * 2000,
            "x" * 5000 + "<|im_end",
        ):
            self.assert_encoding(text)
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_unsupported_normalization_and_stripping_tokens_fall_back(self):
        for mode in ("normalizer", "strip", "overlap", "dropout"):
            with self.subTest(mode=mode):
                t = tokenizer()
                if mode == "normalizer":
                    t.backend_tokenizer.normalizer = normalizers.Prepend(" ")
                elif mode == "strip":
                    t.backend_tokenizer.add_special_tokens(
                        [AddedToken("<|im_end|>", normalized=False, rstrip=True)]
                    )
                elif mode == "overlap":
                    t.backend_tokenizer.add_special_tokens(
                        [AddedToken("<|im_end|>suffix", normalized=False)]
                    )
                else:
                    t.backend_tokenizer.model.dropout = 0.1
                cache = PromptTokenizer(t)
                self.assertFalse(cache.enabled)
                if mode == "dropout":
                    # Check fallback rather than comparing two random encodes.
                    t.backend_tokenizer.model.dropout = None
                text = "Hello world!" * 500 + "<|im_end|>suffix e\u0301"
                self.assertEqual(
                    cache.encode(text), t(text, add_special_tokens=False)["input_ids"]
                )
                self.assertEqual(cache.stats()["entries"], 0)

    def test_partial_marker_consumed_by_another_added_token_is_not_cached(self):
        self.tokenizer.backend_tokenizer.add_special_tokens(
            [AddedToken("hello<|im_end", normalized=False)]
        )
        cache = PromptTokenizer(self.tokenizer)
        self.assertTrue(cache.enabled)
        text = "Hello world!" * 500 + "hello<|im_end|>suffix"
        self.assert_encoding(text, cache)
        self.assertEqual(cache.stats()["entries"], 0)

    def test_input_start_sensitive_pretokenizer_falls_back(self):
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="first")
        backend.post_processor = processors.ByteLevel(trim_offsets=False)
        backend.train_from_iterator(
            ["hello world tail"],
            trainers.BpeTrainer(vocab_size=50, show_progress=False),
        )
        backend.add_special_tokens([AddedToken("<|im_end|>", normalized=False)])
        t = PreTrainedTokenizerFast(tokenizer_object=backend)
        cache = PromptTokenizer(t)
        self.assertFalse(cache.enabled)
        prefix, tail = "hello world " * 500 + "<|im_end|>", "tail"
        expected = t(prefix + tail, add_special_tokens=False)["input_ids"]
        split = t(prefix, add_special_tokens=False)["input_ids"]
        split += t(tail, add_special_tokens=False)["input_ids"]
        self.assertNotEqual(split, expected)
        self.assertEqual(cache.encode(prefix + tail), expected)
        self.assertEqual(cache.stats()["entries"], 0)
