import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from server import constraints


class ConstraintCacheTests(unittest.TestCase):
    def factory(self, budget=12):
        class Matcher:
            @staticmethod
            def validate_grammar(grammar, tokenizer):
                return None

            def __init__(self, tokenizer, grammar, log_level):
                self.grammar = grammar

            def is_error(self):
                return False

            def deep_copy(self):
                return self.grammar

        constraint = mock.Mock(side_effect=lambda matcher, executor: matcher)
        constraint.VOCABULARY = constraints.TokenConstraint.VOCABULARY
        constraint.EOS_TOKENS = constraints.TokenConstraint.EOS_TOKENS
        for target, replacement in (
            ("guidance_tokenizer", lambda *args, **kwargs: None),
            ("LLMatcher", Matcher),
            ("LLExecutor", lambda: None),
            ("TokenConstraint", constraint),
        ):
            patch = mock.patch.object(constraints, target, replacement)
            patch.start()
            self.addCleanup(patch.stop)
        return constraints.ConstraintFactory(object(), cache_source_bytes=budget)

    def test_byte_budget_evicts_lru_and_counts_utf8(self):
        factory = self.factory()
        for grammar in ("one", "two", "one", "é" * 4):
            self.assertEqual(factory.create(grammar), grammar)
        self.assertEqual(list(factory.cache), ["one", "é" * 4])
        self.assertEqual(factory.stats()["source_bytes"], 11)
        self.assertEqual(factory.stats()["hits"], 1)

    def test_oversized_grammar_is_usable_without_displacing_cache(self):
        factory = self.factory()
        factory.create("warm")
        for _ in range(2):
            self.assertEqual(factory.create("x" * 13), "x" * 13)
        self.assertEqual(list(factory.cache), ["warm"])
        self.assertEqual(factory.stats()["source_bytes"], 4)
        self.assertEqual(factory.stats()["misses"], 3)
        factory.create("warm")
        self.assertEqual(factory.stats()["hits"], 1)

    def test_concurrent_churn_remains_bounded(self):
        factory = self.factory()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(factory.create, (str(i) for i in range(200))))
        self.assertEqual(results, [str(i) for i in range(200)])
        self.assertLessEqual(factory.source_bytes, 12)
        self.assertEqual(
            factory.source_bytes, sum(len(key.encode()) for key in factory.cache)
        )
        self.assertLessEqual(len(factory.cache), factory.cache_size)

    def test_invalid_budget_is_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.factory(value)
