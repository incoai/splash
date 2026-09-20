import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from unittest import mock

from server import constraints


class ConstraintCacheTests(unittest.TestCase):
    def factory(self, budget=12, validate=None):
        class Matcher:
            @staticmethod
            def validate_grammar(grammar, tokenizer):
                return None if validate is None else validate(grammar)

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

    def test_cold_compile_does_not_block_hits_stats_or_other_compilation(self):
        entered, release = threading.Event(), threading.Event()

        def validate(grammar):
            if grammar == "cold":
                entered.set()
                self.assertTrue(release.wait(5))

        factory = self.factory(validate=validate)
        factory.create("hot")
        with ThreadPoolExecutor(max_workers=2) as pool:
            cold = pool.submit(factory.create, "cold")
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(pool.submit(factory.create, "hot").result(2), "hot")
                self.assertEqual(pool.submit(factory.stats).result(2)["hits"], 1)
                self.assertEqual(
                    pool.submit(factory.create, "other").result(2), "other"
                )
            finally:
                release.set()
            self.assertEqual(cold.result(2), "cold")

    def test_concurrent_oversized_miss_and_failure_are_shared(self):
        entered, joined, release = (threading.Event() for _ in range(3))
        builds = []
        failure = None

        def observed_wait(*args, **kwargs):
            joined.set()
            return wait(*args, **kwargs)

        def validate(grammar):
            builds.append(grammar)
            entered.set()
            self.assertTrue(release.wait(5))
            if isinstance(failure, Exception):
                raise failure
            return failure

        factory = self.factory(budget=1, validate=validate)
        with mock.patch.object(constraints, "wait", observed_wait):
            for failure in (None, "invalid schema", TimeoutError("compiler failed")):
                entered.clear()
                joined.clear()
                release.clear()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(factory.create, "oversized")
                    try:
                        self.assertTrue(entered.wait(2))
                        second = pool.submit(factory.create, "oversized")
                        self.assertTrue(joined.wait(2))
                    finally:
                        release.set()
                    for result in (first, second):
                        if failure is None:
                            self.assertEqual(result.result(2), "oversized")
                        elif isinstance(failure, Exception):
                            with self.assertRaisesRegex(type(failure), str(failure)):
                                result.result(2)
                        else:
                            with self.assertRaisesRegex(constraints.APIError, failure):
                                result.result(2)
                self.assertFalse(factory.pending)
                self.assertFalse(factory.cache)
        self.assertEqual(builds, ["oversized"] * 3)
        # A failed build must not poison subsequent attempts.
        failure = None
        self.assertEqual(factory.create("oversized"), "oversized")

    def test_waiter_timeout_does_not_cancel_shared_compilation(self):
        entered, release = threading.Event(), threading.Event()

        def validate(grammar):
            entered.set()
            self.assertTrue(release.wait(5))

        factory = self.factory(validate=validate)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(factory.create, "shared")
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaisesRegex(constraints.APIError, "request timed out"):
                    factory.create("shared", timeout=0.01)
            finally:
                release.set()
            self.assertEqual(first.result(2), "shared")
        self.assertEqual(factory.create("shared"), "shared")
        self.assertEqual(factory.stats()["misses"], 1)

    def test_real_matchers_compile_and_copy_independently_under_concurrency(self):
        from dev.tests.engine.test_structured_tools import StructuredToolGrammarTest

        StructuredToolGrammarTest.setUpClass()
        with mock.patch.object(
            constraints,
            "guidance_tokenizer",
            return_value=StructuredToolGrammarTest.guidance,
        ):
            factory = constraints.ConstraintFactory(object())
        barrier = threading.Barrier(8)

        def generate(index):
            barrier.wait(2)
            schema = {"const": str(index)} if distinct else {"type": "boolean"}
            grammar = json.dumps({"grammars": [{"json_schema": schema}]})
            constraint = factory.create(grammar)
            text = json.dumps(str(index) if distinct else bool(index % 2))
            tokens = StructuredToolGrammarTest.tokenizer.encode(text).ids
            self.assertTrue(constraint.matcher.consume_tokens(tokens))
            self.assertTrue(constraint.matcher.is_accepting())
            return constraint

        for distinct in (False, True):
            with ThreadPoolExecutor(max_workers=8) as pool:
                instances = list(pool.map(generate, range(8)))
            self.assertEqual(len({id(instance.matcher) for instance in instances}), 8)
        self.assertEqual(factory.stats()["misses"], 9)
        self.assertEqual(factory.stats()["hits"], 7)
