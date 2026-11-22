import dataclasses
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from types import SimpleNamespace
from unittest import mock

from dev.tests.server_fixtures import READY, grammar_tokenizers
from server import constraints
from server import server as api
from server.chat_format import QWEN

# A grammars' table of READY's vocabulary, for factories whose matchers the
# tests replace.
TABLE = SimpleNamespace(vocab_size=READY.vocabulary_size)


def ready_for(table):
    """READY with the vocabulary of a real table."""
    return dataclasses.replace(READY, vocabulary_size=table.vocab_size)


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

        constraint = mock.Mock(side_effect=lambda matcher, _format: matcher)
        for target, replacement in (
            ("LLMatcher", Matcher),
            ("TokenConstraint", constraint),
        ):
            self.enterContext(mock.patch.object(constraints, target, replacement))
        self.enterContext(
            mock.patch.object(
                constraints.ConstraintFactory, "CACHE_SOURCE_BYTES", budget
            )
        )
        return constraints.ConstraintFactory(TABLE, READY)

    def test_table_takes_the_installations_vocabulary_and_the_formats_stops(self):
        """The table is built before Ready, of the installation's vocabulary
        and the chat format's stop texts."""
        tokenizer, _ = grammar_tokenizers()
        fast = SimpleNamespace(
            encode=lambda text, add_special_tokens: [tokenizer.token_to_id(text)]
        )
        chat_format = SimpleNamespace(stop_tokens=("<eos>",))
        with mock.patch.object(constraints, "guidance_tokenizer") as build:
            build.return_value = SimpleNamespace(vocab_size=300)
            table = constraints.ConstraintFactory.table(fast, chat_format, 300)
        build.assert_called_once()
        self.assertIs(build.call_args.args[0], fast)
        self.assertEqual(build.call_args.kwargs["n_vocab"], 300)
        self.assertEqual(
            build.call_args.kwargs["eos_token"], [tokenizer.token_to_id("<eos>")]
        )
        ready = dataclasses.replace(READY, vocabulary_size=300)
        self.assertIs(constraints.ConstraintFactory(table, ready).tokenizer, table)

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
        self.assertLessEqual(factory.cache.bytes, 12)
        self.assertEqual(
            factory.cache.bytes, sum(len(key.encode()) for key in factory.cache)
        )
        self.assertLessEqual(len(factory.cache), factory.CACHE_SIZE)

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

    def test_compiler_errors_are_reported_without_internals(self):
        from server import tool_schema

        _, guidance = grammar_tokenizers()
        factory = constraints.ConstraintFactory(guidance, ready_for(guidance))
        unsatisfiable = {"type": "array", "minItems": 5, "maxItems": 2}
        # More grammar symbols than the compiler can index make it panic.
        array = {"type": "array", "maxItems": tool_schema.MAX_GRAMMAR_BOUND}
        oversized = {"properties": {f"p{i}": array for i in range(1000)}}
        for schema, reason in (
            (unsatisfiable, "minItems (5) is greater than maxItems (2)"),
            (oversized, "tool or output schema is too large to compile"),
        ):
            with (
                self.subTest(reason=reason),
                self.assertRaises(constraints.APIError) as caught,
            ):
                factory.create(
                    tool_schema.json_grammar(schema, False, chat_format=QWEN)
                )
            self.assertEqual(caught.exception.status, 400)
            self.assertIn(reason, caught.exception.message)
            self.assertNotIn("\n", caught.exception.message)
            self.assertNotIn("%llguidance", caught.exception.message)

    def test_real_matchers_compile_and_copy_independently_under_concurrency(self):
        tokenizer, guidance = grammar_tokenizers()
        factory = constraints.ConstraintFactory(guidance, ready_for(guidance))
        barrier = threading.Barrier(8)

        def generate(index):
            barrier.wait(2)
            schema = {"const": str(index)} if distinct else {"type": "boolean"}
            grammar = json.dumps({"grammars": [{"json_schema": schema}]})
            constraint = factory.create(grammar)
            text = json.dumps(str(index) if distinct else bool(index % 2))
            tokens = tokenizer.encode(text).ids
            self.assertTrue(constraint.matcher.consume_tokens(tokens))
            self.assertTrue(constraint.matcher.is_accepting())
            return constraint

        for distinct in (False, True):
            with ThreadPoolExecutor(max_workers=8) as pool:
                instances = list(pool.map(generate, range(8)))
            self.assertEqual(len({id(instance.matcher) for instance in instances}), 8)
        self.assertEqual(factory.stats()["misses"], 9)
        self.assertEqual(factory.stats()["hits"], 7)

    def test_constraint_factory_caches_pristine_matchers_with_lru(self):
        testcase = self

        class Matcher:
            builds = []

            @staticmethod
            def validate_grammar(grammar, tokenizer):
                testcase.assertIs(tokenizer, TABLE)
                return "invalid" if grammar == "bad" else None

            def __init__(self, tokenizer, grammar, log_level=0):
                testcase.assertEqual((tokenizer, log_level), (TABLE, 0))
                self.grammar = grammar
                self.builds.append(grammar)

            def is_error(self):
                return False

            def deep_copy(self):
                return SimpleNamespace(grammar=self.grammar, copy=True)

        self.enterContext(
            mock.patch.object(constraints.ConstraintFactory, "CACHE_SIZE", 2)
        )
        with (
            mock.patch("server.constraints.LLMatcher", Matcher),
            mock.patch(
                "server.constraints.TokenConstraint",
                side_effect=lambda matcher, _format: (matcher,),
            ),
        ):
            factory = constraints.ConstraintFactory(TABLE, READY)
            first = factory.create("one")
            second = factory.create("one")
            factory.create("two")
            factory.create("three")
            factory.create("one")
            with self.assertRaisesRegex(api.APIError, "invalid"):
                factory.create("bad")

        self.assertIsNot(first[0], second[0])
        self.assertEqual(Matcher.builds, ["one", "two", "three", "one"])
        self.assertEqual(
            factory.stats(),
            {
                "entries": 2,
                "capacity": 2,
                "source_bytes": 8,
                "source_budget_bytes": factory.CACHE_SOURCE_BYTES,
                "hits": 1,
                "misses": 4,
            },
        )

    def test_constraint_factory_shares_concurrent_cold_grammar_build(self):
        class Matcher:
            builds = 0

            @staticmethod
            def validate_grammar(_grammar, _tokenizer):
                return None

            def __init__(self, _tokenizer, grammar, log_level=0):
                self.grammar = grammar
                type(self).builds += 1

            def is_error(self):
                return False

            def deep_copy(self):
                return SimpleNamespace(grammar=self.grammar)

        with (
            mock.patch("server.constraints.LLMatcher", Matcher),
            mock.patch(
                "server.constraints.TokenConstraint",
                side_effect=lambda matcher, _format: matcher,
            ),
        ):
            factory = constraints.ConstraintFactory(TABLE, READY)
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(factory.create, ["shared"] * 4))

        self.assertTrue(all(result.grammar == "shared" for result in results))
        self.assertEqual(Matcher.builds, 1)
        self.assertEqual(factory.stats()["hits"], 3)

    def test_constraint_factory_checks_prefixes_when_it_compiles(self):
        class Matcher:
            @staticmethod
            def validate_grammar(_grammar, _tokenizer):
                return None

            def __init__(self, _tokenizer, grammar, log_level=0):
                self.grammar = grammar

            def is_error(self):
                return False

            def deep_copy(self):
                return self

            def consume_tokens(self, tokens):
                return tokens != [9]

        compiled = []

        def prefixes(*extra):
            compiled.append(extra)
            return [([1], "fine"), *extra]

        with (
            mock.patch("server.constraints.LLMatcher", Matcher),
            mock.patch(
                "server.constraints.TokenConstraint",
                side_effect=lambda matcher, _format: matcher,
            ),
        ):
            factory = constraints.ConstraintFactory(TABLE, READY)
            for _ in range(2):
                with self.assertRaisesRegex(api.APIError, "^too wide$") as caught:
                    factory.create("wide", prefixes=lambda: prefixes(([9], "too wide")))
                self.assertEqual(caught.exception.status, 400)
            factory.create("narrow", prefixes=prefixes)
            factory.create("narrow", prefixes=prefixes)

        # A failing grammar is not cached; a cached one is not checked again.
        self.assertEqual(compiled, [(([9], "too wide"),)] * 2 + [()])
        self.assertEqual(list(factory.cache), ["narrow"])
