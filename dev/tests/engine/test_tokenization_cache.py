import concurrent.futures
import threading
import time
import unittest
from unittest import mock

from server import tokenization
from server.errors import APIError
from server.tokenization import TokenizationCache


class TokenizationCacheTests(unittest.TestCase):
    @staticmethod
    def tokenizer(text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(char) for char in text]}
        if add_special_tokens:
            result["input_ids"].insert(0, 1)
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def test_exact_text_options_offsets_and_owned_outputs(self):
        cache = TokenizationCache(self.tokenizer)
        deadline = time.monotonic() + 10
        for text in ("你好🌊", " hello ", ""):
            for special in (False, True):
                for offsets in (False, True):
                    expected = self.tokenizer(text, special, offsets)
                    first = cache.encode(
                        text, deadline, add_special_tokens=special, offsets=offsets
                    )
                    self.assertEqual(first, expected)
                    first["input_ids"].append(99)
                    if offsets:
                        first["offset_mapping"].clear()
                    self.assertEqual(
                        cache.encode(
                            text, deadline, add_special_tokens=special, offsets=offsets
                        ),
                        expected,
                    )
        self.assertEqual(cache.stats()["hits"], 12)

    def test_return_shape_is_independent_of_cache_admission(self):
        def tokenizer(text, **options):
            return {
                **self.tokenizer(text, **options),
                "attention_mask": [1] * len(text),
            }

        for limits in ({}, {"budget_bytes": 1}, {"max_entries": 0}):
            with self.subTest(limits=limits):
                cache = TokenizationCache(tokenizer, **limits)
                for offsets in (False, True):
                    expected = self.tokenizer("same", return_offsets_mapping=offsets)
                    for _ in range(2):
                        self.assertEqual(
                            cache.encode(
                                "same", time.monotonic() + 10, offsets=offsets
                            ),
                            expected,
                        )

    def test_inflight_results_and_failures_are_shared_without_lru_admission(self):
        for limits in ({"budget_bytes": 1}, {"max_entries": 0}):
            for fails in (False, True):
                with self.subTest(limits=limits, fails=fails):
                    entered = threading.Event()
                    release = threading.Event()
                    joined = threading.Event()
                    calls = []

                    def tokenizer(text, **options):
                        calls.append(text)
                        entered.set()
                        if not release.wait(5):
                            raise RuntimeError("test timed out")
                        if fails:
                            raise ValueError("bad input")
                        return self.tokenizer(text, **options)

                    def wait(futures, **options):
                        joined.set()
                        return concurrent.futures.wait(futures, **options)

                    cache = TokenizationCache(tokenizer, **limits)
                    with (
                        mock.patch.object(tokenization, "wait", side_effect=wait),
                        concurrent.futures.ThreadPoolExecutor(2) as pool,
                    ):
                        first = pool.submit(cache.encode, "same", time.monotonic() + 10)
                        try:
                            self.assertTrue(entered.wait(5))
                            second = pool.submit(
                                cache.encode, "same", time.monotonic() + 10
                            )
                            self.assertTrue(joined.wait(5))
                        finally:
                            release.set()
                        if fails:
                            for future in (first, second):
                                with self.assertRaisesRegex(ValueError, "bad input"):
                                    future.result(5)
                        else:
                            one, two = first.result(5), second.result(5)
                            self.assertEqual(one, two)
                            one["input_ids"].clear()
                            self.assertEqual(two, self.tokenizer("same"))
                    self.assertEqual(calls, ["same"])
                    self.assertEqual(cache.stats()["entries"], 0)
                    self.assertEqual(cache.stats()["pending"], 0)
                    if fails:
                        with self.assertRaises(ValueError):
                            cache.encode("same", time.monotonic() + 10)
                    else:
                        cache.encode("same", time.monotonic() + 10)
                    self.assertEqual(calls, ["same", "same"])

    def test_owner_timeout_does_not_discard_a_waiters_result(self):
        entered = threading.Event()
        release = threading.Event()
        joined = threading.Event()
        owner_deadline = time.monotonic() + 10
        remaining = tokenization.remaining_request_time

        def tokenizer(text, **options):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timed out")
            return self.tokenizer(text, **options)

        def check_deadline(deadline):
            if deadline == owner_deadline and release.is_set():
                raise APIError(504, "request timed out", "request_timeout")
            return remaining(deadline)

        def wait(futures, **options):
            joined.set()
            return concurrent.futures.wait(futures, **options)

        cache = TokenizationCache(tokenizer, max_entries=0)
        with (
            mock.patch.object(
                tokenization, "remaining_request_time", side_effect=check_deadline
            ),
            mock.patch.object(tokenization, "wait", side_effect=wait),
            concurrent.futures.ThreadPoolExecutor(2) as pool,
        ):
            owner = pool.submit(cache.encode, "same", owner_deadline)
            try:
                self.assertTrue(entered.wait(5))
                waiter = pool.submit(cache.encode, "same", owner_deadline + 10)
                self.assertTrue(joined.wait(5))
            finally:
                release.set()
            with self.assertRaises(APIError) as error:
                owner.result(5)
            self.assertEqual(error.exception.status, 504)
            self.assertEqual(waiter.result(5), self.tokenizer("same"))
        self.assertEqual(cache.stats()["pending"], 0)

    def test_byte_budget_lru_and_oversize_bypass(self):
        cache = TokenizationCache(self.tokenizer, budget_bytes=16)
        deadline = time.monotonic() + 10
        for text in ("aa", "bb", "aa", "cc"):
            cache.encode(text, deadline)
        self.assertEqual(cache.stats()["bytes"], 16)
        cache.encode("oversize", deadline)
        self.assertEqual(cache.stats()["bytes"], 16)
        hits = cache.stats()["hits"]
        cache.encode("aa", deadline)
        self.assertEqual(cache.stats()["hits"], hits + 1)
        misses = cache.stats()["misses"]
        cache.encode("bb", deadline)
        self.assertEqual(cache.stats()["misses"], misses + 1)

    def test_entry_limit_bounds_empty_inputs(self):
        cache = TokenizationCache(
            lambda text, **kwargs: {"input_ids": []}, max_entries=2
        )
        for text in ("a", "b", "c"):
            cache.encode(text, time.monotonic() + 10)
        self.assertEqual(cache.stats()["entries"], 2)
        self.assertEqual(cache.stats()["bytes"], 0)

    def test_duplicate_work_waiter_deadline_and_failure_recovery(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def tokenizer(text, **options):
            calls.append(text)
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timed out")
            return self.tokenizer(text, **options)

        cache = TokenizationCache(tokenizer)
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            first = pool.submit(cache.encode, "same", time.monotonic() + 10)
            self.assertTrue(entered.wait(5))
            second = pool.submit(cache.encode, "same", time.monotonic() + 10)
            try:
                with self.assertRaises(APIError) as error:
                    cache.encode("same", time.monotonic() + 0.02)
                self.assertEqual(error.exception.status, 504)
            finally:
                release.set()
            self.assertEqual(first.result(), second.result())
        self.assertEqual(calls, ["same"])
        self.assertEqual(cache.stats()["pending"], 0)

        failures = TokenizationCache(
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad input"))
        )
        for _ in range(2):
            with self.assertRaises(ValueError):
                failures.encode("bad", time.monotonic() + 10)
        self.assertEqual(failures.stats()["pending"], 0)


if __name__ == "__main__":
    unittest.main()
