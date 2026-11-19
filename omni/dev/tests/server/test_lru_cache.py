import unittest

from server.lru import LRUCache


class LRUCacheTests(unittest.TestCase):
    def test_least_recently_used_values_leave_first_to_fit_the_budget(self):
        cache = LRUCache(10)
        self.assertTrue(cache.put("a", 1, 4))
        self.assertTrue(cache.put("b", 2, 4))
        # Reading a value makes it the most recently used.
        self.assertEqual(cache.get("a"), 1)
        self.assertTrue(cache.put("c", 3, 4))
        self.assertEqual(list(cache), ["a", "c"])
        self.assertIsNone(cache.get("b"))
        self.assertEqual((len(cache), cache.bytes, cache.evictions), (2, 8, 1))

    def test_capacity_bounds_the_entries(self):
        cache = LRUCache(100, capacity=2)
        for key in "abc":
            cache.put(key, key.upper(), 1)
        self.assertEqual(list(cache), ["b", "c"])
        self.assertEqual((cache.bytes, cache.evictions), (2, 1))

    def test_a_value_larger_than_the_budget_is_not_kept(self):
        cache = LRUCache(10)
        cache.put("a", 1, 6)
        for key in ("a", "b"):
            self.assertFalse(cache.put(key, 2, 11))
        self.assertEqual((list(cache), cache.get("a"), cache.bytes), (["a"], 1, 6))
        # A value as large as the whole budget is kept alone.
        self.assertTrue(cache.put("b", 2, 10))
        self.assertEqual((list(cache), cache.bytes, cache.evictions), (["b"], 10, 1))

    def test_a_value_replaces_the_one_kept_for_its_key(self):
        cache = LRUCache(10)
        cache.put("a", 1, 6)
        cache.put("b", 2, 2)
        self.assertTrue(cache.put("a", 3, 8))
        self.assertEqual(list(cache), ["b", "a"])
        self.assertEqual((cache.get("a"), cache.bytes, cache.evictions), (3, 10, 0))
        self.assertEqual(cache.pop("a"), 3)
        self.assertIsNone(cache.pop("a"))
        self.assertEqual((list(cache), cache.bytes), (["b"], 2))


if __name__ == "__main__":
    unittest.main()
