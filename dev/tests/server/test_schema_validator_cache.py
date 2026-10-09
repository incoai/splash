import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from jsonschema import SchemaError

from server import schema_validation as validation
from server import tool_schema
from server.errors import APIError
from server.lru import LRUCache


def empty_cache(name, budget=None, count=None):
    """A patch of schema_validation's cache `name` with an empty one, of the
    server's budget and entry count unless given."""
    cache = getattr(validation, name)
    return mock.patch.object(
        validation,
        name,
        LRUCache(
            cache.budget_bytes if budget is None else budget,
            cache.capacity if count is None else count,
        ),
    )


class ValidatorCacheTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(empty_cache("_validator_cache"))

    def build(self, schema):
        return validation.build_validator(schema)

    def test_eviction_obeys_both_count_and_source_byte_limits(self):
        for count, budget in ((2, 10000), (256, 180)):
            with (
                self.subTest(count=count, budget=budget),
                empty_cache("_validator_cache", budget, count),
            ):
                for i in range(20):
                    self.build({"type": "string", "description": str(i) + "x" * 30})
                    cache = validation._validator_cache
                    self.assertLessEqual(len(cache), count)
                    self.assertLessEqual(cache.bytes, budget)
                    self.assertEqual(cache.bytes, sum(len(key) for key in cache))

    def test_cache_hit_refreshes_lru(self):
        with empty_cache("_validator_cache", count=2):
            a, b, c = ({"const": i} for i in range(3))
            first = self.build(a)
            second = self.build(b)
            self.assertIs(first, self.build(a))
            self.build(c)
            self.assertIs(first, self.build(a))
            self.assertIsNot(second, self.build(b))

    def test_oversized_schema_is_validated_without_retention(self):
        with empty_cache("_validator_cache", budget=100):
            schema = {"type": "integer", "description": "x" * 200}
            first = self.build(schema)
            self.assertTrue(first.is_valid(1))
            self.assertFalse(first.is_valid("1"))
            self.assertIsNot(first, self.build(schema))
            self.assertEqual(validation._validator_cache.bytes, 0)
            self.assertFalse(validation._validator_cache)
            with self.assertRaises(SchemaError):
                self.build({**schema, "type": "invalid"})

    def test_invalid_schema_is_not_cached(self):
        for _ in range(2):
            with self.assertRaises(SchemaError):
                self.build({"type": "invalid"})
        self.assertFalse(validation._validator_cache)
        self.assertEqual(validation._validator_cache.bytes, 0)

    def test_unevaluated_properties_cannot_reach_unbounded_patterns(self):
        # jsonschema matches these patterns with the standard-library engine,
        # which has no time limit, however the keywords are connected.
        patterns = {"patternProperties": {"^(a+)+$": {"type": "integer"}}}
        closed = {"unevaluatedProperties": False}
        draft = {"$schema": "https://json-schema.org/draft/2019-09/schema"}
        for schema in (
            {**patterns, **closed},
            {"allOf": [patterns], **closed},
            {"$ref": "#/x-stash", "x-stash": patterns, **closed},
            {**draft, **patterns, **closed},
        ):
            with self.subTest(schema=schema), self.assertRaises(APIError) as caught:
                self.build(schema)
            self.assertEqual(caught.exception.status, 400)
        self.assertFalse(validation._validator_cache)
        self.assertFalse(self.build(patterns).is_valid({"aa": "1"}))
        validator = self.build({"properties": {"a": {}}, **closed})
        self.assertTrue(validator.is_valid({"a": 1}))
        self.assertFalse(validator.is_valid({"b": 1}))

    def test_references_keep_patterns_bounded(self):
        # A reference into an unknown keyword reaches a $schema the build
        # leaves, and jsonschema would validate there with that dialect's
        # class, which matches patterns with the unbounded standard engine.
        for container, dialect in (
            ("$defs", "https://json-schema.org/draft/2020-12/schema"),
            ("x-stash", "https://json-schema.org/draft/2020-12/schema"),
            ("x-stash", "http://json-schema.org/draft-07/schema#"),
        ):
            target = {"$schema": dialect, "type": "string", "pattern": "^a+$"}
            validator = self.build(
                {"$ref": f"#/{container}/s", container: {"s": target}}
            )
            with (
                self.subTest(container=container, dialect=dialect),
                mock.patch.object(validation.regex, "search", side_effect=TimeoutError),
                self.assertRaises(validation.SchemaEvaluationError),
            ):
                validator.is_valid("aa")

    def test_references_keep_their_dialect(self):
        # The draft-07 target's items list is a schema per position, which
        # the document's 2020-12 dialect would refuse.
        target = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "items": [{"type": "string"}],
            "additionalItems": False,
        }
        validator = self.build({"$ref": "#/x-stash/s", "x-stash": {"s": target}})
        self.assertTrue(validator.is_valid(["a"]))
        self.assertFalse(validator.is_valid(["a", 1]))

    def test_caller_mutation_does_not_change_cached_validator(self):
        schema = {"properties": {"x": {"type": "integer"}}}
        first = self.build(schema)
        schema["properties"]["x"]["type"] = "string"
        second = self.build(schema)
        self.assertTrue(first.is_valid({"x": 1}))
        self.assertFalse(first.is_valid({"x": "1"}))
        self.assertFalse(second.is_valid({"x": 1}))
        self.assertTrue(second.is_valid({"x": "1"}))

    def test_concurrent_misses_account_for_one_entry(self):
        barrier = threading.Barrier(4)
        bounded_class = validation._bounded_class

        def delayed_class(base):
            barrier.wait(timeout=5)
            return bounded_class(base)

        with (
            mock.patch.object(validation, "_bounded_class", delayed_class),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            results = list(pool.map(self.build, [{"type": "integer"}] * 4))
        self.assertTrue(all(v is results[0] for v in results))
        cache = validation._validator_cache
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.bytes, sum(len(key) for key in cache))


class SchemaCheckCacheTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(empty_cache("_checked_schemas"))
        self.enterContext(empty_cache("_validator_cache"))

    @staticmethod
    def tools(*schemas):
        tools = [
            {
                "type": "function",
                "function": {"name": f"t{index}", "parameters": schema},
            }
            for index, schema in enumerate(schemas)
        ]
        return tool_schema.normalize_tools(tools, "auto", True)

    def test_tool_schemas_are_checked_once_without_validators(self):
        schemas = [{"type": "object", "properties": {"x": {"type": "integer"}}}] * 2
        self.tools(*schemas)
        self.assertEqual(len(validation._checked_schemas), 1)
        # Response formats keep the validators' slots to themselves.
        self.assertFalse(validation._validator_cache)
        # The next turn's tools are not checked again.
        with mock.patch.object(validation.validators, "validator_for") as check:
            self.tools(*schemas)
        check.assert_not_called()

    def test_invalid_tool_schema_is_refused_every_time(self):
        for _ in range(2):
            with self.assertRaises(APIError) as caught:
                self.tools({"type": "invalid"})
            self.assertEqual(caught.exception.status, 400)
            self.assertTrue(
                caught.exception.message.startswith("invalid tool schema for t0: ")
            )
        self.assertFalse(validation._checked_schemas)

    def test_eviction_obeys_both_count_and_source_byte_limits(self):
        for count, budget in ((2, 10000), (1024, 180)):
            with (
                self.subTest(count=count, budget=budget),
                empty_cache("_checked_schemas", budget, count),
            ):
                for i in range(20):
                    validation.check_schema(
                        {"type": "string", "description": str(i) + "x" * 30}
                    )
                    checked = validation._checked_schemas
                    self.assertLessEqual(len(checked), count)
                    self.assertLessEqual(checked.bytes, budget)
                    self.assertEqual(checked.bytes, sum(len(key) for key in checked))

    def test_tools_take_schemas_only_a_validator_could_not_evaluate(self):
        # Calls are not validated, so a tool's schema need only be valid; a
        # response format is, and refuses what jsonschema would evaluate
        # without bounds.
        schema = {"patternProperties": {"^a": {}}, "unevaluatedProperties": False}
        self.tools(schema)
        with self.assertRaises(APIError):
            tool_schema.normalize_response_format(
                {"type": "json_schema", "json_schema": {"schema": schema}}
            )
