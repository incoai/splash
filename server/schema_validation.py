"""JSON Schema validation with bounded regular-expression evaluation."""

import copy
import json
import threading
from collections import OrderedDict
from functools import lru_cache

import attrs
import regex
from jsonschema import ValidationError, validators
from jsonschema.exceptions import UndefinedTypeCheck
from referencing import Registry

from .errors import APIError


class SchemaEvaluationError(Exception):
    """A validator could not evaluate an already accepted schema."""


def _matches(pattern, value):
    try:
        return regex.search(pattern, value, timeout=0.05, concurrent=True) is not None
    except TimeoutError:
        raise SchemaEvaluationError(
            "schema pattern exceeded its evaluation time limit"
        ) from None
    except regex.error:
        raise SchemaEvaluationError("schema pattern could not be evaluated") from None


def _pattern(validator, pattern, instance, schema):
    if validator.is_type(instance, "string") and not _matches(pattern, instance):
        yield ValidationError("string does not match its pattern")


def _pattern_properties(validator, patterns, instance, schema):
    if validator.is_type(instance, "object"):
        for pattern, subschema in patterns.items():
            for key, value in instance.items():
                if _matches(pattern, key):
                    yield from validator.descend(
                        value, subschema, path=key, schema_path=pattern
                    )


def _additional_properties(validator, additional, instance, schema):
    if validator.is_type(instance, "object"):
        properties = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        for key, value in instance.items():
            if key not in properties and not any(
                _matches(pattern, key) for pattern in patterns
            ):
                if additional is False:
                    yield ValidationError(
                        "additional property is not allowed", path=[key]
                    )
                elif isinstance(additional, dict):
                    yield from validator.descend(value, additional, path=key)


def json_objects(value):
    """Yield every object in a JSON document."""
    pending = [value]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            yield value
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


# JSON Schema keywords whose values are schemas: maps from names to schemas,
# then single schemas or lists of schemas. ``dependencies`` holds a schema or
# a list of property names per entry and is told apart by shape. Draft 3's
# ``type`` and ``disallow`` lists may hold schemas beside type names.
SCHEMA_MAP_KEYWORDS = {
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
}
SUBSCHEMA_KEYWORDS = {
    "items",
    "prefixItems",
    "additionalItems",
    "contains",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
    "propertyNames",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
    "extends",
}


def subschemas(schema):
    """Yield ``schema`` and, depth first, every schema nested under it.

    Only schema positions are visited, so property names and literal const,
    enum, default and examples data are never mistaken for schemas.
    """
    yield schema
    if not isinstance(schema, dict):
        return
    for key, item in schema.items():
        if key in SCHEMA_MAP_KEYWORDS and isinstance(item, dict):
            children = item.values()
        elif key == "dependencies" and isinstance(item, dict):
            children = (child for child in item.values() if not isinstance(child, list))
        elif key in SUBSCHEMA_KEYWORDS:
            children = item if isinstance(item, list) else (item,)
        elif key in ("type", "disallow") and isinstance(item, list):
            children = (child for child in item if isinstance(child, dict))
        else:
            continue
        for child in children:
            yield from subschemas(child)


def _known_type(base, name):
    try:
        base.TYPE_CHECKER.is_type(None, name)
    except UndefinedTypeCheck:
        return False
    return True


@lru_cache(maxsize=8)
def _bounded_class(base):
    bounded = validators.extend(
        base,
        {
            "pattern": _pattern,
            "patternProperties": _pattern_properties,
            "additionalProperties": _additional_properties,
        },
    )
    evolve = bounded.evolve

    # evolve validates a schema with the standard class its $schema names. A
    # reference can reach a declaration under any keyword, not only at the
    # positions build_validator removes them from: keep its dialect, bounded.
    def bounded_evolve(self, **changes):
        schema = changes.get("schema")
        if isinstance(schema, dict) and "$schema" in schema:
            dialect = base
            if isinstance(schema["$schema"], str):
                dialect = validators.validator_for(schema, default=base)
            changes["schema"] = {k: v for k, v in schema.items() if k != "$schema"}
            if dialect is not base:
                # As evolve builds its class, with the dialect's bounded one.
                for field in attrs.fields(bounded):
                    if field.init and field.alias not in changes:
                        changes[field.alias] = getattr(self, field.name)
                return _bounded_class(dialect)(**changes)
        return evolve(self, **changes)

    bounded.evolve = bounded_evolve
    return bounded


_VALIDATOR_CACHE_SIZE = 256
_VALIDATOR_CACHE_SOURCE_BYTES = 8 * 1024 * 1024
_validator_cache_lock = threading.Lock()
_validator_cache = OrderedDict()
_validator_cache_bytes = 0
# Empty: callers refuse remote references, so a schema refers only to itself.
_REGISTRY = Registry()


def build_validator(schema):
    global _validator_cache_bytes
    # check_schema walks the whole JSON Schema meta-schema; tool and
    # response_format schemas are the same on every turn of a conversation,
    # so cache the built validator instead of re-validating and rebuilding it.
    # json.dumps uses ASCII escapes, so character count equals source bytes.
    key = json.dumps(schema, sort_keys=True)
    with _validator_cache_lock:
        cached = _validator_cache.get(key)
        if cached is not None:
            _validator_cache.move_to_end(key)
            return cached
    base = validators.validator_for(schema)
    base.check_schema(schema)
    # Draft 3 accepts any type name, and validation fails on one the dialect
    # does not define with an error that is not a validation error.
    for node in subschemas(schema):
        if not isinstance(node, dict):
            continue
        for keyword in ("type", "disallow"):
            names = node.get(keyword)
            for name in names if isinstance(names, list) else (names,):
                if isinstance(name, str) and not _known_type(base, name):
                    raise APIError(400, f"unknown schema type: {name}")
    # jsonschema matches patternProperties with the unbounded standard-library
    # engine to find the properties unevaluatedProperties applies to. A
    # reference can reach any object, so one document cannot use both.
    keywords = {key for node in json_objects(schema) for key in node}
    if {"patternProperties", "unevaluatedProperties"} <= keywords:
        raise APIError(
            400, "unevaluatedProperties with patternProperties is not supported"
        )
    validated = copy.deepcopy(schema)
    # A document uses one dialect. Removing identical declarations prevents
    # jsonschema.evolve from replacing the bounded class at a local reference.
    for node in subschemas(validated):
        if isinstance(node, dict) and "$schema" in node:
            if validators.validator_for(node) is not base:
                raise APIError(400, "mixed schema dialects are not supported")
            node.pop("$schema")
    validator = _bounded_class(base)(validated, registry=_REGISTRY)
    if len(key) > _VALIDATOR_CACHE_SOURCE_BYTES:
        return validator
    with _validator_cache_lock:
        # Another preparation thread may have filled the same miss.
        cached = _validator_cache.get(key)
        if cached is not None:
            _validator_cache.move_to_end(key)
            return cached
        _validator_cache[key] = validator
        _validator_cache_bytes += len(key)
        while (
            len(_validator_cache) > _VALIDATOR_CACHE_SIZE
            or _validator_cache_bytes > _VALIDATOR_CACHE_SOURCE_BYTES
        ):
            evicted_key, _ = _validator_cache.popitem(last=False)
            _validator_cache_bytes -= len(evicted_key)
    return validator
