"""Strict JSON and compact UTF-8 encoding for API payloads and retained state."""

import json
import math
import re
from itertools import chain

# Arrays and objects nest at most this deep in JSON the server reads: deeper
# than any request needs, and shallow enough for every later stage that
# recurses, such as the pure-Python encoder that sizes retained input.
# Schemas, read with more stack a level, keep their own 64-level limit.
MAX_DEPTH = 128
_CONTAINERS = (dict, list)
_TOO_DEEP = f"JSON nests deeper than {MAX_DEPTH} levels"


class JSONEncodingError(RuntimeError):
    """A server-side value could not be serialized as JSON."""


_ENCODER = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
# Escape lone surrogates and separators recognized by text-based SSE readers.
_ESCAPED = re.compile("[\ud800-\udfff\u0085\u2028\u2029]")


def _escape(text):
    return _ESCAPED.sub(lambda match: f"\\u{ord(match[0]):04x}", text)


def _reject_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number is outside the supported range")
    return number


def loads(value):
    """The value JSON text spells; ValueError for text that is not strict
    JSON or nests deeper than MAX_DEPTH."""
    try:
        document = json.loads(
            value, parse_constant=_reject_json_constant, parse_float=_finite_json_float
        )
    except RecursionError:
        # The parser's own bound, near 10,000 levels.
        raise ValueError(_TOO_DEEP) from None
    # The containers of each level in turn, read without recursion.
    level = [document] if type(document) in _CONTAINERS else []
    depth = 0
    while level:
        depth += 1
        if depth > MAX_DEPTH:
            raise ValueError(_TOO_DEEP)
        children = chain.from_iterable(
            item.values() if type(item) is dict else item for item in level
        )
        level = [child for child in children if type(child) in _CONTAINERS]
    return document


def dumps(value):
    try:
        return _escape(_ENCODER.encode(value))
    except (TypeError, ValueError, RecursionError) as error:
        raise JSONEncodingError("value is not JSON serializable") from error


def encode(value):
    return dumps(value).encode("utf-8")


def encoded_size(value):
    """Count encoded bytes without materializing the complete document."""
    try:
        return sum(
            len(_escape(part).encode("utf-8")) for part in _ENCODER.iterencode(value)
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise JSONEncodingError("value is not JSON serializable") from error
