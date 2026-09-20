"""Strict JSON and compact UTF-8 encoding for API payloads and retained state."""

import json
import math
import re


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
    return json.loads(
        value, parse_constant=_reject_json_constant, parse_float=_finite_json_float
    )


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
