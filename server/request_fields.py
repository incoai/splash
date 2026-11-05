"""The request fields every API reads alike, each read in one place."""

import hashlib

from . import protocol as wire
from .errors import APIError

# The priority a request names, by its lowercase wire name.
_PRIORITIES = {priority.name.lower(): priority for priority in wire.RequestPriority}


def streaming(body):
    """Whether a generation request streams its answer. Every generation
    API reads its stream here first, before its other fields, so an invalid
    one is refused alike everywhere."""
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise APIError(400, "stream must be a boolean")
    return stream


def progress_flag(body, streamed):
    """The request flag of a generation request's return_progress, which only
    a stream takes; `streamed` is the request's streaming(body)."""
    value = body.get("return_progress", False)
    if not isinstance(value, bool) or (value and not streamed):
        raise APIError(
            400, "return_progress requires stream: true and must be a boolean"
        )
    return wire.RequestFlag.RETURN_PROGRESS if value else wire.RequestFlag(0)


def priority(body):
    """The scheduling priority the request names; normal without one."""
    priority_name = body.get("priority", "normal")
    if not isinstance(priority_name, str) or priority_name not in _PRIORITIES:
        raise APIError(400, "priority must be foreground, normal, or background")
    return _PRIORITIES[priority_name]


def cache_salt(body):
    """The request's cache_salt as the engine keys its cache: 128 bits of
    its hash, zero without one. A request shares cached prompts only with
    requests of the same salt."""
    salt = body.get("cache_salt")
    if salt is None:
        return 0
    if not isinstance(salt, str) or not salt:
        raise APIError(400, "cache_salt must be a non-empty string")
    digest = hashlib.blake2b(
        salt.encode("utf-8", "surrogatepass"), digest_size=16, person=b"splash-salt"
    ).digest()
    return int.from_bytes(digest, "little")
