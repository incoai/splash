"""The request fields every API reads alike, each read in one place.

Standard library only, with the server modules that are too: the launcher
checks a --sampling-default by the rule a request's value meets
(serve_options) before .venv exists.
"""

import hashlib

from . import protocol as wire
from .errors import APIError
from .metrics import is_finite_number

# The priority a request names, by its lowercase wire name.
_PRIORITIES = {priority.name.lower(): priority for priority in wire.RequestPriority}

# The fields every generation API takes under Chat's names, as Chat takes
# them: how a request samples (the frame's sampling fields, logit_bias, which
# Splash does not implement and refuses unless empty, ignore_eos and seed),
# and how it is scheduled and shares cache. A dialect hands Chat each one a
# request sends, whether its own API defines it or not, so that Chat's
# validation refuses a bad value with the same 400 on every route and no
# route drops one: Anthropic defines 3 of the 7 sampling fields, and Messages
# dropped the other 4, penalties some clients can set nowhere else.
CHAT_GENERATION_FIELDS = (
    *wire.SAMPLING_FIELDS,
    "logit_bias",
    "ignore_eos",
    "seed",
    "priority",
    "cache_salt",
)


MIN_FLOAT32_SUBNORMAL = float.fromhex("0x1p-149")
FLOAT32_MAX = float.fromhex("0x1.fffffep127")
# The sampling numbers a request may set: each with its default, the value
# that leaves the logits unchanged, the values the engine accepts and how a
# 400 names them. The served family's published defaults
# (Frontend.sampling_defaults) take the place of these. A nonzero top_p or
# repetition_penalty must not round to zero in the engine's float32, which
# would make the value invalid.
SAMPLING_NUMBERS = {
    "temperature": (
        1.0,
        lambda value: 0 <= value <= 2,
        "a number in [0, 2]",
    ),
    "top_p": (
        1.0,
        lambda value: MIN_FLOAT32_SUBNORMAL <= value <= 1,
        "a number in (0, 1]",
    ),
    "presence_penalty": (0.0, lambda value: -2 <= value <= 2, "a number in [-2, 2]"),
    "frequency_penalty": (0.0, lambda value: -2 <= value <= 2, "a number in [-2, 2]"),
    "repetition_penalty": (
        1.0,
        lambda value: MIN_FLOAT32_SUBNORMAL <= value <= FLOAT32_MAX,
        "a positive number",
    ),
    "min_p": (0.0, lambda value: 0 <= value <= 1, "a number in [0, 1]"),
}
# With top_k, whose default 0 keeps every token, the table covers every
# sampling option the frame carries.
assert set(SAMPLING_NUMBERS) | {"top_k"} == set(wire.SAMPLING_FIELDS)


def sampling_refusal(name, value):
    """Why a request may not set the sampling option name to value, as its
    400 says it; None if it may. A top_k of 0 or -1 keeps every token. A
    --sampling-default and the served family's sampling are held to the
    same rule."""
    if name == "top_k":
        if isinstance(value, int) and not isinstance(value, bool) and value >= -1:
            return None
        return "top_k must be 0 or -1 (disabled) or a positive integer"
    if name not in SAMPLING_NUMBERS:
        return f"{name} is not a sampling option"
    _, accepts, requirement = SAMPLING_NUMBERS[name]
    if is_finite_number(value) and accepts(value):
        return None
    return f"{name} must be {requirement}"


def generation(body):
    """Whether a generation request streams its answer, and the request flags
    its answer sets: RETURN_PROGRESS for return_progress, which only a stream
    takes. Every generation API reads these here first, stream before its
    other fields, so an invalid one is refused alike everywhere. The job
    adds the flags its generation options set (Frontend._generation_job)."""
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise APIError(400, "stream must be a boolean")
    progress = body.get("return_progress", False)
    if not isinstance(progress, bool) or (progress and not stream):
        raise APIError(
            400, "return_progress requires stream: true and must be a boolean"
        )
    flags = wire.RequestFlag.RETURN_PROGRESS if progress else wire.RequestFlag(0)
    return stream, flags


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
