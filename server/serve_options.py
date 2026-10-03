"""The options `splash serve` and the server share: each flag's check, help
and default, and how the launcher passes a value on to the server.

Standard library only, with the server modules that are too: the launcher
parses these options before .venv exists.
"""

import argparse
import copy
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

from . import images, origins
from .http_security import validate_api_key

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
MAX_CONTEXT_TOKENS = 262144
DEFAULT_MAX_REQUEST_BYTES = 128 * 1024 * 1024
DEFAULT_QUEUE_SIZE = 32
_SIZE_UNITS = {
    unit + suffix: 1024**power
    for power, unit in enumerate(("K", "M", "G"), 1)
    for suffix in ("", "B", "IB")
}


def parse_max_context(value):
    """'auto' (None), or a token count; K is 1024 tokens."""
    normalized = value.strip().upper()
    if normalized == "AUTO":
        return None
    try:
        tokens = (
            int(normalized[:-1]) * 1024 if normalized.endswith("K") else int(normalized)
        )
    except ValueError:
        tokens = 0
    if not 1 <= tokens <= MAX_CONTEXT_TOKENS:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a token count up to 256K, such as 100K"
        )
    return tokens


def parse_max_memory(value):
    """'auto' (None), or a byte count with an optional K, M or G suffix."""
    normalized = value.strip().upper()
    if normalized == "AUTO":
        return None
    multiplier = 1
    for suffix in sorted(_SIZE_UNITS, key=len, reverse=True):
        if normalized.endswith(suffix):
            normalized, multiplier = normalized[: -len(suffix)], _SIZE_UNITS[suffix]
            break
    try:
        size = int(normalized) * multiplier
    except ValueError:
        size = 0
    if not 1 <= size <= 2**63 - 1:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a positive byte count such as 32G"
        )
    return size


def parse_max_cache_disk(value):
    if value.strip() == "0":
        return 0
    try:
        size = parse_max_memory(value)
    except argparse.ArgumentTypeError:
        size = None
    if size is None:
        raise argparse.ArgumentTypeError("use 0 to disable, or a size such as 5G")
    return size


def parse_request_size(value):
    try:
        size = parse_max_memory(value)
    except argparse.ArgumentTypeError:
        size = None
    if size is None:
        raise argparse.ArgumentTypeError("must be a positive byte count such as 128M")
    return size


def parse_request_timeout(value):
    try:
        seconds = float(value)
    except ValueError:
        seconds = math.nan
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive number of seconds such as 3600"
        )
    return seconds


def parse_queue_size(value):
    try:
        size = int(value)
    except ValueError:
        size = 0
    if size <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive number of requests such as 32"
        )
    return size


def parse_decode_share(value):
    try:
        share = float(value)
    except ValueError:
        share = math.nan
    if not math.isfinite(share) or share < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative number such as 0.5")
    return share


def parse_max_image_pixels(value):
    try:
        pixels = int(value)
    except ValueError:
        pixels = 0
    if not images.MIN_PIXELS <= pixels <= images.MAX_PIXELS:
        raise argparse.ArgumentTypeError(
            f"must be between {images.MIN_PIXELS} and {images.MAX_PIXELS} pixels"
        )
    return pixels


def parse_served_model_name(value):
    if (
        not isinstance(value, str)
        or not value
        or any(not c.isprintable() or c.isspace() or c in "\\%?#" for c in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise argparse.ArgumentTypeError(
            "model alias must be a non-empty name without whitespace or URL delimiters"
        )
    return value


def parse_allowed_origin(value):
    """An --allowed-origin value as the server compares Origin headers with it."""
    try:
        return origins.parse_allowed_origin(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _origin_text(origin):
    """A parsed --allowed-origin value as parse_allowed_origin reads it back."""
    if origin == origins.ANY_ORIGIN:
        return origin
    scheme, host, port = origin
    authority = f"[{host}]" if ":" in host else host
    return f"{scheme}://{authority}" + ("" if port is None else f":{port}")


def parse_api_key(value):
    try:
        return validate_api_key(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def parse_reasoning_effort(value):
    if value not in REASONING_EFFORTS:
        raise argparse.ArgumentTypeError(
            f"must be one of {', '.join(REASONING_EFFORTS)}"
        )
    return value


@dataclass(frozen=True)
class ServeOption:
    """An option of both commands, and how the launcher passes it on."""

    flag: str
    # add_argument's keywords, help included.
    options: dict
    # The environment variable that gives the default.
    environment: str | None = None
    # A secret reaches the server in its environment variable, never on its
    # command line, where other processes can read it.
    secret: bool = False
    # A parsed value as the server's command line spells it.
    text: Callable[[object], str] = str

    @property
    def dest(self):
        return self.flag.removeprefix("--").replace("-", "_")

    def default(self):
        if self.environment is not None:
            return os.environ.get(self.environment)
        return self.options.get("default")


SERVE_OPTIONS = (
    ServeOption(
        "--host",
        dict(
            default="127.0.0.1",
            help="HTTP bind address (default: 127.0.0.1; 0.0.0.0 for all IPv4 "
            "interfaces); clients use an IP address, localhost or a name given "
            "with --allowed-host",
        ),
    ),
    ServeOption(
        "--served-model-name",
        dict(
            action="append",
            default=[],
            type=parse_served_model_name,
            help="additional API model name (repeatable); responses report the "
            "loaded model ID unless --announce-served-name",
        ),
    ),
    ServeOption(
        "--announce-served-name",
        dict(
            action="store_true",
            default=False,
            help="report the first --served-model-name in API responses and list "
            "it first in /v1/models; /status keeps the loaded model ID",
        ),
    ),
    ServeOption(
        "--default-reasoning-effort",
        dict(
            type=parse_reasoning_effort,
            metavar="{" + ",".join(REASONING_EFFORTS) + "}",
            help="Chat/Responses effort when unspecified (default: "
            "SPLASH_DEFAULT_REASONING_EFFORT or model template)",
        ),
        environment="SPLASH_DEFAULT_REASONING_EFFORT",
    ),
    ServeOption(
        "--kv-format",
        dict(
            choices=("int8", "bf16"),
            default="int8",
            help="target KV cache storage (default: int8); bf16 uses more memory",
        ),
    ),
    ServeOption(
        "--max-memory",
        dict(
            type=parse_max_memory,
            default=None,
            help="Metal budget ceiling, e.g. 28G (default: auto)",
        ),
    ),
    ServeOption(
        "--max-cache-disk",
        dict(
            type=parse_max_cache_disk,
            default=0,
            help="SSD quota for cached KV pages and states, e.g. 5G (default: 0, "
            "disabled)",
        ),
    ),
    ServeOption(
        "--max-context",
        dict(
            type=parse_max_context,
            default=None,
            help="context token limit, up to 256K (K = 1024; default: auto within "
            "the memory budget)",
        ),
    ),
    ServeOption(
        "--decode-share",
        dict(
            type=parse_decode_share,
            default=None,
            help="decode time owed per unit of prefill time while other requests "
            "decode (default: 0.5; 0 alternates one command each)",
        ),
    ),
    ServeOption(
        "--allowed-host",
        dict(
            action="append",
            default=[],
            metavar="HOST",
            help="additional HTTP Host name to accept, e.g. mymac.local; does not "
            "change the bind address (repeatable)",
        ),
    ),
    ServeOption(
        "--allowed-origin",
        dict(
            action="append",
            default=[],
            type=parse_allowed_origin,
            metavar="ORIGIN",
            help="origin whose pages may call the API from a browser or webview, "
            "e.g. tauri://localhost; '*' for any (repeatable)",
        ),
        text=_origin_text,
    ),
    ServeOption(
        "--max-request-size",
        dict(
            type=parse_request_size,
            default=DEFAULT_MAX_REQUEST_BYTES,
            help="maximum HTTP request body size, e.g. 128M (default: 128M); "
            "shared input budget is max(512M, twice this limit)",
        ),
    ),
    ServeOption(
        "--max-image-pixels",
        dict(
            type=parse_max_image_pixels,
            default=images.MAX_PIXELS,
            help=f"maximum resized pixels per image, {images.MIN_PIXELS}–"
            f"{images.MAX_PIXELS} (default: {images.MAX_PIXELS}); bounds the "
            "vision scratch one image needs",
        ),
    ),
    ServeOption(
        "--request-timeout",
        dict(
            type=parse_request_timeout,
            default=None,
            help="seconds before a queued or in-flight request expires with 504 "
            "(default: none)",
        ),
    ),
    ServeOption(
        "--queue-size",
        dict(
            type=parse_queue_size,
            default=DEFAULT_QUEUE_SIZE,
            help="requests admitted at once, running or waiting; more get 503 "
            f"(default: {DEFAULT_QUEUE_SIZE})",
        ),
    ),
    ServeOption(
        "--api-key",
        dict(
            type=parse_api_key,
            help="API key (default: SPLASH_API_KEY environment variable)",
        ),
        environment="SPLASH_API_KEY",
        secret=True,
    ),
    ServeOption(
        "--no-webui",
        dict(action="store_true", default=False, help="disable the chat page"),
    ),
)


def add_serve_arguments(parser):
    """Add every shared option to `parser`. Defaults from the environment are
    read now, and checked like a value given on the command line; a list
    default is copied, so no parse returns the table's own list."""
    for option in SERVE_OPTIONS:
        parser.add_argument(
            option.flag, **{**option.options, "default": copy.copy(option.default())}
        )


def check_serve_arguments(parser, args):
    """The checks of the shared options that involve more than one."""
    if args.announce_served_name and not args.served_model_name:
        parser.error("--announce-served-name needs --served-model-name")


def serve_argv(args):
    """The server's command line for the shared options `args` sets to other
    than their defaults; the server reads the defaults itself."""
    argv = []
    for option in SERVE_OPTIONS:
        value = getattr(args, option.dest)
        if option.secret or value == option.default():
            continue
        if option.options.get("action") == "store_true":
            argv.append(option.flag)
        elif option.options.get("action") == "append":
            argv.extend(f"{option.flag}={option.text(item)}" for item in value)
        else:
            argv.append(f"{option.flag}={option.text(value)}")
    return argv


def serve_environment(args):
    """The environment the server reads the shared secrets from."""
    return {
        option.environment: getattr(args, option.dest)
        for option in SERVE_OPTIONS
        if option.secret and getattr(args, option.dest) is not None
    }
