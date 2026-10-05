"""The options `splash serve` and the server share: each flag's check, help
and default, and how the launcher passes a value on to the server.

Standard library only, with the server modules that are too: the launcher
parses these options before .venv exists.
"""

import argparse
import copy
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import images, origins
from .http_security import validate_api_key

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
MAX_CONTEXT_TOKENS = 262144
DEFAULT_MAX_REQUEST_BYTES = 128 * 1024 * 1024
DEFAULT_QUEUE_SIZE = 32
# The port both commands serve on unless --port names another, or for the
# launcher SPLASH_PORT does.
DEFAULT_PORT = 8000
# Where --persistent-cache keeps its files unless --cache-dir says otherwise.
DEFAULT_CACHE_DIR = Path.home() / "Library/Caches/Splash/prefix-cache"
_SIZE_UNITS = {
    unit + suffix: 1024**power
    for power, unit in enumerate(("K", "M", "G"), 1)
    for suffix in ("", "B", "IB")
}
# A model ID, as --model takes it: OWNER/REPO, a Hugging Face repository ID
# by the Hub's rule, then optionally :VARIANT, which names the GGUF to serve,
# such as UD-Q4_K_M.
REPO_ID = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/"
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
)
VARIANT_SEPARATOR = ":"
VARIANT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def check_repo_id(value):
    """`value`, if it is a full Hugging Face repository ID; ValueError
    otherwise."""
    if (
        not isinstance(value, str)
        or not REPO_ID.fullmatch(value)
        or "--" in value
        or ".." in value
        or value.endswith(".git")
    ):
        raise ValueError("model must be a full Hugging Face repository ID (owner/repo)")
    return value


def split_model_id(value):
    """A model ID's repository ID and variant, None without one; ValueError
    for a value that is no model ID."""
    if not isinstance(value, str):
        raise ValueError("model must be a full Hugging Face repository ID (owner/repo)")
    repo_id, separator, variant = value.partition(VARIANT_SEPARATOR)
    check_repo_id(repo_id)
    if not separator:
        return repo_id, None
    if not VARIANT.fullmatch(variant) or ".." in variant:
        raise ValueError(
            "model variant must be a short name such as UD-Q4_K_M "
            f"(owner/repo{VARIANT_SEPARATOR}VARIANT)"
        )
    return repo_id, variant


def parse_model_id(value):
    try:
        split_model_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return value


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


def parse_cache_dir(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("must name a directory")
    return Path(value).expanduser().absolute()


def parse_request_size(value):
    try:
        size = parse_max_memory(value)
    except argparse.ArgumentTypeError:
        size = None
    if size is None:
        raise argparse.ArgumentTypeError("must be a positive byte count such as 128M")
    return size


# The units a duration takes after its number; seconds without one.
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600}


def _duration_seconds(value):
    """A duration in seconds, written as seconds or with an s, m or h suffix;
    None unless positive and finite."""
    text = value.strip().lower()
    unit = _DURATION_UNITS.get(text[-1:])
    try:
        seconds = float(text[:-1]) * unit if unit else float(text)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def parse_request_timeout(value):
    if (seconds := _duration_seconds(value)) is None:
        raise argparse.ArgumentTypeError(
            "must be a positive duration such as 30m, 2h or 3600"
        )
    return seconds


def parse_idle_release(value):
    """--idle-release in seconds, math.inf for off."""
    if value.strip().lower() == "off":
        return math.inf
    if (seconds := _duration_seconds(value)) is None:
        raise argparse.ArgumentTypeError(
            "must be off or a positive duration such as 30m, 2h or 600"
        )
    return seconds


def idle_release_text(seconds):
    """--idle-release as the server's and the engine's command lines spell it."""
    return "off" if math.isinf(seconds) else str(seconds)


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


# The help's option groups, in the order `splash serve --help` lists them:
# each group's key and title. The launcher's own model options and --port
# join "model" and "network".
GROUPS = (
    ("model", "model"),
    ("network", "network and access"),
    ("memory", "memory and context"),
    ("cache", "SSD cache"),
    ("api", "API"),
    ("requests", "requests"),
)


@dataclass(frozen=True)
class ServeOption:
    """An option of both commands, and how the launcher passes it on."""

    flag: str
    # The key of the help group that lists it (GROUPS).
    group: str
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
        "network",
        dict(
            default="127.0.0.1",
            metavar="ADDRESS",
            help="HTTP bind address (default: 127.0.0.1; 0.0.0.0 for all IPv4 "
            "interfaces); clients use an IP address, localhost or a name given "
            "with --allowed-host",
        ),
    ),
    ServeOption(
        "--allowed-host",
        "network",
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
        "network",
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
        "--api-key",
        "network",
        dict(
            type=parse_api_key,
            metavar="KEY",
            help="require this key as a bearer token or x-api-key on every route "
            "but the chat page, /health and /ready (default: SPLASH_API_KEY)",
        ),
        environment="SPLASH_API_KEY",
        secret=True,
    ),
    ServeOption(
        "--no-webui",
        "network",
        dict(action="store_true", default=False, help="disable the chat page"),
    ),
    ServeOption(
        "--max-memory",
        "memory",
        dict(
            type=parse_max_memory,
            default=None,
            metavar="SIZE",
            help="Metal memory budget, e.g. 28G; it can only lower the automatic "
            "budget (default: auto)",
        ),
    ),
    ServeOption(
        "--max-context",
        "memory",
        dict(
            type=parse_max_context,
            default=None,
            metavar="TOKENS",
            help="context token limit, e.g. 100K (K = 1024 tokens); it can only "
            "lower the automatic limit, the model's native window when memory "
            "holds it (default: auto)",
        ),
    ),
    ServeOption(
        "--kv-format",
        "memory",
        dict(
            choices=("int8", "bf16"),
            default="int8",
            help="target KV cache storage (default: int8); bf16 uses more memory",
        ),
    ),
    ServeOption(
        "--idle-release",
        "memory",
        dict(
            type=parse_idle_release,
            default=None,
            metavar="DURATION",
            help="time without a request before the engine unwires its memory and "
            "frees the weights: seconds, or with an s, m or h suffix, e.g. 30m "
            "(default: 10m); off keeps both",
        ),
        text=idle_release_text,
    ),
    ServeOption(
        "--max-cache-disk",
        "cache",
        dict(
            type=parse_max_cache_disk,
            default=0,
            metavar="SIZE",
            help="SSD quota for cached KV pages and states, e.g. 5G (default: 0, "
            "disabled)",
        ),
    ),
    ServeOption(
        "--persistent-cache",
        "cache",
        dict(
            action="store_true",
            default=False,
            help="keep the SSD cache across restarts (needs --max-cache-disk)",
        ),
    ),
    ServeOption(
        "--cache-dir",
        "cache",
        dict(
            type=parse_cache_dir,
            default=None,
            metavar="DIRECTORY",
            help="where --persistent-cache keeps its files; needs "
            "--persistent-cache (default: ~/Library/Caches/Splash/prefix-cache)",
        ),
    ),
    ServeOption(
        "--served-model-name",
        "api",
        dict(
            action="append",
            default=[],
            type=parse_served_model_name,
            metavar="NAME",
            help="additional API model name (repeatable); responses report the "
            "loaded model ID unless --announce-served-name",
        ),
    ),
    ServeOption(
        "--announce-served-name",
        "api",
        dict(
            action="store_true",
            default=False,
            help="report the first --served-model-name in API responses and list "
            "it first in /v1/models; /status keeps the loaded model ID (needs "
            "--served-model-name)",
        ),
    ),
    ServeOption(
        "--default-reasoning-effort",
        "api",
        dict(
            type=parse_reasoning_effort,
            metavar="EFFORT",
            help="effort of Chat and Responses requests that set none: "
            + ", ".join(REASONING_EFFORTS[:-1])
            + f" or {REASONING_EFFORTS[-1]} (default: "
            "SPLASH_DEFAULT_REASONING_EFFORT, else the model template's)",
        ),
        environment="SPLASH_DEFAULT_REASONING_EFFORT",
    ),
    ServeOption(
        "--max-request-size",
        "requests",
        dict(
            type=parse_request_size,
            default=DEFAULT_MAX_REQUEST_BYTES,
            metavar="SIZE",
            help="maximum HTTP request body size, e.g. 128M (default: 128M); "
            "shared input budget is max(512M, twice this limit)",
        ),
    ),
    ServeOption(
        "--max-image-pixels",
        "requests",
        dict(
            type=parse_max_image_pixels,
            default=images.MAX_PIXELS,
            metavar="PIXELS",
            help=f"maximum resized pixels per image, {images.MIN_PIXELS}–"
            f"{images.MAX_PIXELS} (default: {images.MAX_PIXELS}); bounds the "
            "vision scratch one image needs",
        ),
    ),
    ServeOption(
        "--request-timeout",
        "requests",
        dict(
            type=parse_request_timeout,
            default=None,
            metavar="DURATION",
            help="time before a queued or in-flight request expires with 504: "
            "seconds, or with an s, m or h suffix, e.g. 30m (default: none)",
        ),
    ),
    ServeOption(
        "--queue-size",
        "requests",
        dict(
            type=parse_queue_size,
            default=DEFAULT_QUEUE_SIZE,
            metavar="REQUESTS",
            help="requests admitted at once, running or waiting; more get 503 "
            f"(default: {DEFAULT_QUEUE_SIZE})",
        ),
    ),
    ServeOption(
        "--decode-share",
        "requests",
        dict(
            type=parse_decode_share,
            default=None,
            metavar="SHARE",
            help="decode time owed per unit of prefill time while other requests "
            "decode (default: 0.5; 0 alternates one command each)",
        ),
    ),
    ServeOption(
        "--allow-idle-sleep",
        "requests",
        dict(
            action="store_true",
            default=False,
            help="let the Mac sleep automatically while requests run (default: it "
            "stays awake until they finish; the display may still sleep)",
        ),
    ),
)


def option_groups(parser):
    """The help groups of `parser`, by key, created in GROUPS order. The
    server's parser leaves "model" empty, and help omits an empty group."""
    return {key: parser.add_argument_group(title) for key, title in GROUPS}


def add_serve_arguments(parser, groups=None):
    """Add every shared option to its group of `parser`, from
    option_groups(parser) unless the caller made them. Defaults from the
    environment are read now, and checked like a value given on the command
    line; a list default is copied, so no parse returns the table's own
    list."""
    if groups is None:
        groups = option_groups(parser)
    for option in SERVE_OPTIONS:
        groups[option.group].add_argument(
            option.flag, **{**option.options, "default": copy.copy(option.default())}
        )


def check_serve_arguments(parser, args):
    """The checks of the shared options that involve more than one."""
    if args.announce_served_name and not args.served_model_name:
        parser.error("--announce-served-name needs --served-model-name")
    if args.persistent_cache and not args.max_cache_disk:
        parser.error("--persistent-cache needs --max-cache-disk")
    if args.cache_dir is not None and not args.persistent_cache:
        parser.error("--cache-dir needs --persistent-cache")


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
