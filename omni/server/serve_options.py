"""The options `splash serve` and the server share: each flag's check, help
and default, and how the launcher passes a value on to the server.

Standard library only, with the server modules that are too: the launcher
parses these options before .venv exists.
"""

import argparse
import copy
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

from . import origins
from .http_security import validate_api_key

DEFAULT_MAX_REQUEST_BYTES = 128 * 1024 * 1024
DEFAULT_QUEUE_SIZE = 32
# The port both commands serve on unless --port names another, or for the
# launcher SPLASH_PORT does.
DEFAULT_PORT = 8000
_SIZE_UNITS = {
    unit + suffix: 1024**power
    for power, unit in enumerate(("K", "M", "G"), 1)
    for suffix in ("", "B", "IB")
}
# The draft a Qwen3-ASR-1.7B checkpoint decodes with unless --no-draft-model
# turns speculation off or --draft-model names another
# (docs/omni-performance.md, "DFlash draft"): a DFlash draft that reads the
# served model's hidden states and proposes up to 7 tokens in one forward,
# which the served model verifies, so the transcripts stay the served
# model's. About 0.7 GB on disk and 1.2 GiB of memory. Any Qwen3-ASR-0.6B
# checkpoint still serves as a draft through --draft-model.
DEFAULT_DRAFT_MODEL = "wilsonzheng0327/Qwen3-ASR-1.7B-dflash"
# Qwen3-ASR-0.6B's own DFlash draft: the same five layers at the 0.6B's
# width, reading its layers 2, 14 and 25; about 0.25 GB.
SMALL_DRAFT_MODEL = "wilsonzheng0327/Qwen3-ASR-0.6B-dflash"
# The default draft by the transcription decoder's width.
DEFAULT_DRAFTS = {2048: DEFAULT_DRAFT_MODEL, 1024: SMALL_DRAFT_MODEL}
# A model ID, as --model takes it: OWNER/REPO, a Hugging Face repository ID
# by the Hub's rule.
REPO_ID = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/"
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
)


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


def parse_model_id(value):
    try:
        return check_repo_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


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


def parse_segment_seconds(value: str) -> float:
    """A segment length in seconds: finite and not negative (0 disables)."""
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid segment length: {value}") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError(f"invalid segment length: {value}")
    return seconds


def parse_draft_model(value):
    """--draft-model: a model directory (made absolute, so the launcher and
    the server name the same one), or a Hugging Face repository ID the server
    finds in the Hub cache or downloads."""
    path = os.path.expanduser(value)
    if value and os.path.isdir(path):
        return os.path.abspath(path)
    try:
        return check_repo_id(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be a model directory or a Hugging Face repository ID (owner/repo)"
        ) from None


def hub_snapshot(repo_id):
    """The Hub cache's snapshot of repo_id that refs/main names, else its
    newest; None when the cache holds none. The cache is HF_HUB_CACHE, or
    HF_HOME's hub, or ~/.cache/huggingface/hub."""
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
        "hub",
    )
    folder = os.path.join(cache, "models--" + repo_id.replace("/", "--"))
    try:
        with open(os.path.join(folder, "refs", "main")) as ref:
            commit = ref.read().strip()
        if os.path.isfile(os.path.join(folder, "snapshots", commit, "config.json")):
            return os.path.join(folder, "snapshots", commit)
    except OSError:
        pass
    snapshots = os.path.join(folder, "snapshots")
    try:
        candidates = [
            os.path.join(snapshots, name)
            for name in os.listdir(snapshots)
            if os.path.isfile(os.path.join(snapshots, name, "config.json"))
        ]
    except OSError:
        return None
    return max(candidates, key=os.path.getmtime, default=None)


def default_draft_model(model_root):
    """The DFlash draft of the Qwen3-ASR model model_root's config.json
    describes, by its decoder's width (DEFAULT_DRAFTS: 1.7B or 0.6B, 4-bit
    or bf16), else None: speech models have no draft."""
    try:
        with open(os.path.join(model_root, "config.json"), encoding="utf-8") as file:
            config = json.load(file)
        text = config["thinker_config"]["text_config"]
        # The engine reads a decoder without hidden_size as the 1.7B's.
        width = text.get("hidden_size", 2048)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    if config.get("model_type") != "qwen3_asr" or isinstance(width, bool):
        return None
    return DEFAULT_DRAFTS.get(width)


def download_model(repo_id):
    """repo_id's snapshot in the Hub cache, downloaded by huggingface_hub
    (which honors HF_HUB_OFFLINE, HF_TOKEN and the cache variables). It
    raises what the download raises: offline, unreachable or refused."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id)


def parse_draft_tokens(value):
    try:
        tokens = int(value)
    except ValueError:
        tokens = 0
    if not 1 <= tokens <= 15:
        raise argparse.ArgumentTypeError("must be a number of tokens from 1 to 15")
    return tokens


def parse_speculative_lanes(value):
    try:
        lanes = int(value)
    except ValueError:
        lanes = 0
    if not 1 <= lanes <= 8:
        raise argparse.ArgumentTypeError("must be a number of lanes from 1 to 8")
    return lanes


def parse_lanes(value):
    try:
        lanes = int(value)
    except ValueError:
        lanes = 0
    if not 1 <= lanes <= 64:
        raise argparse.ArgumentTypeError("must be a number of lanes from 1 to 64")
    return lanes


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


# The help's option groups, in the order `splash serve --help` lists them:
# each group's key and title. The launcher's own model options and --port
# join "model" and "network".
GROUPS = (
    ("model", "model"),
    ("network", "network and access"),
    ("memory", "memory"),
    ("api", "API"),
    ("requests", "requests"),
    ("speculation", "speculative transcription"),
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
            "but /health and /ready (default: SPLASH_API_KEY)",
        ),
        environment="SPLASH_API_KEY",
        secret=True,
    ),
    ServeOption(
        "--max-memory",
        "memory",
        dict(
            type=parse_max_memory,
            default=None,
            metavar="SIZE",
            help="Metal memory budget, e.g. 28G; it can only lower the automatic "
            "budget, and sizes the KV page pool after the weights (default: auto)",
        ),
    ),
    ServeOption(
        "--idle-release",
        "memory",
        dict(
            type=parse_idle_release,
            default=None,
            metavar="DURATION",
            help="time without a request before the engine releases what it can: "
            "seconds, or with an s, m or h suffix, e.g. 30m (default: the "
            "engine's); off keeps everything loaded",
        ),
        text=idle_release_text,
    ),
    ServeOption(
        "--disable-ane",
        "memory",
        dict(
            action="store_true",
            default=False,
            help="the GPU alone; by default a 4-bit transcription model's "
            "prefill chunks of 160 tokens or more split their FFN with the "
            "Neural Engine, and speech decodes its codec's convolutions there",
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
        "--lanes",
        "requests",
        dict(
            type=parse_lanes,
            default=None,
            metavar="LANES",
            help="decode lanes, the requests the engine runs at once in lock step "
            "(default: the engine's, 8; a speech model takes at most 16, a "
            "transcription model 32)",
        ),
    ),
    ServeOption(
        "--segment-seconds",
        "requests",
        dict(
            type=parse_segment_seconds,
            default=30.0,
            metavar="SECONDS",
            help="transcription: audio longer than this is cut at quiet points "
            "into about one segment per lane, none shorter, decoded concurrently "
            "(default 30; 0 decodes up to 2 minutes as one request, cutting "
            "longer audio there)",
        ),
    ),
    ServeOption(
        "--draft-model",
        "speculation",
        dict(
            type=parse_draft_model,
            default=None,
            metavar="PATH_OR_ID",
            help="transcription: decode greedily with this draft model proposing "
            "tokens the served model verifies: a directory, or a repository ID, "
            "downloaded into the Hugging Face cache if missing; the transcripts "
            f"stay the served model's (default: {DEFAULT_DRAFT_MODEL} beside "
            f"Qwen3-ASR-1.7B, {SMALL_DRAFT_MODEL} beside Qwen3-ASR-0.6B, none "
            "for speech models)",
        ),
    ),
    ServeOption(
        "--no-draft-model",
        "speculation",
        dict(
            action="store_true",
            default=False,
            help="transcription: decode plainly, without the default draft "
            "(saves its download and about 1.3 GiB of memory)",
        ),
    ),
    ServeOption(
        "--draft-tokens",
        "speculation",
        dict(
            type=parse_draft_tokens,
            default=None,
            metavar="TOKENS",
            help="tokens the draft proposes per lane and step, fewer when many "
            "lanes speculate (default: the engine's, 5 for a 4-bit model, 7 for "
            "bf16)",
        ),
    ),
    ServeOption(
        "--speculative-lanes",
        "speculation",
        dict(
            type=parse_speculative_lanes,
            default=None,
            metavar="LANES",
            help="speculate while at most this many requests decode, plainly "
            "above (default: the engine's, 1 for a 4-bit model, 8 for bf16, "
            "where rounds of that many requests pay for their drafts)",
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
    if args.no_draft_model:
        for flag in ("--draft-model", "--draft-tokens", "--speculative-lanes"):
            if getattr(args, flag[2:].replace("-", "_")) is not None:
                parser.error(f"{flag} needs a draft; --no-draft-model turns it off")


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
