#!/usr/bin/env python3
"""Serve in the foreground, or connect an installed agent to the local server."""

import argparse
import fcntl
import http.client
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request

try:
    from . import catalog, clients, paths
    from . import models as model_artifacts
except ImportError:  # Executed directly by the source or packaged entry point.
    import catalog
    import clients
    import models as model_artifacts
    import paths

ROOT = paths.ROOT
RUNTIME_DIR = paths.RUNTIME
PORT = 8000
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
BASE_URL = f"http://127.0.0.1:{PORT}"


class LauncherError(RuntimeError):
    pass


def _base_url(port):
    return f"http://127.0.0.1:{port}"


def _runtime_dir(port):
    return RUNTIME_DIR if port == PORT else RUNTIME_DIR / "ports" / str(port)


def _request_json(path, timeout=2, *, port=PORT):
    request = urllib.request.Request(_base_url(port) + path)
    if key := os.environ.get("SPLASH_API_KEY"):
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise LauncherError(
                "Splash authentication failed; set SPLASH_API_KEY to the server's key"
            ) from None
        return None
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        urllib.error.URLError,
        http.client.HTTPException,
    ):
        return None


def _running_status(port=PORT):
    status = _request_json("/status", timeout=10, port=port)
    if not isinstance(status, dict):
        return None
    return status


def _ensure_installed(model_id):
    if not paths.PACKAGED:
        # Serialize builds across ports; make keeps the lock if the launcher exits.
        with (RUNTIME_DIR / "build.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for command in (
                ["make", "platform-check", "install-environment"],
                ["make", "-j4", "all"],
            ):
                if subprocess.run(
                    command, cwd=ROOT, pass_fds=(lock.fileno(),)
                ).returncode:
                    raise LauncherError("source build failed; see the output above")
    command = [
        str(paths.PYTHON),
        str(ROOT / "install/models.py"),
        "--models",
        str(paths.MODELS),
        "--model",
        model_id,
        "prepare",
    ]
    if subprocess.run(command, cwd=ROOT).returncode:
        raise LauncherError("model download or verification failed")


def _serve_lock_owner(lock):
    try:
        lock.seek(0)
        owner = json.load(lock)
    except (OSError, UnicodeError, ValueError):
        return ""
    if not isinstance(owner, dict):
        return ""
    pid, model, port = owner.get("pid"), owner.get("model"), owner.get("port")
    if (
        type(pid) is not int
        or pid <= 0
        or not isinstance(model, str)
        or not model
        or not model.isprintable()
        or type(port) is not int
        or not 1 <= port <= 65535
    ):
        return ""
    return f" (PID {pid}, model {model}, port {port})"


def serve(args):
    # Keep both locks across exec until the foreground server exits.
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with (
        (RUNTIME_DIR / "serve.lock").open("a+") as installation,
        (RUNTIME_DIR / f"serve-{args.port}.lock").open("a+") as lock,
    ):
        # Servers share the installation; upgrades require exclusive access.
        try:
            fcntl.flock(installation, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LauncherError(
                "Splash installation is busy; "
                "stop the running server or wait for the upgrade to finish"
            ) from None
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LauncherError(
                f"Splash is already serving{_serve_lock_owner(lock)}; "
                "stop it with Ctrl+C first"
            ) from None
        lock.seek(0)
        lock.truncate()
        json.dump({"pid": os.getpid(), "model": args.model, "port": args.port}, lock)
        lock.flush()
        # Fail before downloads/builds if another service owns the selected port.
        # The HTTP server also binds before loading weights, closing the race.
        with socket.socket() as probe:
            # Match the HTTP listener: closed connections in TIME_WAIT must
            # not block a restart; a live listener still owns the address.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((args.host, args.port))
            except OSError as error:
                raise LauncherError(
                    f"cannot bind {args.host}:{args.port}: {error}"
                ) from None
        _ensure_installed(args.model)
        root = model_artifacts.installed_root(paths.MODELS, args.model)
        command = [
            str(paths.PYTHON),
            "-u",
            str(ROOT / "server/server.py"),
            str(root / "target"),
            str(root / "draft"),
            "--tokenizer",
            str(root / "tokenizer"),
            "--model",
            args.model,
            "--binary",
            str(paths.BINARY),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--max-memory",
            "auto" if args.max_memory is None else str(args.max_memory),
            "--max-context",
            "auto" if args.max_context is None else str(args.max_context),
        ]
        if args.kv_format != "int8":
            command.extend(("--kv-format", args.kv_format))
        for name in args.served_model_name:
            command.append(f"--served-model-name={name}")
        if args.default_reasoning_effort is not None:
            command.extend(
                ["--default-reasoning-effort", args.default_reasoning_effort]
            )
        if args.max_request_size is not None:
            command.extend(["--max-request-size", str(args.max_request_size)])
        if args.max_image_pixels is not None:
            command.extend(["--max-image-pixels", str(args.max_image_pixels)])
        if args.no_webui:
            command.append("--no-webui")
        for host in args.allowed_host:
            command.extend(["--allowed-host", host])
        environment = dict(
            os.environ, PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error"
        )
        if args.api_key is not None:
            environment["SPLASH_API_KEY"] = args.api_key
        # Detached, because execve replaces this process a line later and a
        # thread would not survive it. Failure is silent by design.
        catalog.spawn_refresh()
        os.set_inheritable(installation.fileno(), True)
        os.set_inheritable(lock.fileno(), True)
        os.execve(command[0], command, environment)


def coding_client(args):
    path = clients.find_executable(args.command)
    snapshot = _running_status(args.port)
    if snapshot is None:
        raise LauncherError(
            f"No ready Splash server at {_base_url(args.port)}. "
            "Run 'splash serve --model <HF_REPO_ID>' "
            "in another terminal first."
        )
    catalog = _request_json("/v1/models", port=args.port)
    models = catalog.get("data", []) if isinstance(catalog, dict) else []
    if (
        not isinstance(models, list)
        or not models
        or not isinstance(models[0], dict)
        or models[0].get("owned_by") != "splash"
    ):
        raise LauncherError("Could not identify the local Splash server")
    model, context = models[0].get("id"), snapshot.get("maximum_context_tokens")
    if type(context) is not int or context <= 0:
        raise LauncherError(
            "Splash is running but its context limit is not available yet; wait and retry"
        )
    # Only opencode needs its major version: the launch defaults changed
    # between its first and second major releases. A failed probe adds nothing.
    client_version = (
        clients.probe_major_version(path) if args.command == "opencode" else None
    )
    command, environment = clients.command(
        args.command,
        path,
        _base_url(args.port),
        model,
        context,
        _runtime_dir(args.port),
        client_args=args.client_args,
        client_version=client_version,
    )
    print(f"Starting {args.command}: {model} · {context:,} context tokens", flush=True)
    if args.command == "claude":
        print(
            "Claude hosted WebSearch is unavailable. "
            "WebFetch, local tools and MCP are unchanged.",
            flush=True,
        )
    elif args.command == "codex":
        print(
            "Codex hosted WebSearch is disabled: Splash does not provide "
            "OpenAI's search service. Local tools and MCP are unchanged.",
            flush=True,
        )
    os.execvpe(path, command, environment)


def _parse_port(value):
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "port must be an integer from 1 to 65535"
        ) from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _parse_max_memory(value):
    normalized = value.strip().upper()
    if normalized == "AUTO":
        return None
    suffixes = {
        unit + suffix: 1024**power
        for power, unit in enumerate(("K", "M", "G"), 1)
        for suffix in ("", "B", "IB")
    }
    multiplier = 1
    for suffix in sorted(suffixes, key=len, reverse=True):
        if normalized.endswith(suffix):
            normalized, multiplier = normalized[: -len(suffix)], suffixes[suffix]
            break
    try:
        result = int(normalized) * multiplier
    except ValueError:
        raise argparse.ArgumentTypeError("use a value such as 32G") from None
    if not 1 <= result <= 2**63 - 1:
        raise argparse.ArgumentTypeError("use a positive value such as 32G")
    return result


def _parse_request_size(value):
    size = _parse_max_memory(value)
    if size is None:
        raise argparse.ArgumentTypeError("use a positive byte count such as 128M")
    return size


def _parse_max_context(value):
    normalized = value.strip().upper()
    if normalized == "AUTO":
        return None
    try:
        result = (
            int(normalized[:-1]) * 1024 if normalized.endswith("K") else int(normalized)
        )
    except ValueError:
        raise argparse.ArgumentTypeError("use a value such as 100K") from None
    if not 1 <= result <= 262144:
        raise argparse.ArgumentTypeError("must be between 1 and 256K tokens")
    return result


def _version():
    if not paths.PACKAGED:
        return "Splash (source checkout)"
    return "Splash " + str(
        json.loads((paths.ROOT / "release.json").read_text())["version"]
    )


def _parse_served_model_name(value):
    if (
        not value
        or any(not c.isprintable() or c.isspace() or c in "\\%?#" for c in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise argparse.ArgumentTypeError(
            "model alias must be a non-empty name without whitespace or URL delimiters"
        )
    return value


def _parse_max_image_pixels(value):
    try:
        pixels = int(value)
    except ValueError:
        pixels = 0
    # Match the server's supported image budget without importing its runtime
    # dependencies before help, argument validation or first-time installation.
    if not 65_536 <= pixels <= 4_194_304:
        raise argparse.ArgumentTypeError("must be between 65536 and 4194304 pixels")
    return pixels


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    client_args = []
    if argv and argv[0] in clients.INSTALL_URLS:
        argv, client_args = argv[:1], argv[1:]
        if client_args[:1] == ["--"]:
            client_args = client_args[1:]
    elif "--" in argv:
        boundary = argv.index("--")
        argv, client_args = argv[:boundary], argv[boundary + 1 :]
    parser = argparse.ArgumentParser(
        prog="splash",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  splash serve --model incoai/Qwen3.8-27B-Splash\n"
            "  splash opencode  # in another terminal, after Ready\n\n"
            "Use splash serve --help for server settings. Client arguments,\n"
            "including --help, are passed through to the installed agent."
        ),
    )
    parser.add_argument("--version", action="version", version=_version())
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser(
        "serve",
        help="run the local server; Ctrl+C stops it",
        description="Download a Splash model package if needed, then serve in the foreground.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  splash serve --model incoai/Qwen3.8-27B-Splash\n"
            "  splash serve --model incoai/Qwen3.6-35B-A3B-Splash --max-context 128K\n\n"
            "After Ready, open http://127.0.0.1:8000 or connect an installed agent.\n"
            "The startup summary and /status report the effective context limit.\n"
            "A client may impose a smaller limit. Keep this terminal open; Ctrl+C stops serving."
        ),
    )
    server.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind address (default: 127.0.0.1; 0.0.0.0 for all IPv4 interfaces)",
    )
    server.add_argument(
        "--port",
        type=_parse_port,
        default=os.environ.get("SPLASH_PORT", str(PORT)),
        help="HTTP port (default: SPLASH_PORT or 8000)",
    )
    server.add_argument(
        "--model",
        type=model_artifacts.parse_repo_id,
        required=True,
        metavar="OWNER/REPO",
        help="Hugging Face repository containing a Splash package",
    )
    server.add_argument(
        "--served-model-name",
        action="append",
        default=[],
        type=_parse_served_model_name,
        help="additional API model name; responses keep the loaded model ID (repeatable)",
    )
    server.add_argument(
        "--default-reasoning-effort",
        choices=REASONING_EFFORTS,
        default=os.environ.get("SPLASH_DEFAULT_REASONING_EFFORT"),
        help="Chat/Responses effort when unspecified (default: SPLASH_DEFAULT_REASONING_EFFORT or model template)",
    )
    server.add_argument(
        "--kv-format",
        choices=("int8", "bf16"),
        default="int8",
        help="target KV cache storage (default: int8); bf16 uses more memory",
    )
    server.add_argument(
        "--max-memory",
        type=_parse_max_memory,
        help="Metal budget ceiling, e.g. 28G (default: auto)",
    )
    server.add_argument(
        "--max-context",
        type=_parse_max_context,
        help="context token limit, up to 256K (K = 1024; default: auto within the memory budget)",
    )
    server.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST",
        help="additional HTTP Host name to accept; does not change the bind address "
        "(repeatable)",
    )
    server.add_argument(
        "--max-request-size",
        type=_parse_request_size,
        help="maximum HTTP request body size, e.g. 128M (default: 128M); "
        "shared input budget is max(512M, twice this limit)",
    )
    server.add_argument(
        "--max-image-pixels",
        type=_parse_max_image_pixels,
        help="maximum resized pixels per image, 65536–4194304 (default: 4194304)",
    )
    server.add_argument(
        "--api-key",
        default=os.environ.get("SPLASH_API_KEY"),
        help="API key (default: SPLASH_API_KEY environment variable)",
    )
    server.add_argument("--no-webui", action="store_true", help="disable the chat page")
    for name in clients.INSTALL_URLS:
        commands.add_parser(name, help=f"connect {name} to the running server")
    args = parser.parse_args(argv)
    if (
        args.command == "serve"
        and args.default_reasoning_effort is not None
        and args.default_reasoning_effort not in REASONING_EFFORTS
    ):
        parser.error(
            "invalid --default-reasoning-effort / SPLASH_DEFAULT_REASONING_EFFORT"
        )
    if args.command in clients.INSTALL_URLS:
        try:
            args.port = _parse_port(os.environ.get("SPLASH_PORT", str(PORT)))
        except argparse.ArgumentTypeError as error:
            parser.error(f"SPLASH_PORT: {error}")
    if args.command == "serve" and args.api_key is not None:
        if not args.api_key or any(ord(c) <= 32 or ord(c) >= 127 for c in args.api_key):
            parser.error("API key must contain only visible ASCII characters")
    if client_args and args.command == "serve":
        parser.error("arguments after -- are only supported for coding clients")
    args.client_args = client_args
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return serve(args) if args.command == "serve" else coding_client(args)
    except (LauncherError, clients.ClientError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
