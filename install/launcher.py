#!/usr/bin/env python3
"""Serve in the foreground, or connect an installed agent to the local server."""

import argparse
import errno
import fcntl
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script by the PATH wrappers and ./splash: import siblings as
    # the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

# The options serve shares with the server, in modules of the standard
# library alone: the launcher runs before .venv exists.
from server import serve_options

from . import assembly, catalog, clients, paths
from . import models as model_artifacts

ROOT = paths.ROOT
RUNTIME_DIR = paths.RUNTIME
PORT = 8000
# Either stops `splash serve` wherever it is. The programs with handlers of
# their own, the installer it runs and the server it executes, start with
# them blocked, not ignored, until those handlers are in place, so one sent
# meanwhile waits for its handler instead of being lost or ending the
# program in a traceback. make and the device check run with them unblocked.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class LauncherError(RuntimeError):
    pass


class StopSignal(KeyboardInterrupt):
    """One of STOP_SIGNALS, raised where it arrives, as Ctrl+C is."""

    def __init__(self, number):
        super().__init__(number)
        self.number = number


def _interrupt(number, _frame):
    raise StopSignal(number)


def _run_held(command, **options):
    """Run a program that unblocks the stop signals itself, holding them from
    its spawn. One the launcher takes meanwhile ends the program too."""
    signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    try:
        with subprocess.Popen(command, **options) as program:
            try:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
                return program.wait()
            except BaseException:
                program.kill()
                raise
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)


def _base_url(port):
    return f"http://127.0.0.1:{port}"


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


def _ensure_installed(selection):
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
    # The engine refuses an unsupported Mac only once the model is prepared;
    # its own check refuses it before tens of GB are downloaded.
    check = subprocess.run(
        [str(paths.BINARY), "device-check"], capture_output=True, text=True
    )
    if check.returncode:
        # The binary's own refusal is its last line; one that dies before
        # main() (dyld on an older macOS) leaves a report worth showing whole.
        report = check.stderr.strip()
        raise LauncherError(
            report.splitlines()[-1].removeprefix("error: ")
            if check.returncode > 0 and report
            else f"the engine's device check failed: {report or f'status {check.returncode}'}"
        )
    command = [
        str(paths.PYTHON),
        str(ROOT / "install/models.py"),
        "--models",
        str(selection.models_root),
        "--model",
        selection.model,
        "prepare",
    ]
    for flag, value in (
        ("--revision", selection.revision),
        ("--draft-model", selection.draft_model),
    ):
        if value is not None:
            command[-1:-1] = [flag, value]
    if selection.language_only:
        command.insert(-1, "--language-only")
    if _run_held(command, cwd=ROOT):
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


def _check_port(host, port):
    """Raise OSError if another process owns host:port. The probe binds with
    SO_REUSEADDR, as the HTTP listener does, so closed connections in
    TIME_WAIT do not block a restart; a live listener at the address still
    refuses the bind. The option also lets the bind succeed beside another
    process's listener at a wider or narrower address of the port (0.0.0.0
    or a dual-stack :: beside 127.0.0.1, or the reverse), and the two would
    then split the address's connections, so a listener that accepts one
    there owns the port too."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        address = probe.getsockname()[0]
    # A wildcard address is tried on loopback, where the launcher's clients
    # connect.
    if address == "0.0.0.0":
        address = "127.0.0.1"
    with socket.socket() as client:
        client.settimeout(1)
        if client.connect_ex((address, port)) == 0:
            raise OSError(errno.EADDRINUSE, os.strerror(errno.EADDRINUSE))


def serve(args):
    # Started in the background from a non-interactive shell, the launcher
    # inherits SIGINT as ignored; take both stop signals from the start.
    for number in STOP_SIGNALS:
        signal.signal(number, _interrupt)
    if args.offline:
        # The installer, the catalog refresh and the server all read it.
        os.environ["HF_HUB_OFFLINE"] = "1"
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
                "Splash is being upgraded; wait for the upgrade to finish"
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
        try:
            _check_port(args.host, args.port)
        except OSError as error:
            raise LauncherError(
                f"cannot bind {args.host}:{args.port}: {error}"
            ) from None
        selection = model_artifacts.Selection.of(
            paths.MODELS,
            args.model,
            revision=args.revision,
            language_only=args.language_only,
            draft_model=args.draft_model,
        )
        _ensure_installed(selection)
        # A concurrent install may advance the selection link. Keep this
        # process's tokenizer, draft and target on one immutable assembly,
        # held until the server exits.
        root, record = assembly.hold(selection.link, selection.models_root)
        if record is not None:
            os.set_inheritable(record.fileno(), True)
        # The server package of this installation, from any working
        # directory: -P keeps the directory, which may hold a package of the
        # same name, off sys.path, and PYTHONPATH names the root.
        command = [
            str(paths.PYTHON),
            "-u",
            "-P",
            "-m",
            "server.server",
            str(root),
            "--tokenizer",
            str(root / "tokenizer"),
            "--model",
            args.model,
            "--binary",
            str(paths.BINARY),
            "--port",
            str(args.port),
            *serve_options.serve_argv(args),
        ]
        environment = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            TRANSFORMERS_VERBOSITY="error",
            PYTHONPATH=str(ROOT),
            **serve_options.serve_environment(args),
        )
        # Detached, because execve replaces this process a line later and a
        # thread would not survive it. Failure is silent by design.
        catalog.spawn_refresh()
        os.set_inheritable(installation.fileno(), True)
        os.set_inheritable(lock.fileno(), True)
        # The exec resets the handlers; the server unblocks the signals once
        # its own are in place, past its imports.
        signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
        os.execve(command[0], command, environment)


def coding_client(args):
    path = clients.find_executable(args.command)
    listing = _request_json("/v1/models", port=args.port)
    if listing is None:
        raise LauncherError(
            f"No ready Splash server at {_base_url(args.port)}. "
            "Run 'splash serve --model <HF_REPO_ID>' "
            "in another terminal first."
        )
    models = listing.get("data", []) if isinstance(listing, dict) else []
    if (
        not isinstance(models, list)
        or not models
        or not isinstance(models[0], dict)
        or models[0].get("owned_by") != "splash"
        or type(models[0].get("context_length")) is not int
        or models[0]["context_length"] <= 0
    ):
        raise LauncherError("Could not identify the local Splash server")
    # The first entry is the name responses report.
    model, context = models[0].get("id"), models[0]["context_length"]
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
        input_modalities=models[0].get("input_modalities"),
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


def _version():
    if not paths.PACKAGED:
        return "Splash (source checkout)"
    return "Splash " + str(
        json.loads((paths.ROOT / "release.json").read_text())["version"]
    )


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
            "  splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M\n"
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
        description="Load an upstream model, automatically select its DFlash2 draft, and serve in the foreground.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M\n"
            "  splash serve --model unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M --max-context 128K\n\n"
            "SIZE is bytes, or a number with K, M or G (1024-based), such as 28G.\n"
            "DURATION is seconds, or a number with s, m or h, such as 30m.\n\n"
            "After Ready, open http://127.0.0.1:PORT in a browser on this Mac, or\n"
            "connect an installed agent; both reach the server on loopback, which\n"
            "the default --host and 0.0.0.0 include. The startup summary and\n"
            "/status report the effective context limit; a client may impose a\n"
            "smaller one. Keep this terminal open; Ctrl+C stops serving."
        ),
    )
    # The launcher's own options join the shared options' groups: the model
    # options first, --port among the network options.
    groups = serve_options.option_groups(server)
    model = groups["model"]
    model.add_argument(
        "--model",
        type=model_artifacts.parse_model_id,
        required=True,
        metavar="OWNER/REPO[:VARIANT]",
        help="upstream Hugging Face model, with a GGUF variant after ':' (e.g. :UD-Q4_K_M)",
    )
    model.add_argument(
        "--revision",
        metavar="REVISION",
        help="model branch, tag or commit (default: the repository's default branch)",
    )
    model.add_argument(
        "--draft-model",
        type=model_artifacts.parse_draft_model,
        metavar="DRAFT",
        help="DFlash2 draft repository or local directory to use instead of the "
        "automatically selected one",
    )
    model.add_argument(
        "--language-only",
        action="store_true",
        help="skip vision preparation and loading; image and PDF input is refused",
    )
    model.add_argument(
        "--offline",
        action="store_true",
        help="start the installed model without contacting the Hugging Face Hub (as HF_HUB_OFFLINE=1)",
    )
    groups["network"].add_argument(
        "--port",
        type=_parse_port,
        default=os.environ.get("SPLASH_PORT", str(PORT)),
        help="HTTP port (default: SPLASH_PORT or 8000)",
    )
    serve_options.add_serve_arguments(server, groups)
    for name in clients.INSTALL_URLS:
        commands.add_parser(name, help=f"connect {name} to the running server")
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve_options.check_serve_arguments(parser, args)
    if args.command in clients.INSTALL_URLS:
        try:
            args.port = _parse_port(os.environ.get("SPLASH_PORT", str(PORT)))
        except argparse.ArgumentTypeError as error:
            parser.error(f"SPLASH_PORT: {error}")
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
    except StopSignal as stop:
        # The status a shell gives a program the signal ends: 130 for
        # SIGINT, 143 for SIGTERM.
        return 128 + stop.number
    except KeyboardInterrupt:
        return 128 + signal.SIGINT


if __name__ == "__main__":
    raise SystemExit(main())
