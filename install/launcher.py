#!/usr/bin/env python3
"""Serve in the foreground, or connect an installed agent to the local server."""

import argparse
import contextlib
import errno
import fcntl
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script by the PATH wrappers and ./splash: import siblings as
    # the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

# The options, the record and the console serve shares with the server, in
# modules of the standard library alone: the launcher runs before .venv
# exists.
from server import diagnostics, serve_options, serve_record

from . import clients, paths
from . import models as model_artifacts

ROOT = paths.ROOT
RUNTIME_DIR = paths.RUNTIME
# Either stops `splash serve` wherever it is. The programs with handlers of
# their own, the installer it runs and the server it executes, start with
# them blocked, not ignored, until those handlers are in place, so one sent
# meanwhile waits for its handler instead of being lost or ending the
# program in a traceback. make and the device check run with them unblocked.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class LauncherError(RuntimeError):
    """A refusal of the launcher's, of a serve_record kind. said: whether the
    program that failed printed it already, so the launcher records it
    without printing it again. subject: what the launcher could not do, as
    "cannot prepare PATH", which the printed line puts before the message
    and the record leaves to the app."""

    def __init__(
        self, message, kind=serve_record.INTERNAL, *, said=False, subject=None
    ):
        super().__init__(message)
        self.kind = kind
        self.said = said
        self.subject = subject


class ArgumentRefusal(SystemExit):
    """A command line the launcher refuses, with argparse's status 2."""

    def __init__(self, message):
        super().__init__(2)
        self.message = message


class _Parser(argparse.ArgumentParser):
    """An argument parser whose refusal carries its message (ArgumentRefusal),
    which serve records."""

    def error(self, message):
        try:
            super().error(message)
        except SystemExit:
            raise ArgumentRefusal(message) from None


class StopSignal(KeyboardInterrupt):
    """One of STOP_SIGNALS, raised where it arrives, as Ctrl+C is."""

    def __init__(self, number):
        super().__init__(number)
        self.number = number


def _interrupt(number, _frame):
    raise StopSignal(number)


def _run_held(command, *, relay=None, **options):
    """Run a program that unblocks the stop signals itself, holding them from
    its spawn. One the launcher takes meanwhile ends the program too. With
    `relay`, each line of its output goes there as it comes."""
    signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    if relay is not None:
        options |= {"stdout": subprocess.PIPE, "text": True, "errors": "replace"}
    try:
        with subprocess.Popen(command, **options) as program:
            try:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
                for line in program.stdout if relay is not None else ():
                    relay(line)
                return program.wait()
            except BaseException:
                program.kill()
                raise
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)


def _flush_console():
    """Wait, LAST_LINES_SECONDS at most, for the launcher's lines to be
    written before an exec, which drops those its console (diagnostics) still
    holds."""
    diagnostics.flush_console(diagnostics.LAST_LINES_SECONDS)


def _base_url(port):
    return f"http://127.0.0.1:{port}"


# The server is on this Mac's loopback: requests to it connect directly, so
# no proxy, the environment's or the system's, receives the server's key.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request_json(path, timeout=2, *, port=serve_options.DEFAULT_PORT):
    request = urllib.request.Request(_base_url(port) + path)
    if key := os.environ.get("SPLASH_API_KEY"):
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with _DIRECT.open(request, timeout=timeout) as response:
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


def _ensure_runtime():
    if not paths.PACKAGED:
        # Serialize builds across ports; make keeps the lock if the launcher exits.
        with (RUNTIME_DIR / "build.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                diagnostics.print_line("Another Splash build is running; waiting...")
                fcntl.flock(lock, fcntl.LOCK_EX)
            for command in (
                ["make", "platform-check", "install-environment"],
                ["make", "-j4", "all"],
            ):
                if subprocess.run(
                    command, cwd=ROOT, pass_fds=(lock.fileno(),)
                ).returncode:
                    raise LauncherError("source build failed")
    # The engine refuses an unsupported Mac only once the model is prepared;
    # its own check refuses it before tens of GB are downloaded.
    try:
        model_artifacts.run_engine(["device-check"], "device check")
    except model_artifacts.ModelError as error:
        raise LauncherError(str(error), serve_record.DEVICE) from None


def _ensure_installed(selection):
    _ensure_runtime()
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
    # Its messages go on to the terminal; its failure, an error event
    # (install/models.py main), is what the launcher records. The installer
    # prints that failure on stderr itself, so only one it ended without is
    # the launcher's to say.
    failure = None

    def relay(line):
        nonlocal failure
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        if isinstance(event, dict) and event.get("event") == "error":
            failure = event["message"]
        else:
            diagnostics.print_line(line.removesuffix("\n"))

    if _run_held(command, cwd=ROOT, relay=relay):
        if failure is None:
            raise LauncherError(
                "model download or verification failed", serve_record.INSTALL
            )
        raise LauncherError(failure, serve_record.INSTALL, said=True)


def _prepare_local_model(args):
    _ensure_runtime()
    # Preparation needs Splash's dependencies, which the launcher's Python
    # need not have. The helper prints only the public interface's result,
    # or its refusal as a serve record failure; progress goes to the
    # terminal on stderr.
    command = [
        str(paths.PYTHON),
        "-P",
        "-m",
        "install.local",
        "--models",
        str(paths.MODELS),
    ]
    for flag, value in (("--draft-model", args.draft_model), ("--mmproj", args.mmproj)):
        if value is not None:
            command = [*command, flag, value]
    if args.language_only:
        command = [*command, "--language-only"]
    command = [*command, "--", args.model_dir]
    with tempfile.TemporaryFile(mode="w+") as result:
        failed = _run_held(
            command,
            cwd=ROOT,
            stdout=result,
            env=dict(os.environ, PYTHONPATH=str(ROOT)),
        )
        result.seek(0)
        output = result.read().strip()
    if failed:
        # Its refusal; a helper that crashed printed none.
        try:
            refusal = json.loads(output)
            message, kind = refusal["message"], refusal["kind"]
        except (ValueError, TypeError, KeyError):
            raise LauncherError("local model preparation failed") from None
        raise LauncherError(message, kind, subject=f"cannot prepare {args.model_dir}")
    prepared = Path(output)
    if not prepared.is_absolute():
        raise LauncherError("local model preparation returned no absolute directory")
    return prepared


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
    if (path := serve_options.unix_path(host)) is not None:
        with socket.socket(socket.AF_UNIX) as client:
            if client.connect_ex(path) == 0:
                raise OSError(errno.EADDRINUSE, os.strerror(errno.EADDRINUSE))
        return
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


def _prepare_start(args):
    """The model directory to serve, once the port is free and the model
    prepared."""
    # Fail before downloads/builds if another service owns the selected port.
    # The HTTP server also binds before loading weights, closing the race.
    try:
        _check_port(args.host, args.port)
    except OSError as error:
        raise LauncherError(
            f"cannot bind {args.host}:{args.port}: {error}", serve_record.PORT
        ) from None
    if args.model_dir is not None:
        return _prepare_local_model(args)
    selection = model_artifacts.Selection.of(
        paths.MODELS,
        args.model,
        revision=args.revision,
        language_only=args.language_only,
        draft_model=args.draft_model,
    )
    _ensure_installed(selection)
    return selection.link


def _server(args, model_dir, record):
    """The server's command line and environment."""
    # The server package of this installation, from any working directory:
    # -P keeps the directory, which may hold a package of the same name, off
    # sys.path, and PYTHONPATH names the root. The server holds the assembly
    # the selection link names when it starts, and serves from it until it
    # exits, however an installation advances the link meanwhile.
    command = [
        str(paths.PYTHON),
        "-u",
        "-P",
        "-m",
        "server.server",
        str(model_dir),
        "--model",
        args.model,
        "--binary",
        str(paths.BINARY),
        "--port",
        str(args.port),
        "--serve-record",
        str(record),
        *serve_options.serve_argv(args),
    ]
    environment = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        TRANSFORMERS_VERBOSITY="error",
        PYTHONPATH=str(ROOT),
        **serve_options.serve_environment(args),
    )
    return command, environment


def _serve_lock(port):
    return RUNTIME_DIR / f"serve-{port}.lock"


def _serve_record(port):
    """Beside the port's lock, which cannot be replaced atomically without
    breaking its flock: the app follows a start there. It is the lock
    holder's to write."""
    return RUNTIME_DIR / f"serve-{port}.json"


def _record_refusal(port, failure):
    """Record a start refused before it held the port's lock, under that lock.
    A lock this process cannot take leaves the record as it is: one another
    process holds, whose record it is, or one that cannot be created. The
    refusal printed is the answer."""
    with contextlib.ExitStack() as held:
        try:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            lock = held.enter_context(_serve_lock(port).open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        serve_record.failed(_serve_record(port), failure)


def serve(args):
    # Started in the background from a non-interactive shell, the launcher
    # inherits SIGINT as ignored; take both stop signals from the start.
    for number in STOP_SIGNALS:
        signal.signal(number, _interrupt)
    if args.offline:
        # The installer and the server both read it.
        os.environ["HF_HUB_OFFLINE"] = "1"
    if sys.dont_write_bytecode:
        # A child inherits the environment, not -B: every Python program
        # serve starts writes no bytecode either (the server's PDF workers,
        # started with no environment, have -B of their own), so a runtime
        # inside a signed app never gains __pycache__ folders, which break
        # its signature.
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # Keep both locks across exec until the foreground server exits.
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    record = _serve_record(args.port)
    with (
        _serve_lock(args.port).open("a+") as lock,
        (RUNTIME_DIR / "serve.lock").open("a+") as installation,
    ):
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
        # The record is the lock holder's: from here on a start that fails
        # says why in it, as the server, which keeps this pid, does too. A
        # start that cannot record it began fails here, before any model work
        # (serve_record).
        serve_record.starting(record)
        try:
            # Servers share the installation; upgrades require exclusive
            # access.
            try:
                fcntl.flock(installation, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LauncherError(
                    "Splash is being upgraded; wait for the upgrade to finish",
                    serve_record.UPGRADE,
                ) from None
            command, environment = _server(args, _prepare_start(args), record)
            os.set_inheritable(installation.fileno(), True)
            os.set_inheritable(lock.fileno(), True)
            _flush_console()
            # The exec resets the handlers; the server unblocks the signals
            # once its own are in place, past its imports.
            signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
            os.execve(command[0], command, environment)
        except LauncherError as error:
            serve_record.failed(record, serve_record.failure(error.kind, str(error)))
            raise
        except OSError as error:
            serve_record.failed(
                record, serve_record.failure(serve_record.INTERNAL, str(error))
            )
            raise


def _renders_later_system(port):
    """Whether every chat template of the server renders a later system
    message in place; a server that does not say renders none."""
    status = _request_json("/status", port=port)
    template = status.get("chat_template") if isinstance(status, dict) else None
    outcome = template.get("later_system") if isinstance(template, dict) else None
    outcomes = list(outcome.values()) if isinstance(outcome, dict) else [outcome]
    return all(outcome in ("native", "patched") for outcome in outcomes)


def coding_client(args):
    path = clients.find_executable(args.command)
    listing = _request_json("/v1/models", port=args.port)
    if listing is None:
        message = (
            f"No ready Splash server at {_base_url(args.port)}. "
            "Run 'splash serve --model <HF_REPO_ID>' in another terminal first."
        )
        # OpenCode and Hermes have a --port of their own, which goes after --.
        if args.explicit_port:
            message += (
                f" If --port was meant for {args.command} itself, put it after --."
            )
        raise LauncherError(message)
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
        later_system=args.command == "pi" and _renders_later_system(args.port),
    )
    diagnostics.print_line(
        f"Starting {args.command}: {model} · {context:,} context tokens"
    )
    if args.command == "claude":
        diagnostics.print_line(
            "Claude hosted WebSearch is unavailable. "
            "WebFetch, local tools and MCP are unchanged."
        )
    elif args.command == "codex":
        diagnostics.print_line(
            "Codex hosted WebSearch is disabled: Splash does not provide "
            "OpenAI's search service. Local tools and MCP are unchanged."
        )
    _flush_console()
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


def _local_path(value):
    # Absolute, its parent resolved, as the local helper takes it and as a
    # refusal names it. The file's own name stays: resolving a GGUF's link
    # in a Hub snapshot would replace its .gguf name with a cache blob's
    # digest.
    path = Path(os.path.abspath(Path(value).expanduser()))
    return str(path.parent.resolve() / path.name)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    client_args = []
    is_client = bool(argv and argv[0] in clients.INSTALL_URLS)
    if not is_client and "--" in argv:
        boundary = argv.index("--")
        argv, client_args = argv[:boundary], argv[boundary + 1 :]
    parser = _Parser(
        prog="splash",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M\n"
            "  splash opencode  # in another terminal, after Ready\n\n"
            "Use splash serve --help for server settings. An agent command takes\n"
            "--port PORT for a server on another port and passes every other\n"
            "argument, including --help, to the installed agent. To pass the\n"
            "agent's own --port, start its arguments with --:\n"
            "  splash opencode --port 8001 -- --port 4096"
        ),
    )
    parser.add_argument("--version", action="version", version=_version())
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser(
        "serve",
        help="run the local server; Ctrl+C stops it",
        description="Load an upstream or local model with its DFlash2 draft, and serve in the foreground.",
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
        allow_abbrev=False,
    )
    # The launcher's own options join the shared options' groups: the model
    # options first, --port among the network options.
    groups = serve_options.option_groups(server)
    model = groups["model"]
    model.add_argument(
        "--model",
        type=serve_options.parse_model_id,
        required=True,
        metavar="OWNER/REPO[:VARIANT]",
        help="upstream Hugging Face model, with a GGUF variant after ':' (e.g. :UD-Q4_K_M); "
        "with --model-dir, the ID reported by the API",
    )
    model.add_argument(
        "--model-dir",
        type=_local_path,
        metavar="PATH",
        help="prepare a local GGUF file or GGUF/MLX model directory without Hub access",
    )
    model.add_argument(
        "--mmproj",
        type=_local_path,
        metavar="FILE",
        help="local GGUF vision projector (requires --model-dir)",
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
        "automatically selected one (a local directory with --model-dir)",
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
    _add_port(
        groups["network"],
        f"HTTP port (default: SPLASH_PORT or {serve_options.DEFAULT_PORT})",
    )
    serve_options.add_serve_arguments(server, groups)
    for name in clients.INSTALL_URLS:
        client = commands.add_parser(
            name,
            help=f"connect {name} to the running server",
            add_help=False,
            allow_abbrev=False,
        )
        _add_port(
            client,
            f"server port (default: SPLASH_PORT or {serve_options.DEFAULT_PORT})",
        )
    if is_client:
        # Only --port belongs to Splash; preserve the agent's other arguments,
        # including --help, and stop interpreting options at its separator.
        args, client_args = parser.parse_known_args(argv)
        if client_args[:1] == ["--"]:
            client_args = client_args[1:]
    else:
        args = parser.parse_args(argv)
    # An explicit --port wins; SPLASH_PORT is read only without one.
    args.explicit_port = args.port is not None
    if not args.explicit_port:
        try:
            args.port = _environment_port()
        except argparse.ArgumentTypeError as error:
            parser.error(f"SPLASH_PORT: {error}")
    if args.command == "serve":
        if args.model_dir is not None:
            if args.revision is not None:
                parser.error("--revision cannot be used with --model-dir")
            if args.draft_model is not None and not Path(args.draft_model).is_dir():
                parser.error(
                    "--draft-model must be a local DFlash2 draft directory with --model-dir"
                )
        elif args.mmproj is not None:
            parser.error("--mmproj requires --model-dir")
        if args.mmproj is not None and args.language_only:
            parser.error("--mmproj cannot be used with --language-only")
        serve_options.check_serve_arguments(parser, args)
    if client_args and args.command == "serve":
        parser.error("arguments after -- are only supported for coding clients")
    args.client_args = client_args
    return args


def _add_port(parser, help_text=None):
    """The --port option, which serve, the agent commands and a refused serve
    command line (_serve_port) read alike."""
    parser.add_argument("--port", type=_parse_port, help=help_text)


def _environment_port():
    return _parse_port(os.environ.get("SPLASH_PORT", str(serve_options.DEFAULT_PORT)))


def _serve_port(argv):
    """The port a serve command line names, read alone, so that a refusal of
    its other arguments still reaches the record there; None for another
    command, or a port that does not parse."""
    if argv[:1] != ["serve"]:
        return None
    # As serve's parser: no abbreviations.
    probe = argparse.ArgumentParser(
        add_help=False, allow_abbrev=False, exit_on_error=False
    )
    _add_port(probe)
    try:
        port = probe.parse_known_args(argv[1:])[0].port
        return _environment_port() if port is None else port
    except (argparse.ArgumentError, argparse.ArgumentTypeError):
        return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parse_args(argv)
    except ArgumentRefusal as refusal:
        if (port := _serve_port(argv)) is not None:
            _record_refusal(
                port, serve_record.failure(serve_record.CONFIGURATION, refusal.message)
            )
        raise
    try:
        return serve(args) if args.command == "serve" else coding_client(args)
    except LauncherError as error:
        if not error.said:
            subject = f"{error.subject}: " if error.subject else ""
            diagnostics.print_line(f"error: {subject}{error}", error=True)
        return 1
    except (clients.ClientError, OSError, serve_record.NotRecorded) as error:
        diagnostics.print_line(f"error: {error}", error=True)
        return 1
    except StopSignal as stop:
        # The status a shell gives a program the signal ends: 130 for
        # SIGINT, 143 for SIGTERM.
        return 128 + stop.number
    except KeyboardInterrupt:
        return 128 + signal.SIGINT


if __name__ == "__main__":
    raise SystemExit(main())
