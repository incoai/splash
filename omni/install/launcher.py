#!/usr/bin/env python3
"""Serve a speech or transcription model in the foreground, or choose one
and try it in a browser (splash-omni ui)."""

import argparse
import errno
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script by the PATH wrappers and ./splash-omni: import siblings as
    # the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

# The options serve shares with the server, in modules of the standard
# library alone: the launcher runs before .venv exists.
from server import serve_options

from . import assembly, families, paths
from . import models as model_artifacts

ROOT = paths.ROOT
RUNTIME_DIR = paths.RUNTIME
# Either stops `splash-omni serve` wherever it is. The programs with handlers of
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


def _build_from_source():
    """Build the engine and its environment with make, one build at a time
    across ports; make keeps the lock if the launcher exits."""
    with (RUNTIME_DIR / "build.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another Splash build is running; waiting...", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        for command in (
            ["make", "platform-check", "install-environment"],
            ["make", "-j4", "all"],
        ):
            if subprocess.run(command, cwd=ROOT, pass_fds=(lock.fileno(),)).returncode:
                raise LauncherError("source build failed; see the output above")


def _ensure_installed(selection, draft=True):
    _build_from_source()
    # The engine refuses an unsupported Mac only once the model is prepared;
    # its own check refuses it before gigabytes are downloaded.
    try:
        model_artifacts.run_engine(["device-check"], "device check")
    except model_artifacts.ModelError as error:
        raise LauncherError(str(error)) from None
    command = [
        str(paths.PYTHON),
        str(ROOT / "install/models.py"),
        "--models",
        str(selection.models_root),
        "--model",
        selection.model,
        "prepare",
    ]
    if selection.revision is not None:
        command[-1:-1] = ["--revision", selection.revision]
    # The default draft of a Qwen3-ASR-1.7B model downloads with it; the
    # server fetches a --draft-model of another itself.
    if not draft:
        command.append("--no-draft-model")
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
        # The installer and the server both read it.
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
            paths.MODELS, args.model, revision=args.revision
        )
        _ensure_installed(
            selection, draft=not args.no_draft_model and args.draft_model is None
        )
        # A concurrent install may advance the selection link. Keep this
        # process's tokenizer and target on one immutable assembly, held
        # until the server exits.
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
            families.SERVER,
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
        os.set_inheritable(installation.fileno(), True)
        os.set_inheritable(lock.fileno(), True)
        # The exec resets the handlers; the server unblocks the signals once
        # its own are in place, past its imports.
        signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
        os.execve(command[0], command, environment)


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
    return "splash-omni (source checkout)"


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="splash-omni",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            f"  splash-omni serve --model {families.SUPPORTED_MODELS[0]}\n"
            "  splash-omni ui    (choose the model and its settings in a browser)\n\n"
            "Use splash-omni serve --help for server settings."
        ),
    )
    parser.add_argument("--version", action="version", version=_version())
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser(
        "serve",
        help="run the local server; Ctrl+C stops it",
        description="Serve one of the supported Qwen3-TTS or Qwen3-ASR "
        "checkpoints in the foreground as a speech or transcription API. "
        "A checkpoint already installed or in the Hugging Face cache starts "
        "without contacting the Hub; only a missing one is downloaded.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported models:\n"
            + "".join(f"  {model}\n" for model in families.SUPPORTED_MODELS)
            + "\nExamples:\n"
            f"  splash-omni serve --model {families.SUPPORTED_MODELS[0]}\n"
            f"  splash-omni serve --model {families.SUPPORTED_MODELS[2]} --port 8001\n\n"
            "SIZE is bytes, or a number with K, M or G (1024-based), such as 28G.\n"
            "DURATION is seconds, or a number with s, m or h, such as 30m.\n\n"
            "After Ready, clients reach the server on loopback, which the\n"
            "default --host and 0.0.0.0 include. Keep this terminal open;\n"
            "Ctrl+C stops serving."
        ),
    )
    # The launcher's own options join the shared options' groups: the model
    # options first, --port among the network options.
    groups = serve_options.option_groups(server)
    model = groups["model"]
    model.add_argument(
        "--model",
        type=serve_options.parse_model_id,
        required=True,
        metavar="OWNER/REPO",
        help="one of the supported Hugging Face checkpoints (see below)",
    )
    model.add_argument(
        "--revision",
        metavar="REVISION",
        help="model branch, tag or commit (default: the repository's default branch)",
    )
    model.add_argument(
        "--offline",
        action="store_true",
        help="never contact the Hugging Face Hub, not even to download a "
        "missing model (as HF_HUB_OFFLINE=1); a model installed or in the "
        "Hugging Face cache starts without the Hub either way",
    )
    groups["network"].add_argument(
        "--port",
        type=_parse_port,
        help=f"HTTP port (default: SPLASH_PORT or {serve_options.DEFAULT_PORT})",
    )
    serve_options.add_serve_arguments(server, groups)
    page = commands.add_parser(
        "ui",
        help="open the setup and demo page in a browser; Ctrl+C stops it",
        description="Serve a page on loopback that picks a model and its "
        "options, downloads only that checkpoint, starts splash-omni serve with "
        "them, and then sends speech and transcription requests to the "
        "server's API.",
    )
    page.add_argument(
        "--port",
        type=_parse_port,
        help="the page's port (default: 7860, or the next free one)",
    )
    page.add_argument(
        "--no-browser",
        action="store_true",
        help="print the page's address instead of opening a browser",
    )
    page.add_argument(
        "--exit-with-parent",
        action="store_true",
        help="stop the page and its servers when the program that started it "
        "exits (the desktop app's own page)",
    )
    args = parser.parse_args(argv)
    if args.command == "ui":
        return args
    # An explicit --port wins; SPLASH_PORT is read only without one.
    args.explicit_port = args.port is not None
    if not args.explicit_port:
        try:
            args.port = _parse_port(
                os.environ.get("SPLASH_PORT", str(serve_options.DEFAULT_PORT))
            )
        except argparse.ArgumentTypeError as error:
            parser.error(f"SPLASH_PORT: {error}")
    serve_options.check_serve_arguments(parser, args)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "ui":
        # Standard library only, as the launcher: the page starts serve.
        from gui import app

        try:
            return app.main(
                args.port,
                open_browser=not args.no_browser,
                exit_with_parent=args.exit_with_parent,
            )
        except OSError as error:
            print(f"error: cannot serve the page: {error}", file=sys.stderr)
            return 1
    try:
        return serve(args)
    except (LauncherError, OSError) as error:
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
