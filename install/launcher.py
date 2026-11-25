#!/usr/bin/env python3
"""Serve in the foreground, generate one video, or connect an installed agent
to the local server."""

import argparse
import contextlib
import errno
import fcntl
import http.client
import json
import mimetypes
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
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
from server import diagnostics, serve_options, serve_record, unix_socket

from . import clients, families, paths
from . import models as model_artifacts

ROOT = paths.ROOT
RUNTIME_DIR = paths.RUNTIME
# Either stops `splash serve` or `splash generate` wherever it is. The
# programs with handlers of their own, the installer it runs and the server
# it executes or starts, start with them blocked, not ignored, until those
# handlers are in place, so one sent meanwhile waits for its handler instead
# of being lost or ending the program in a traceback. make and the device
# check run with them unblocked.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)
# How often generate asks a running server for its job, and how long that
# server may stay silent before generate takes it for stopped.
POLL_SECONDS = 1.0
SILENCE_SECONDS = 30.0
# The free ports generate's own server is started on at most: another
# program may take the one found before the server binds it. That server
# may take STOP_SECONDS to stop, its engine's videos.CLOSE_SECONDS among
# them, before it is killed.
PORT_ATTEMPTS = 3
STOP_SECONDS = 60.0


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


# The --host values under which a server listens at _base_url's address: the
# default, its name, and every IPv4 address.
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0")


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


def _call(port, path, data=None, content_type=None, method=None, timeout=30):
    """The body of one request to the server on port, with SPLASH_API_KEY as
    _request_json sends it: LauncherError with the server's message for an
    error status, OSError when the server does not answer."""
    request = urllib.request.Request(_base_url(port) + path, data, method=method)
    if key := os.environ.get("SPLASH_API_KEY"):
        request.add_header("Authorization", f"Bearer {key}")
    if content_type is not None:
        request.add_header("Content-Type", content_type)
    try:
        with _DIRECT.open(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        with error:
            if error.code == 401:
                raise LauncherError(
                    "Splash authentication failed; set SPLASH_API_KEY to the server's key"
                ) from None
            try:
                message = json.loads(error.read())["error"]["message"]
            except (
                OSError,
                ValueError,
                LookupError,
                TypeError,
                http.client.HTTPException,
            ):
                message = f"HTTP {error.code}"
        raise LauncherError(message) from None
    except http.client.HTTPException as error:
        raise OSError(f"the server's answer broke off: {error!r}") from None


def _ensure_runtime(video=False):
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
    # its own check refuses it before tens of GB are downloaded, with the
    # video engine's rule too for a video model (video).
    check = ["device-check", "video"] if video else ["device-check"]
    try:
        model_artifacts.run_engine(check, "device check")
    except model_artifacts.ModelError as error:
        raise LauncherError(str(error), serve_record.DEVICE) from None


def _ensure_installed(selection, spec=None):
    _ensure_runtime(video=selection.repo_id == families.H3.release)
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
        ("--spec", spec),
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


def _printable(value):
    """Whether `value` is a string a line can show as it is."""
    return isinstance(value, str) and value != "" and value.isprintable()


def _serve_lock_owner(lock):
    """The process holding a port's lock, by the names a refusal words
    (_words): its PID, model and port, as serve wrote them in the lock, with
    its --host where the lock names one and the phase its serve record says
    where the record is that PID's; none for a lock without the first
    three."""
    try:
        lock.seek(0)
        owner = json.load(lock)
    except (OSError, UnicodeError, ValueError):
        return {}
    if not isinstance(owner, dict):
        return {}
    pid, model, port = owner.get("pid"), owner.get("model"), owner.get("port")
    if (
        type(pid) is not int
        or pid <= 0
        or not _printable(model)
        or type(port) is not int
        or not 1 <= port <= 65535
    ):
        return {}
    names = {"PID": pid, "model": model, "port": port}
    if _printable(host := owner.get("host")):
        names["host"] = host
    try:
        record = json.loads(_serve_record(port).read_bytes())
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and record.get("pid") == pid:
        if _printable(phase := record.get("phase")):
            names["phase"] = phase
    return names


def _words(owner):
    """A lock's owner (_serve_lock_owner) in a refusal's words, after
    "serving"; nothing for a lock that names none."""
    if not owner:
        return ""
    return f" ({', '.join(f'{name} {value}' for name, value in owner.items())})"


def _port_holder(port):
    """The process holding `port`'s lock, as _serve_lock_owner names it, or
    None while none holds it: the lock taken shared without waiting, which
    only a holder refuses, as the app probes it."""
    try:
        lock = _serve_lock(port).open()
    except OSError:
        return None
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _serve_lock_owner(lock)
    return None


def _check_port(host, port):
    """Raise OSError if another process owns host:port. The probe binds with
    SO_REUSEADDR, as the HTTP listener does, so closed connections in
    TIME_WAIT do not block a restart; a live listener at the address still
    refuses the bind. The option also lets the bind succeed beside another
    process's listener at a wider or narrower address of the port (0.0.0.0
    or a dual-stack :: beside 127.0.0.1, or the reverse), and the two would
    then split the address's connections, so a listener that accepts one
    there owns the port too. A UNIX socket's path is claimed and bound as the
    HTTP listener claims and binds it, and given up again."""
    if (path := unix_socket.path(host)) is not None:
        with (
            contextlib.closing(unix_socket.Claim(path)) as claim,
            socket.socket(socket.AF_UNIX) as probe,
        ):
            claim.bind(probe)
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
        # A UNIX socket has no port: --host names all of where it listens.
        address = (
            args.host
            if unix_socket.path(args.host) is not None
            else f"{args.host}:{args.port}"
        )
        raise LauncherError(
            f"cannot bind {address}: {error}", serve_record.PORT
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


def _share_installation(installation):
    """Hold the installation lock shared, as every server does; an upgrade
    holds it exclusively."""
    try:
        fcntl.flock(installation, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        raise LauncherError(
            "Splash is being upgraded; wait for the upgrade to finish",
            serve_record.UPGRADE,
        ) from None


def _program(module, model_dir, model, arguments, environment):
    """The command line and environment of the server package's module on
    model_dir.

    The server package of this installation, from any working directory: -P
    keeps the directory, which may hold a package of the same name, off
    sys.path, and PYTHONPATH names the root. The program holds the assembly
    an installed model's selection link names when it starts, and runs from
    it until it exits, however an installation advances the link meanwhile."""
    command = [
        str(paths.PYTHON),
        "-u",
        "-P",
        "-m",
        module,
        str(model_dir),
        "--model",
        model,
        "--binary",
        str(paths.BINARY),
        *arguments,
    ]
    environment = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        TRANSFORMERS_VERBOSITY="error",
        PYTHONPATH=str(ROOT),
        **environment,
    )
    return command, environment


def _server(args, model_dir, record):
    """The server's command line and environment."""
    return _program(
        "server.server",
        model_dir,
        args.model,
        [
            "--port",
            str(args.port),
            "--serve-record",
            str(record),
            *serve_options.serve_argv(args),
        ],
        serve_options.serve_environment(args),
    )


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
        # inside a signed app gains no bytecode beside what it ships, which
        # would break its signature.
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
            # The port names the lock whatever --host listens on, a UNIX
            # socket too: a second server takes another.
            raise LauncherError(
                f"Splash is already serving{_words(_serve_lock_owner(lock))}; "
                "stop it with Ctrl+C first, or serve on another --port"
            ) from None
        lock.seek(0)
        lock.truncate()
        # With where it listens, which an agent command that finds no server
        # at 127.0.0.1:PORT names (coding_client).
        json.dump(
            {
                "pid": os.getpid(),
                "model": args.model,
                "port": args.port,
                "host": args.host,
            },
            lock,
        )
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


def _served_model(port, model=None):
    """The first /v1/models entry of the splash serve holding port's lock
    (serve), or None when none holds it, or when it still starts a model
    that is not a video model and generate names a model to run itself."""
    try:
        with _serve_lock(port).open() as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                return None
            except BlockingIOError:
                owner = _serve_lock_owner(lock)
                lock.seek(0)
                record = lock.read()
    except FileNotFoundError:
        return None
    listing = _request_json("/v1/models", port=port)
    models = listing.get("data") if isinstance(listing, dict) else None
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        # A language model still starting leaves --model the local path (owner
        # is empty unless its record names a model).
        started = json.loads(record)["model"] if owner else families.H3.release
        if model is not None and started.partition(":")[0] != families.H3.release:
            return None
        raise LauncherError(
            f"Splash is starting on port {port}{_words(owner)}; run generate once it is Ready"
        )
    return models[0]


def _status(message):
    # As the clip's program prints one (server/diagnostics.print_status), on
    # stderr.
    diagnostics.print_status(message, error=True)


def _write_whole(path, data):
    """Write path at once: a stop or a failure leaves no partial mp4."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        try:
            # The mode path.write_bytes gives.
            umask = os.umask(0)
            os.umask(umask)
            os.fchmod(file.fileno(), 0o666 & ~umask)
            file.write(data)
        except BaseException:
            os.unlink(file.name)
            raise
    try:
        os.replace(file.name, path)
    except BaseException:
        os.unlink(file.name)
        raise


def _generate_on_server(args, served, request, port=None):
    """Run the clip as a job of the video model's server on port, by default
    args.port (POST /v1/videos, server/videos.py), and write its mp4s."""
    port, model = args.port if port is None else port, served.get("id")
    running = f"Splash serves {model} on port {port}"
    if "video" not in (served.get("output_modalities") or ()):
        raise LauncherError(f"{running}, a language model, not a video model")
    repo, _, variant = (args.model or "").partition(":")
    if args.model is not None and repo != str(model).partition(":")[0]:
        raise LauncherError(f"{running}, not {args.model}")
    if variant:
        request["variant"] = variant
    if not args.out.parent.is_dir():
        raise LauncherError(f"{args.out.parent} is not a directory")
    files = [("input_reference", args.first_frame), ("last_frame", args.last_frame)]
    files += [("references", path) for path in args.reference]
    files = [(name, path) for name, path in files if path is not None]
    body, content_type = json.dumps(request).encode(), "application/json"
    if files:
        # The multipart form curl -F sends.
        boundary = "splash-" + os.urandom(16).hex()
        content_type, body = f"multipart/form-data; boundary={boundary}", b""
        for name, value in [*request.items(), *files]:
            if isinstance(value, Path):
                # The server finds a reference's type by its bytes or its name.
                quoted = value.name.replace("\\", "\\\\").replace('"', '\\"')
                head = (
                    f'; filename="{quoted}"\r\nContent-Type: application/octet-stream'
                )
                value = value.read_bytes()
            else:
                value = (
                    str(value).lower() if isinstance(value, bool) else value
                ).encode()
                head = "\r\nContent-Type: text/plain; charset=utf-8"
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"{head}\r\n\r\n'.encode()
            body += value + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
    job = None
    try:
        job = json.loads(_call(port, "/v1/videos", body, content_type, timeout=600))
        _status(f"Video · {job['id']} · {running}")
        started = time.monotonic()
        preview, shown, silent = args.out.with_suffix(".preview.mp4"), None, None
        # What the job waits for before its first phase, as last said, and
        # the time and progress its denoising was first seen at, which the
        # time left is estimated from.
        waiting, denoising = None, None
        while job["status"] not in ("completed", "failed"):
            time.sleep(POLL_SECONDS)
            try:
                job = json.loads(_call(port, f"/v1/videos/{job['id']}"))
                if job["status"] == "queued":
                    # The jobs before it that have not finished: the server
                    # runs them in their order of arrival.
                    older = f"/v1/videos?order=desc&after={job['id']}&limit=100"
                    ahead = sum(
                        other["status"] in ("queued", "in_progress")
                        for other in json.loads(_call(port, older))["data"]
                    )
            except (OSError, ValueError):
                silent = silent or time.monotonic()
                if time.monotonic() - silent > SILENCE_SECONDS:
                    raise LauncherError(f"it stopped answering during {job['id']}")
                continue
            silent = None
            if job["status"] == "queued":
                state = f"queued · {ahead} ahead"
            elif job["status"] == "in_progress" and not job["phase"]:
                state = "starting the engine"
            else:
                state = None
            if state is not None and state != waiting:
                waiting = state
                _status(f"Video · {state}")
            if job["phase"] and (job["phase"], job["progress"]) != shown:
                shown = job["phase"], job["progress"]
                now = time.monotonic()
                line = f"Video · {shown[0]} · {shown[1]}% · {now - started:.0f} s"
                if denoising is None:
                    if shown[0] == "denoise":
                        denoising = now, shown[1]
                elif shown[1] > denoising[1]:
                    rate = (shown[1] - denoising[1]) / (now - denoising[0])
                    line += f" · ~{(100 - shown[1]) / rate:.0f} s left"
                _status(line)
            if args.preview and job["preview"] and preview:
                content = f"/v1/videos/{job['id']}/content?variant=preview"
                _write_whole(preview, _call(port, content))
                _status(f"Video · preview · {preview}")
                preview = None
        if job["status"] == "failed":
            raise LauncherError((job["error"] or {}).get("message", "the clip failed"))
        _write_whole(args.out, _call(port, f"/v1/videos/{job['id']}/content"))
    except StopSignal:
        # The server cancels a running job at its next DiT block or decode clip.
        if job is not None:
            _status("Video · cancelling")
            with contextlib.suppress(OSError, LauncherError):
                _call(port, f"/v1/videos/{job['id']}", method="DELETE")
        raise
    except LauncherError as error:
        raise LauncherError(f"{running}: {error}") from None
    # The clip's frames, as the server counts them (videos.clip_frames).
    frames = int(job["seconds"]) * families.H3.fps
    while frames % families.H3.frame_period != families.H3.frame_offset:
        frames += 1
    wall = job["phases"]["total"]["seconds"]
    _status(f"Video · {args.out} · {frames} frames · {wall:.1f} s")
    return 0


def _reference_variant():
    """The H3 variant whose requests take references (the Ref2VA base)."""
    return next(
        n for n, v in families.H3.variants.items() if v.spec and v.spec["references"]
    )


def _check_generate(parser, args):
    """Refuse at once a generate command line that cannot make a clip, which
    the server would refuse only after an install, a load or an upload:
    keyframes and references together, a last frame alone, an
    input that is missing or a directory (a pipe passes), an --out that
    cannot be written, more references than a request takes, and a variant
    of MiniMax-H3 named in --model that the request's references do not
    fit."""
    H3 = families.H3
    if args.last_frame is not None and args.first_frame is None:
        parser.error(
            "--last-frame needs --first-frame: the clip runs from one to the other"
        )
    if args.reference and (args.first_frame or args.last_frame) is not None:
        parser.error(
            "--reference cannot be used with --first-frame or --last-frame: "
            "a clip takes references or keyframes, not both"
        )
    inputs = [("--first-frame", args.first_frame), ("--last-frame", args.last_frame)]
    for flag, path in [*inputs, *(("--reference", p) for p in args.reference)]:
        if path is not None and not path.exists():
            parser.error(f"{flag} {path}: no such file")
        if path is not None and path.is_dir():
            parser.error(f"{flag} {path} is a directory: name a file")
    if len(args.reference) > H3.max_references:
        parser.error(
            f"--reference: at most {H3.max_references} files ({H3.max_images} images, "
            f"{H3.max_videos} videos and {H3.max_audio} audio clips)"
        )
    if args.out.is_dir():
        example = args.out / "clip.mp4"
        parser.error(
            f"--out {args.out} is a directory: name the mp4, such as {example}"
        )
    if not args.out.parent.is_dir():
        parser.error(f"--out {args.out}: {args.out.parent} is not a directory")
    repo, _, variant = (args.model or "").partition(":")
    if repo != H3.release or variant not in H3.variants:
        return
    reference = _reference_variant()
    if variant == reference and not args.reference:
        parser.error(
            f"{args.model} takes --reference FILE (an image or a video at least); "
            f"a clip from the prompt or from keyframes takes {repo}:{H3.default_variant}"
        )
    if variant != reference and args.reference:
        parser.error(
            f"--reference takes the {reference} variant: --model {repo}:{reference}"
        )


# A request's `seconds` and `size` as server/videos.py takes them
# (videos.SECONDS_PATTERN, SIZE_PATTERN; clip_seconds and canvas check the
# rest).
_SECONDS = re.compile(r"[1-9][0-9]?")
_SIZE = re.compile(r"([1-9][0-9]{0,4})x([1-9][0-9]{0,4})")
_SEED = re.compile(r"[0-9]+")


class _Seconds(argparse.Action):
    """Stores --seconds or --duration, and the one given, which a refusal of
    its value names."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.seconds_flag = option_string


def _refuse_before_install(args, selection):
    """LauncherError for what MiniMax-H3's server would refuse only once the
    model is installed and loaded: a prompt, a seed, a length or a canvas
    outside the server's rules (videos.parse_request, clip_seconds and
    canvas, in the words of their refusals), references for an
    installation that holds or would install a variant without them, and a
    video or audio reference where the server's Python has no PyAV.
    Another model passes, to be refused once installed."""
    H3 = families.H3
    if selection.repo_id != H3.release:
        return
    if not args.prompt.strip():
        raise LauncherError("--prompt: prompt must be a non-empty string")
    seed = args.seed
    if seed is not None and not (_SEED.fullmatch(seed) and int(seed) < 2**32):
        raise LauncherError(
            f"--seed {seed}: seed must be an integer in [0, 4294967295]"
        )
    flag = getattr(args, "seconds_flag", None) or "--seconds"
    seconds = args.seconds if args.seconds is not None else H3.default_seconds
    if not (_SECONDS.fullmatch(seconds) and int(seconds) in H3.seconds):
        raise LauncherError(
            f"{flag} {seconds}: seconds must be a whole number from "
            f"{H3.seconds[0]} to {H3.seconds[-1]}"
        )
    size = args.size if args.size is not None else H3.size
    match = _SIZE.fullmatch(size)
    sides = sorted(map(int, match.groups())) if match else None
    if not (
        sides
        and all(side % H3.canvas_step == 0 for side in sides)
        and sides[0] in H3.canvas_tiers
        and sides[1] <= 2 * sides[0]
    ):
        raise LauncherError(
            f"--size {size}: size must be WIDTHxHEIGHT of a 768p or 480p canvas: both "
            "multiples of 32, the short side 768 or 480 and the long at most twice it, "
            "such as " + ", ".join(H3.sizes)
        )
    # A model ID without a variant keeps the one its installation holds, or
    # installs the default, which takes no references.
    reference = _reference_variant()
    weights = selection.link / "weights"
    if (
        args.reference
        and selection.variant is None
        and (not selection.link.exists() or weights.is_dir())
        and not (weights / f"dit-{H3.variants[reference].dit}").is_dir()
    ):
        raise LauncherError(
            f"--reference takes the {reference} variant, which {args.model} does not "
            f"hold: --model {H3.release}:{reference}"
        )
    # The server's Python decodes a video or an audio reference with PyAV,
    # which the release leaves out (server/references.py pyav). A source
    # checkout's environment that does not exist yet is the server's to
    # check.
    if (
        not all(map(_image, args.reference))
        and paths.PYTHON.exists()
        and subprocess.run(
            [str(paths.PYTHON), "-P", "-c", "import av"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
    ):
        raise LauncherError(families.pyav_missing(paths.PYTHON))


def _image(path):
    """Whether a --reference is an image as the server finds its type
    (references.content_type): by a file's leading bytes, else by its name,
    a pipe's by its name alone, which only the upload may read."""
    if path.is_file():
        with path.open("rb") as file:
            if file.read(8).startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
                return True
    return mimetypes.guess_type(path.name)[0] in ("image/png", "image/jpeg")


def _holds_video_model(link):
    """Whether the assembly an installed selection link names is a video
    model's (videos.serves, read without the server's packages)."""
    try:
        record = model_artifacts.read_json(link / "model.json")
    except model_artifacts.ModelError:
        return False
    return record.get("target_format") == families.H3.target_format


def _free_port():
    """A port free on loopback as the system picks one; the server binds it
    later, so another program may take it meanwhile (PORT_ATTEMPTS)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _spawn_server(command, environment, held):
    """Start the server's program with the stop signals held from its spawn,
    as _run_held does, its output on stderr, in a process group of its own:
    Ctrl+C reaches generate alone, which cancels the job, then stops it."""
    signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    try:
        return subprocess.Popen(
            command,
            env=environment,
            stdout=2,
            pass_fds=[file.fileno() for file in held],
            process_group=0,
        )
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)


def _await_server(server, record):
    """None once the server writing record says it is ready, or the failure
    it recorded (serve_record), which it printed itself; LauncherError for
    one that exited without either."""
    while True:
        exited = server.poll()
        try:
            state = json.loads(record.read_bytes())
        except (OSError, ValueError):
            state = None
        if isinstance(state, dict) and state.get("pid") == server.pid:
            if state.get("phase") == "ready":
                return None
            if state.get("phase") == "failed" and isinstance(
                state.get("failure"), dict
            ):
                return state["failure"]
        if exited is not None:
            raise LauncherError(f"the video server exited with status {exited}")
        time.sleep(POLL_SECONDS)


def _stop_server(server):
    """Stop the server as SIGTERM stops splash serve, waiting STOP_SECONDS
    for it to release its engine, else kill it; a stop meanwhile kills it."""
    if server.poll() is None:
        server.terminate()
    try:
        server.wait(timeout=STOP_SECONDS)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    except KeyboardInterrupt:
        server.kill()
        server.wait()
        raise


@contextlib.contextmanager
def _own_server(model, model_dir, held):
    """The port of a server of model_dir that generate starts for its clip,
    as splash serve starts one but on a free loopback port and without the
    port's lock, once it is ready; stopped once the clip ends, however it
    ends. It also stops itself once generate has exited (SPLASH_APP_PID)."""
    defaults = {option.dest: option.default() for option in serve_options.SERVE_OPTIONS}
    with tempfile.TemporaryDirectory(prefix="splash-generate-") as directory:
        for attempt in range(PORT_ATTEMPTS):
            port = _free_port()
            record = Path(directory) / f"serve-{port}.json"
            options = argparse.Namespace(model=model, port=port, **defaults)
            command, environment = _server(options, model_dir, record)
            environment["SPLASH_APP_PID"] = str(os.getpid())
            server = _spawn_server(command, environment, held)
            try:
                failure = _await_server(server, record)
            except BaseException:
                _stop_server(server)
                raise
            if failure is None:
                break
            _stop_server(server)
            if failure.get("kind") != serve_record.PORT or attempt + 1 == PORT_ATTEMPTS:
                raise LauncherError(
                    str(failure.get("message")),
                    failure.get("kind", serve_record.INTERNAL),
                    said=True,
                )
        try:
            yield port
        finally:
            _stop_server(server)


def generate(args):
    """Send one clip to the video model's splash serve on --port, as a job of
    its /v1/videos, when one runs there. Else install the model as serve
    does, start a server of it on a free port, send the clip there, and stop
    that server once the clip ends; a language model is refused once
    installed."""
    for number in STOP_SIGNALS:
        signal.signal(number, _interrupt)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    # The flags as a POST /v1/videos body, which the server checks; an absent
    # flag takes the route's default. The keyframes' and references' files
    # go as the multipart form's (_generate_on_server).
    fields = {
        "prompt": args.prompt,
        "seconds": args.seconds,
        "size": args.size,
        "seed": args.seed,
        "preview": args.preview,
        "audio": not args.no_audio,
    }
    request = {name: value for name, value in fields.items() if value is not None}
    served = _served_model(args.port, args.model)
    # A language model may run beside the video model: --model starts a
    # server of the video model then.
    video = served is not None and "video" in (served.get("output_modalities") or ())
    if served is not None and (video or args.model is None):
        return _generate_on_server(args, served, request)
    if args.model is None:
        # The status argparse gives a missing required argument.
        diagnostics.print_line(
            f"error: --model is required: no Splash server runs on port {args.port}",
            error=True,
        )
        return 2
    H3 = families.H3
    selection = model_artifacts.Selection.of(paths.MODELS, args.model)
    _refuse_before_install(args, selection)
    with (RUNTIME_DIR / "serve.lock").open("a+") as installation:
        _share_installation(installation)
        _ensure_installed(selection)
        if not _holds_video_model(selection.link):
            raise LauncherError(
                f"{args.model} is a language model: splash generate takes a video "
                f"model, such as {H3.release}; splash serve serves this one"
            )
        _status(H3.license)
        with _own_server(args.model, selection.link, (installation,)) as port:
            listing = _request_json("/v1/models", port=port)
            models = listing.get("data") if isinstance(listing, dict) else None
            if (
                not isinstance(models, list)
                or not models
                or not isinstance(models[0], dict)
            ):
                raise LauncherError(f"the video server on port {port} lists no model")
            return _generate_on_server(args, models[0], request, port)


def install(args):
    """Install the model as serve does, and start nothing: a variant of
    MiniMax-H3 that Splash does not ship from its spec file too (--spec)."""
    for number in STOP_SIGNALS:
        signal.signal(number, _interrupt)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with (RUNTIME_DIR / "serve.lock").open("a+") as installation:
        _share_installation(installation)
        selection = model_artifacts.Selection.of(paths.MODELS, args.model)
        _ensure_installed(selection, args.spec)
    return 0


def coding_client(args):
    path = clients.find_executable(args.command)
    listing = _request_json("/v1/models", port=args.port)
    if listing is None:
        holder = _port_holder(args.port)
        if holder is None:
            message = (
                f"No ready Splash server at {_base_url(args.port)}. "
                "Run 'splash serve --model <HF_REPO_ID>' in another terminal first."
            )
        elif holder.get("phase") == "ready" and (
            "host" not in holder or holder["host"] in _LOOPBACK_HOSTS
        ):
            # Its record says ready, and it listens where the agent commands
            # ask, or may: a lock that names no address, as 1.6.1's do not,
            # says nothing of where. So only that it did not answer is known:
            # a server that stops closes its listener first and holds the
            # port, its record still ready, until its engine has stopped.
            message = (
                f"Splash is serving{_words(holder)} but does not answer at "
                f"{_base_url(args.port)}: it may be stopping"
                + ("." if "host" in holder else ", or listen on another address.")
            )
        else:
            # The server holding the port listens elsewhere, on another
            # address or a UNIX socket, or is not ready yet; a second one
            # would be refused the port.
            message = (
                f"No ready Splash server at {_base_url(args.port)}, but Splash "
                f"is serving{_words(holder)}. Agent commands connect over TCP "
                "on loopback to a server that is ready."
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
        help="upstream Hugging Face model, with a GGUF variant after ':' (e.g. :UD-Q4_K_M), "
        f"or a video model's ({families.H3.release}:vdn-50); with --model-dir, the ID "
        "reported by the API",
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
    H3 = families.H3
    reference = _reference_variant()
    generator = commands.add_parser(
        "generate",
        help="generate one video clip with a video model",
        description="Write one clip of a prompt as an mp4, as a job of a splash\n"
        "serve of a video model on --port. With none there, install the video\n"
        "model as serve does, start a server of it on a free loopback port for\n"
        "the clip, and stop it once the clip ends. Ctrl+C cancels the clip.",
        epilog="Examples:\n"
        f"  # From the prompt alone: {H3.default_seconds} s at {H3.size}\n"
        f"  splash generate --model {H3.release} --out clip.mp4 \\\n"
        "    --prompt 'A golden retriever runs along a beach at sunset.'\n"
        "  # From a first frame (and a last one), portrait, 480p, 8 s\n"
        f"  splash generate --model {H3.release} --out clip.mp4 --size 480x832 \\\n"
        "    --seconds 8 --first-frame first.png --last-frame last.png \\\n"
        "    --prompt 'The dog turns to the camera.'\n"
        f"  # From references, which take the {reference} variant\n"
        f"  splash generate --model {H3.release}:{reference} --out clip.mp4 \\\n"
        "    --reference cat.png --reference street.mp4 \\\n"
        "    --prompt '<Picture 1> walks into the scene of <Video 1>.'\n\n"
        "A server serves every installed variant of its model, and --model\n"
        f"picks the clip's: references need {H3.release}:{reference} installed.\n\n"
        "The mp4 at --out, and the preview beside it, replace files of those\n"
        "names. Progress goes to stderr. Exit status: 0 once the mp4 is written,\n"
        "1 when the clip is refused or fails, 2 for a wrong command line, 130 or\n"
        "143 when Ctrl+C or SIGTERM stops it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    generator.add_argument(
        "--model",
        type=serve_options.parse_model_id,
        metavar="OWNER/REPO[:VARIANT]",
        help=f"upstream Hugging Face video model, such as {H3.release} (its "
        f"variant {H3.default_variant}, or the one installed), "
        f"{H3.release}:vdn-8 (8 steps) or {H3.release}:{reference} (for "
        "--reference); with a server on --port, its model by default",
    )
    generator.add_argument(
        "--port",
        type=_parse_port,
        help="the port of a running splash serve to send the clip to; with none "
        "there, generate starts a server of its own on a free port (default: "
        f"SPLASH_PORT or {serve_options.DEFAULT_PORT})",
    )
    generator.add_argument("--prompt", required=True, help="what the clip shows")
    generator.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="the mp4 to write, replaced if it exists, in a directory that exists",
    )
    generator.add_argument(
        "--seconds",
        "--duration",
        dest="seconds",
        action=_Seconds,
        metavar="SECONDS",
        help=f"clip length in whole seconds, {H3.seconds[0]} to {H3.seconds[-1]} "
        f"({H3.default_seconds} by default)",
    )
    generator.add_argument(
        "--size",
        metavar="WIDTHxHEIGHT",
        help="canvas: 480p 832x480 (the default), 864x480 or 480x832, 768p "
        "1344x768 or 768x1344, or another of either tier (both sides multiples "
        "of 32, the long at most twice the short)",
    )
    generator.add_argument(
        "--seed",
        metavar="SEED",
        help="noise seed, a whole number from 0 to 4294967295 (default 0)",
    )
    generator.add_argument(
        "--preview",
        action="store_true",
        help="also write the fast preview, to PATH's stem with .preview.mp4, "
        "as soon as it is decoded",
    )
    generator.add_argument(
        "--no-audio", action="store_true", help="a clip without its audio track"
    )
    generator.add_argument(
        "--first-frame",
        type=Path,
        metavar="IMAGE",
        help="start the clip from this PNG or JPEG, stretched onto the canvas: "
        "the canvas does not follow the image, so give a portrait image "
        "--size 768x1344 or 480x832",
    )
    generator.add_argument(
        "--last-frame",
        type=Path,
        metavar="IMAGE",
        help="end the clip on this PNG or JPEG, scaled to cover the canvas and "
        "centre-cropped; needs --first-frame",
    )
    generator.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="condition the clip on this image (PNG, JPEG), video (MP4, "
        "QuickTime, WebM, with its soundtrack) or audio clip (MP3, WAV, M4A, "
        "AAC), its kind from the file's content; repeat it for more (at most "
        f"{H3.max_images} images, {H3.max_videos} videos and {H3.max_audio} "
        f"audio clips, {H3.max_references} in all, an image or a video among "
        "them). The prompt names "
        "them per kind in the order given: <Picture 1>, <Video 1>, <Audio 1>, "
        "...; a video's soundtrack is the next <Audio j>, before its <Video k>. "
        f"Takes the {reference} variant; not with --first-frame or --last-frame",
    )
    installer = commands.add_parser(
        "install",
        help="install a model as serve does; no server starts",
        description="Install a model as serve does; no server starts. A "
        f"checkpoint of {families.H3.release}'s DiT that Splash does not ship "
        "installs as a variant of the name you give it, from a JSON spec "
        "(docs/video.md, Video installation).",
        epilog=f"Example:\n  splash install {families.H3.release}:my-finetune "
        "--spec my-finetune.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    installer.add_argument(
        "model", type=serve_options.parse_model_id, metavar="OWNER/REPO[:VARIANT]"
    )
    installer.add_argument(
        "--spec",
        type=_local_path,
        metavar="FILE",
        help="the variant's spec: checkpoint, adapter, schedule, references, qkfix",
    )
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
    if args.command == "install":
        # install takes no --port: its refusals end here.
        if client_args:
            parser.error("arguments after -- are only supported for coding clients")
        args.client_args = client_args
        return args
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
    if args.command == "generate":
        _check_generate(generator, args)
    if client_args and args.command in ("serve", "generate"):
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
        command = {"serve": serve, "generate": generate, "install": install}.get(
            args.command, coding_client
        )
        return command(args)
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
