"""`splash-omni ui`: the setup and demo page, served on loopback.

The page's backend does setup and lifecycle only: it lists the supported
checkpoints and what the cache holds of them, starts `splash-omni serve` with the
options chosen (session.py), reports its downloads, output and readiness,
and stops it. It never runs a model: the page's speech and transcription
requests are API calls to the server's OpenAI-compatible endpoints, made
from the browser, or through proxy.py's transparent forwarding when the
browser cannot reach the server directly.

Standard library only, with the installer and server modules that are too:
`splash-omni ui` runs before .venv exists.
"""

from __future__ import annotations

import errno
import json
import mimetypes
import os
import signal
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from install import launcher, models, paths, upstream
from server import serve_options

from . import proxy, session, voice

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 7860
# Ports tried after DEFAULT_PORT when it is taken and none was named.
PORT_ATTEMPTS = 20
HOST = "127.0.0.1"
# One feature's servers at a time: a page whose own servers are not the
# ones running would send its requests to none, or to the wrong model.
VOICE_RUNNING = "voice chat is running; end it first"
SPEECH_RUNNING = "a text to speech or transcription server is running; stop it first"
# How often the page checks whether the program that started it still runs
# (--exit-with-parent).
PARENT_POLL_SECONDS = 1.0
# The most a setup request or a proxied API request may carry.
MAX_SETUP_BYTES = 64 * 1024
MAX_PROXY_BYTES = 512 * 1024 * 1024
_LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")


def page_origins(port):
    """The origins the page is served from."""
    return tuple(f"http://{name}:{port}" for name in ("127.0.0.1", "localhost"))


def catalog():
    """The supported checkpoints and the draft, with what the cache and the
    installation hold of each, for the setup page."""
    entries = []
    for checkpoint in session.CHECKPOINTS:
        chosen = models.Selection.of(paths.MODELS, checkpoint.model)
        entries.append(
            {
                "id": checkpoint.model,
                "task": checkpoint.task,
                "precision": checkpoint.precision,
                "variant": checkpoint.variant,
                "label": checkpoint.label,
                "gigabytes": checkpoint.gigabytes,
                "installed": models.installation_kind(chosen.link) == models.ASSEMBLY,
                # Where the installer assembles it from without the Hub.
                "cached": upstream.cached_commit(chosen) is not None,
            }
        )
    draft = serve_options.DEFAULT_DRAFT_MODEL
    return {
        "models": entries,
        "variants": [asdict(variant) for variant in session.VARIANTS],
        "draft": {
            "id": draft,
            "gigabytes": session.DRAFT_GIGABYTES,
            "cached": upstream.draft_present(draft),
        },
        "defaults": {"port": serve_options.DEFAULT_PORT},
        "max_lanes": session.MAX_LANES,
        "built": paths.BINARY.is_file(),
    }


def port_status(port):
    """Whether `splash-omni serve` could listen on port: the launcher's own check
    (install/launcher.py), with what refuses it."""
    try:
        launcher._check_port(HOST, port)
    except OSError as error:
        return {"port": port, "free": False, "reason": error.strerror or str(error)}
    return {"port": port, "free": True, "reason": None}


def static_file(name):
    """The page file `name` names, or None: only a file directly in
    static/, by its exact name."""
    if not name or "/" in name or name.startswith("."):
        return None
    for path in STATIC.iterdir():
        if path.name == name and path.is_file():
            return path
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SplashUI"

    def log_message(self, format, *args):
        pass

    # Requests the page did not make are refused: a Host other than
    # loopback (a DNS-rebound page), or an Origin other than the page's own.

    def _allowed(self):
        host = self.headers.get("Host", "")
        port = self.server.server_port
        if host not in {f"{name}:{port}" for name in _LOOPBACK_NAMES}:
            self._json(403, {"error": "this page is served on loopback only"})
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in page_origins(port):
            self._json(403, {"error": f"origin {origin} may not use this page"})
            return False
        return True

    def _send(self, status, data, content_type, headers=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _body(self, limit):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if self.headers.get("Transfer-Encoding") or not 0 <= length <= limit:
            self.close_connection = True
            self._json(413, {"error": f"request bodies are limited to {limit} bytes"})
            return None
        return self.rfile.read(length) if length else b""

    def _route(self):
        parts = urlsplit(self.path)
        return parts.path, parts.query

    def _proxy(self):
        path, query = self._route()
        body = self._body(MAX_PROXY_BYTES)
        if body is None:
            return
        target = path.removeprefix("/proxy") + (f"?{query}" if query else "")
        proxy.forward(self, self.headers.get(proxy.UPSTREAM_HEADER), target, body)

    def do_GET(self):
        if not self._allowed():
            return
        path, query = self._route()
        if path.startswith("/proxy/"):
            self._proxy()
        elif path in ("/", "/index.html"):
            self._file(STATIC / "index.html")
        elif path in ("/voice", "/voice.html"):
            self._file(STATIC / "voice.html")
        elif path == "/api/voice/catalog":
            self._json(200, {**voice.catalog(), "speech": catalog()})
        elif path == "/api/voice/state":
            fields = parse_qs(query)
            since = {}
            for role in voice.ROLES:
                try:
                    since[role] = int(fields.get(role, ["0"])[0])
                except ValueError:
                    since[role] = 0
            self._json(200, self.server.voice.snapshot(since))
        elif path.startswith("/static/"):
            file = static_file(path.removeprefix("/static/"))
            if file is None:
                self._json(404, {"error": "not found"})
            else:
                self._file(file)
        elif path == "/api/catalog":
            self._json(200, catalog())
        elif path == "/api/port":
            try:
                port = int(parse_qs(query).get("port", [""])[0])
            except ValueError:
                port = 0
            if not 1 <= port <= 65535:
                self._json(400, {"error": "port must be a number from 1 to 65535"})
            elif port == self.server.server_port:
                self._json(200, {"port": port, "free": False, "reason": "this page"})
            else:
                self._json(200, port_status(port))
        elif path == "/api/state":
            try:
                since = int(parse_qs(query).get("since", ["0"])[0])
            except ValueError:
                since = 0
            self._json(200, self.server.session.snapshot(since))
        else:
            self._json(404, {"error": "not found"})

    do_HEAD = do_GET

    def do_POST(self):
        if not self._allowed():
            return
        path, _ = self._route()
        if path.startswith("/proxy/"):
            self._proxy()
            return
        body = self._body(MAX_SETUP_BYTES)
        if body is None:
            return
        if path == "/api/start":
            self._start(body)
        elif path == "/api/stop":
            self.server.session.stop()
            self._json(200, self.server.session.snapshot())
        elif path == "/api/voice/start":
            self._start_voice(body)
        elif path == "/api/voice/stop":
            self.server.voice.stop()
            self._json(200, self.server.voice.snapshot())
        else:
            self._json(404, {"error": "not found"})

    def do_DELETE(self):
        if self._allowed():
            if self._route()[0].startswith("/proxy/"):
                self._proxy()
            else:
                self._json(404, {"error": "not found"})

    def _start(self, body):
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "the setup must be sent as JSON"})
            return
        try:
            selection = session.Selection.from_request(json.loads(body or b"null"))
        except ValueError as error:
            self._json(400, {"error": str(error)})
            return
        if selection.port == self.server.server_port:
            self._json(
                400, {"error": f"port {selection.port} is this page's; choose another"}
            )
            return
        if self.server.voice.running:
            self._json(409, {"error": VOICE_RUNNING})
            return
        try:
            self.server.session.start(selection)
        except session.SelectionError as error:
            # An offline start of a model that is not here.
            self._json(400, {"error": str(error)})
            return
        except RuntimeError as error:
            self._json(409, {"error": str(error)})
            return
        except OSError as error:
            self._json(500, {"error": f"cannot start splash-omni serve: {error}"})
            return
        self._json(200, self.server.session.snapshot())

    def _start_voice(self, body):
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "the setup must be sent as JSON"})
            return
        try:
            chosen = voice.selections(json.loads(body or b"null"))
        except ValueError as error:
            self._json(400, {"error": str(error)})
            return
        main = self.server.session
        if main.selection is not None and main.phase in (
            *session.RUNNING,
            session.STOPPING,
        ):
            self._json(409, {"error": SPEECH_RUNNING})
            return
        taken = {self.server.server_port: "this page's port"}
        for role, selection in chosen.items():
            reason = taken.get(selection.port)
            if reason is None and not self.server.voice.running:
                status = port_status(selection.port)
                if not status["free"]:
                    reason = f"in use ({status['reason']})"
            if reason is not None:
                self._json(
                    400,
                    {
                        "error": f"port {selection.port} ({role}) is {reason}; choose another"
                    },
                )
                return
        try:
            self.server.voice.start(chosen)
        except session.SelectionError as error:
            self._json(400, {"error": str(error)})
            return
        except RuntimeError as error:
            self._json(409, {"error": str(error)})
            return
        except OSError as error:
            self.server.voice.stop()
            self._json(500, {"error": f"cannot start the voice chat servers: {error}"})
            return
        self._json(200, self.server.voice.snapshot())

    def _file(self, path):
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        self._send(200, path.read_bytes(), content_type)


class PageServer(ThreadingHTTPServer):
    daemon_threads = True
    # The launcher's own check (install/launcher.py) is for the server;
    # this binds without SO_REUSEADDR so a port in use is refused.
    allow_reuse_address = False

    def __init__(
        self,
        port,
        session_factory=session.ServerSession,
        voice_factory=voice.VoiceStack,
    ):
        super().__init__((HOST, port), Handler)
        self.session = session_factory(allowed_origins=page_origins(self.server_port))
        self.voice = voice_factory(allowed_origins=page_origins(self.server_port))


def bind(port=None):
    """The page's server on `port`, or on DEFAULT_PORT or the next free one
    after it when none is named."""
    if port is not None:
        return PageServer(port)
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + PORT_ATTEMPTS):
        try:
            return PageServer(candidate)
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
    raise OSError(
        errno.EADDRINUSE,
        f"ports {DEFAULT_PORT}-{DEFAULT_PORT + PORT_ATTEMPTS - 1} are all in use; "
        "choose one with --port",
    )


def _terminate(number, _frame):
    raise KeyboardInterrupt


def _watch_parent(parent):
    """SIGTERM this process once `parent`, the program that started it, has
    exited: the process is then a child of launchd. Its servers stop as on
    Ctrl+C, so a crashed desktop app leaves none behind."""
    while os.getppid() == parent:
        time.sleep(PARENT_POLL_SECONDS)
    os.kill(os.getpid(), signal.SIGTERM)


def main(port=None, open_browser=True, exit_with_parent=False):
    """Serve the page until Ctrl+C, then stop the server it started; with
    exit_with_parent, also once the program that started it exits."""
    server = bind(port)
    url = f"http://{HOST}:{server.server_port}/"
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    if exit_with_parent:
        threading.Thread(
            target=_watch_parent, args=(os.getppid(),), daemon=True
        ).start()
    print(f"splash-omni setup page: {url}", flush=True)
    print(
        "Choose a model there; the server starts from the page. "
        "Keep this terminal open; Ctrl+C stops the page and its server.",
        flush=True,
    )
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if (
            server.session.phase in (*session.RUNNING, session.STOPPING)
            or server.voice.running
        ):
            print("Stopping the server...", file=sys.stderr, flush=True)
        server.session.shutdown()
        server.voice.shutdown()
        server.server_close()
    return 0
