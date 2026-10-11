#!/usr/bin/env python3
"""Multi-model supervisor for Splash.

Manages a pool of single-model ``splash serve`` engine processes, forwarding
requests to the engine serving the requested model and restarting the engine
transparently when a request targets a different model.  A switch never
cuts a request in the flight: before stopping the running engine the
supervisor waits for requests already being forwarded to it to complete,
and requests that arrive while a switch is pending (for any configured
model, including the one still active) are queued and held until their
model is loaded (in arrival order).  A switch does not start until the
model it would stop has neither waiting nor in-flight requests, so an
older request (for example the one that triggered the initial load) is
always processed before a switch requested while it was waiting.  If
loading a model fails, the supervisor restores the previous model, answers
the requests holding for the failed model with a 503
``model_load_failed`` right away, and drops the model from the queue.
Requests that cannot be served within the hold budget get a 503 with a
``Retry-After`` header so that standard OpenAI/Anthropic SDKs retry them
without any application changes.

Before anything is proxied, every request is validated the same way the
single-model server validates it: the Host (and any Origin) header must name
an allowed host, and an API key (when one is configured) must be presented.
Rejections happen in ``parse_request``, i.e. before the body is read and
before a model switch can be triggered, so unauthenticated clients can never
make the supervisor cycle the engine.
"""

from __future__ import annotations

import argparse
import functools
import http.client
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from http import server as http_server
from pathlib import Path

if __package__:
    from . import paths
else:
    import paths

# The model ID syntax, credential checks and error payloads shared with the
# single-model server live in the server package next to install/; make the
# repo root importable for them, as `splash serve` does.
if str(paths.ROOT) not in sys.path:
    sys.path.insert(0, str(paths.ROOT))

from server import serve_options
from server.errors import ANTHROPIC_ERRORS, OPENAI_ERRORS, SYSTEMONE_ERRORS, APIError
from server.http_security import (
    authenticate,
    validate_api_key,
    validate_headers,
)

ROOT = paths.ROOT
# Sentinel for detecting unpassed keyword arguments.
_marker = object()


# ---------------------------------------------------------------------------
# configuration / constants
# ---------------------------------------------------------------------------
HOLD_TIMEOUT = 600  # seconds a request waits while its model loads
# Seconds to wait for an engine to answer /ready before killing and
# relaunching it.  0 means no cap: wait until it is ready or exits, so a
# first-time model download (many GB, possibly tens of minutes) is never
# cut short and restarted mid-transfer.
STARTUP_TIMEOUT = 0.0
READY_POLL_SECONDS = 0.5


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> list[dict]:
    """Load the multi-model JSON config from ``path``.

    Expected shape::

        {"models": [
            {"model": "OWNER/REPO", "aliases": ["alias1", "alias2"],
             "arguments": ["--max-context", "64000"]},
            ...
        ]}

    Every ``model`` entry should be a repo ID that ``model_artifacts`` can
    resolve; aliases are optional and must be valid served-model-name values.

    ``arguments`` (array of '--flag' strings, or one shell-like string)
    passes extra serve flags to that model's engine only.  When the same
    flag is also given after '--' on the command line, the command-line
    value wins for every model.

    ``preload`` (true/false) loads the model as soon as serve-multi
    starts, instead of waiting for the first request; when several models
    set it, the earliest one in the list wins.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read config {path}: {error}") from None
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"config {path} is not valid JSON: {error}") from None
    if not isinstance(document, dict):
        raise TypeError("config must be a JSON object with a 'models' list")
    unknown = sorted(set(document) - {"models"})
    if unknown:
        raise ValueError(
            f"unknown top-level keys in config: {unknown}; "
            "expected only {{'models': [...]}}"
        )
    raw = document.get("models")
    if not isinstance(raw, list):
        raise TypeError("config 'models' must be an array")
    out: list[dict] = []
    for index, entry in enumerate(raw):
        where = f"models[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(
                f"{where} must be an object like "
                '{{"model": "OWNER/REPO", "aliases": [...], "max_context": ...}}'
            )
        unknown = sorted(
            set(entry)
            - {"model", "aliases", "max_context", "arguments", "preload"}
        )
        if unknown:
            raise ValueError(f"{where}: unknown keys {unknown}")
        model = entry.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{where}: 'model' must be a non-empty string")
        try:
            serve_options.parse_model_id(model)
        except ValueError:
            raise ValueError(f"{where}: invalid model repo id '{model}'") from None
        aliases = entry.get("aliases")
        if aliases is None:
            aliases = ()
        elif not isinstance(aliases, list):
            raise ValueError(f"{where}: 'aliases' must be an array")
        else:
            cleaned: list[str] = []
            for alias in aliases:
                if not isinstance(alias, str) or not alias:
                    raise ValueError(f"{where}: alias must be a non-empty string")
                cleaned.append(alias)
            aliases = tuple(cleaned)
        max_context = entry.get("max_context")
        if max_context is not None and not isinstance(max_context, int):
            raise ValueError(f"{where}: 'max_context' must be an integer or null")
        arguments = entry.get("arguments")
        if arguments is not None:
            if isinstance(arguments, str):
                arguments = shlex.split(arguments)
            if not isinstance(arguments, list) or not all(
                isinstance(token, str) for token in arguments
            ):
                raise ValueError(
                    f"{where}: 'arguments' must be an array of '--flag' strings "
                    "or a single string"
                )
            try:
                _merge_server_arguments(arguments)
            except ValueError as error:
                raise ValueError(f"{where} arguments: {error}") from None
        else:
            arguments = []
        preload = entry.get("preload", False)
        if not isinstance(preload, bool):
            raise TypeError(f"{where}: 'preload' must be true or false")
        out.append(
            {
                "model": model,
                "aliases": aliases,
                "max_context": max_context,
                "arguments": arguments,
                "preload": preload,
            }
        )
    if not out:
        raise ValueError("config 'models' must contain at least one model")
    return out


# ---------------------------------------------------------------------------
# supervisor
# ---------------------------------------------------------------------------


class Supervisor:
    """Manages one ``splash serve`` engine process per model in a pool.

    Only one child process runs at a time; the ``ServerForwarder`` routes
    incoming requests to it and triggers a restart whenever a request names
    a different model than the one currently loaded.  Every engine
    lifecycle change — a model switch, a crash restart, a failed-switch
    restore — is claimed before the old engine is touched: the claim waits
    for requests already in flight on the current engine to complete
    (``_wait_for_in_flight``) and, while held, makes requests for any other
    model queue behind it (``_queued_targets``) and requests for the current
    model hold until it is ready again.  Nothing is ever cut in the flight,
    and engines are never (re)launched out of request order.
    """

    def __init__(
        self,
        model_specs: list[dict],
        shared: dict,
        host: str,
        port: int,
    ) -> None:
        self.model_specs = model_specs
        self.shared = shared
        self.passthrough = list(shared.get("passthrough", []))
        self.host = host
        self.port = port
        raw_hold = shared.get("switch_timeout", HOLD_TIMEOUT)
        self.hold_timeout = float(raw_hold) if raw_hold is not None else HOLD_TIMEOUT
        raw_startup = shared.get("startup_timeout", STARTUP_TIMEOUT)
        self.startup_timeout = (
            float(raw_startup) if raw_startup is not None else STARTUP_TIMEOUT
        )
        # Pre-proxy security: mirror server.py's own checks so requests are
        # rejected before they reach the engine (and before a request's model
        # field can trigger a switch).
        self.api_key = shared.get("api_key")
        self.allowed_hosts = {
            host.lower().rstrip(".")
            for host in (
                *shared.get("allowed_host", ()),
                self.host,
                "localhost",
                "127.0.0.1",
                "::1",
            )
            if host not in ("0.0.0.0", "::")
        }
        # As serve_options.parse_allowed_origin returns them.
        self.allowed_origins = frozenset(shared.get("allowed_origin", ()))

        self._lock = threading.RLock()
        # Serializes whole stop/launch lifecycles so the switch,
        # reaper, and watcher threads can never launch engines in parallel.
        self._op_lock = threading.RLock()
        # Requests currently being forwarded to the running engine.  A
        # pending switch refuses new forwards (``begin_forward``) and waits
        # for this count to reach zero (``_wait_for_in_flight``) before
        # stopping the engine, so it never cuts a request in the flight.
        self._in_flight = 0
        self._in_flight_cv = threading.Condition(self._lock)
        self._child: subprocess.Popen | None = None
        self._child_port: int | None = None
        self._child_ready = False
        self._switching = False
        self._switch_target: str | None = None
        self._switch_done = threading.Event()
        self._switch_done.set()  # set == "no switch in progress"
        # Models requested while a switch is in flight, in arrival order;
        # _drain_queue starts them one at a time as each switch finishes.
        self._queued_targets: list[str] = []
        # Models whose most recent switch failed to load.  Requests holding
        # for one of these get a 503 model_load_failed right away instead of
        # waiting out the hold budget, and the model is dropped from the
        # queue so it is not retried automatically.  A new switch to the
        # model clears the flag (client-driven retry).
        self._failed_switches: set[str] = set()
        # Requests currently held (waiting) for each canonical model to
        # become active.  A switch to the next queued model does not start
        # until the model it would stop has neither waiting nor in-flight
        # requests, so an older request is never starved by a newer model's
        # switch (e.g. the first request that triggered the initial load is
        # always processed before a switch requested while it waited).
        self._waiting: dict[str, int] = {}
        self._crash_times: list[float] = []
        self._crash_loop = False  # 3 crashes in 60s; auto-restart parked
        self._last_failed_child: subprocess.Popen | None = None
        self._relaunching: str | None = None  # set while a (re)launch runs
        self._stop = threading.Event()
        self._proxy_server: http_server.BaseServer | None = None
        self._active_model: str | None = None

    # -- public interface ---------------------------------------------------

    @property
    def active_model(self) -> str | None:
        with self._lock:
            return self._active_model

    @property
    def child_port(self) -> int | None:
        with self._lock:
            return self._child_port

    @property
    def child_ready(self) -> bool:
        with self._lock:
            return self._child_ready

    @property
    def switching(self) -> bool:
        with self._lock:
            return self._switching

    @property
    def switch_target(self) -> str | None:
        with self._lock:
            # A crash restart has no switch target; report the model it is
            # (re)loading so /status and errors can name it.
            return self._switch_target or self._relaunching

    @property
    def queued_models(self) -> list[str]:
        with self._lock:
            return list(self._queued_targets)

    @property
    def in_flight(self) -> int:
        """Requests currently being forwarded to the running engine."""
        with self._lock:
            return self._in_flight

    def switch_failed(self, model: str) -> bool:
        """True if the most recent switch to *model* (or an alias) failed to
        load and has not since been retried — a new switch to the model
        clears the flag."""
        canonical = self._resolve_model(model)
        with self._lock:
            return canonical in self._failed_switches

    def begin_forward(self, held_model: str | None = None) -> int | None:
        """Register a request about to be forwarded to the running engine.

        Returns the engine's port, or ``None`` when the request must not
        reach the engine: either no engine is up or a switch is pending.
        The check and the counter increment are one critical section, so
        the drain (``_wait_for_in_flight``) can never miss a forward — a
        forward either counts against the drain or is refused.

        If *held_model* is given, the request is finishing a hold for that
        model: its wait count is swapped for a forward count in the same
        critical section, so a switch cannot slip in between the hold
        ending and the forward starting.
        """
        with self._lock:
            if self._stop.is_set() or self._switching:
                return None
            if self._child_port is None:
                return None
            if held_model is not None:
                self._decrement_waiting(self._resolve_model(held_model))
            self._in_flight += 1
            self._in_flight_cv.notify_all()
            return self._child_port

    def end_forward(self) -> None:
        """Release a forward registered with :meth:`begin_forward`."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._in_flight_cv.notify_all()

    def begin_hold(self, model: str) -> None:
        """Track a request that is now waiting (held) for *model* (or an
        alias) to become active.  Released by :meth:`end_hold`, or by the
        hold→forward handoff in :meth:`begin_forward` when the model is
        ready."""
        canonical = self._resolve_model(model)
        with self._lock:
            self._waiting[canonical] = self._waiting.get(canonical, 0) + 1
            self._in_flight_cv.notify_all()

    def end_hold(self, model: str) -> None:
        """Release :meth:`begin_hold` for a held request that ends without
        forwarding (timed out, its model's load failed, or an error)."""
        with self._lock:
            self._decrement_waiting(self._resolve_model(model))

    def _decrement_waiting(self, canonical: str) -> None:
        """Drop one waiting count; caller holds the lock."""
        count = self._waiting.get(canonical, 0) - 1
        if count <= 0:
            self._waiting.pop(canonical, None)
        else:
            self._waiting[canonical] = count
        self._in_flight_cv.notify_all()

    def waiting_counts(self) -> dict[str, int]:
        """Requests currently held per canonical model (for /status)."""
        with self._lock:
            return dict(self._waiting)

    def _wait_for_model_free(self, model: str) -> None:
        """Block until no request is waiting for *model* or being forwarded
        to the running engine.  Called before the engine is stopped for the
        next queued model, so every request older than that switch is
        processed first."""
        with self._in_flight_cv:
            self._in_flight_cv.wait_for(
                lambda: self._in_flight == 0
                and self._waiting.get(model, 0) == 0
            )

    def _wait_for_in_flight(self) -> None:
        """Block until no request is being forwarded to the running engine.

        Called (holding ``_op_lock``) before a switch stops the engine, so
        a request already in the flight always runs to completion instead
        of being cut by the engine restart.  New forwards are refused while
        a switch is pending, so the count can only go down here.  Bounded
        by the proxy's own upstream socket timeout, not by the client.
        """
        with self._in_flight_cv:
            count = self._in_flight
            if count == 0 or self._stop.is_set():
                return
            print(
                f"serve-multi · waiting for {count} in-flight request(s) "
                "to complete before switching",
                flush=True,
            )
            waited = 0.0
            while self._in_flight > 0 and not self._stop.is_set():
                self._in_flight_cv.wait(5.0)
                waited += 5.0
                if self._in_flight > 0 and not self._stop.is_set():
                    print(
                        f"serve-multi · still waiting for {self._in_flight} "
                        f"in-flight request(s) ({waited:.0f}s)",
                        flush=True,
                    )

    def _claim_restart(self, model: str) -> bool:
        """Claim the engine lifecycle for a crash restart.

        Sets the switching state *before* the caller takes ``_op_lock`` so
        that for the whole restart — including the time spent queued for
        the lifecycle lock — a request for another model is queued behind
        the restart (``switch_to``) instead of killing the model that is
        (re)loading, and new forwards are refused (``begin_forward``) while
        the engine is down, so a request for this model is held until it is
        ready again instead of hitting a dead port.  Returns False when a
        switch already owns the lifecycle; that switch relaunches its own
        model and the crashed model comes back on its next request.
        """
        with self._lock:
            if self._stop.is_set() or self._switching:
                return False
            self._switching = True
            self._relaunching = model
        return True

    def _release_restart(self) -> None:
        """Release a restart claimed with :meth:`_claim_restart` and start
        the next queued switch, if any — its requests are served in arrival
        order, after the restarted model is ready."""
        with self._lock:
            self._switching = False
            self._relaunching = None
            self._switch_done.set()
        self._drain_queue()

    def switch_to(self, model: str) -> None:
        """Request a switch to *model*.  Non-blocking; the switch runs in
        a background thread that first waits for requests already in flight
        on the current engine to complete before stopping it.  If a switch
        is already in flight, a model that differs from the in-flight
        target is queued and started as soon as the current switch finishes
        (``_drain_queue``), so requests holding for it are served in
        arrival order instead of being dropped."""
        canonical = self._resolve_model(model)
        with self._lock:
            if self._active_model == canonical:
                return
            if self._switching:
                if (
                    canonical != self._switch_target
                    and canonical not in self._queued_targets
                ):
                    self._queued_targets.append(canonical)
                    print(
                        f"serve-multi · queued switch to {model} "
                        f"(after {self._switch_target or self._relaunching})",
                        flush=True,
                    )
                return
            self._switching = True
            self._switch_target = canonical
            self._switch_done.clear()
            self._failed_switches.discard(canonical)
        threading.Thread(
            target=self._do_switch,
            args=(model,),
            daemon=True,
            name="serve-multi-switch",
        ).start()

    def _drain_queue(self) -> None:
        """Start the next queued switch, if any, after the current switch or
        load attempt has finished.  Called at the end of every switch/load
        attempt (success or failure) so requests held for a later model are
        served in arrival order.

        Before starting, waits for the model that is now active to be free
        of waiting and in-flight requests — otherwise the switch would
        starve requests that arrived before the queued one (for example the
        first request that triggered the initial load, still waiting for
        its model when a second request queued a switch during the load).
        """
        with self._lock:
            if self._stop.is_set() or self._switching or not self._queued_targets:
                return
        active = self.active_model
        if active is not None:
            self._wait_for_model_free(active)
        with self._lock:
            if self._stop.is_set() or self._switching:
                return
            model = None
            while self._queued_targets:
                candidate = self._queued_targets.pop(0)
                if candidate != self._active_model:
                    model = candidate  # skip entries that are now current
                    break
            if model is None:
                return
            self._switching = True
            self._switch_target = model
            self._switch_done.clear()
            self._failed_switches.discard(model)
        print(f"serve-multi · starting queued switch to {model}", flush=True)
        threading.Thread(
            target=self._do_switch,
            args=(model,),
            daemon=True,
            name="serve-multi-switch",
        ).start()

    def _load_initial_model(self, model: str) -> bool:
        """Load a config model marked 'preload' at startup.  Returns True
        if loading was initiated, False if something is already running,
        already switching, or shutting down."""
        with self._lock:
            if self._active_model is not None:
                return True  # already running
            if self._switching:
                return False  # another load/switch in progress
            self._switching = True
            self._switch_target = model
            self._switch_done.clear()
            self._failed_switches.discard(model)
        print(f"serve-multi · loading preloaded model: {model}", flush=True)
        threading.Thread(
            target=self._do_load_first,
            args=(model,),
            daemon=True,
            name="serve-multi-load-first",
        ).start()
        return True

    def _do_load_first(self, model: str) -> None:
        """Load the initial model at startup.  Runs in a background thread."""
        try:
            with self._op_lock:
                self._wait_for_in_flight()
                self._stop_child()
                self._launch_child(model, wait_ready=True)
            if self._stop.is_set():
                return
            with self._lock:
                self._switching = False
                self._switch_target = None
                self._switch_done.set()
            self._drain_queue()
            print(f"serve-multi · now serving {model}", flush=True)
        except Exception as error:  # noqa: BLE001 — deliberate: contain any engine failure

            with self._lock:
                self._switching = False
                self._switch_target = None
                self._switch_done.set()
                self._failed_switches.add(model)
            self._drain_queue()
            print(
                f"serve-multi · first-model load failed: {error}",
                file=sys.stderr,
                flush=True,
            )

    def shutdown(self) -> None:
        self._stop.set()
        if self._proxy_server is not None:
            try:
                self._proxy_server.server_close()
            except OSError:
                pass
            self._proxy_server = None
        self._stop_child()

    # -- child process ------------------------------------------------------

    def _child_command(self, model: str, spec: dict, port: int) -> list[str]:
        # The engine is the serve command itself: it does its own install
        # (build lock, device check, model prepare), takes the serve locks
        # and the assembly hold, and execs the server.  serve-multi only
        # picks the model, the internal loopback port, and the per-model
        # flags.
        command = [
            str(paths.PYTHON),
            "-u",
            str(ROOT / "install" / "launcher.py"),
            "serve",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        # Precedence, flag by flag: serve flags after '--' > per-model
        # 'arguments' > the legacy 'max_context' key.
        model_arguments: list[str] = []
        if spec.get("max_context") is not None:
            model_arguments.extend(["--max-context", str(spec["max_context"])])
        model_arguments.extend(spec.get("arguments") or [])
        merged = _merge_server_arguments(model_arguments, self.passthrough)
        for name in spec.get("aliases") or ():
            command.append(f"--served-model-name={name}")
        command.extend(_engine_arguments(merged))
        return command

    def _child_environment(self) -> dict:
        # The serve child sets its own run flags; the key travels through
        # the SPLASH_API_KEY environment variable it reads, not the command
        # line.
        environment = dict(os.environ)
        if self.shared.get("api_key") is not None:
            environment["SPLASH_API_KEY"] = self.shared["api_key"]
        return environment

    def _launch_child(self, model: str, *, wait_ready: bool = True) -> None:
        spec = self.model_specs_by_model(model)
        # Defense in depth: never start a second engine while one is alive.
        self._stop_child()
        environment = self._child_environment()
        last_error: Exception | None = None
        for _attempt in range(3):
            if self._stop.is_set():
                return
            port = _pick_free_port("127.0.0.1")
            child = subprocess.Popen(
                self._child_command(model, spec, port),
                stdout=subprocess.PIPE,
                stderr=None,
                env=environment,
                cwd=str(ROOT),
            )
            threading.Thread(
                target=self._relay_output, args=(child,), daemon=True
            ).start()
            with self._lock:
                self._child = child
                self._child_port = port
            threading.Thread(
                target=self._watch_child,
                args=(child,),
                daemon=True,
                name="serve-multi-child-watch",
            ).start()
            ready = self._wait_ready(child, port) if wait_ready else False
            if ready:
                with self._lock:
                    self._child_ready = True
                    self._crash_loop = False
                self._last_failed_child = None
                self._active_model = model
                print(f"serve-multi · serving {model}", flush=True)
                return
            try:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            except OSError:
                pass
            with self._lock:
                self._child = None
                self._child_port = None
                self._child_ready = False
            last_error = RuntimeError(
                f"{model} did not become ready on attempt {_attempt + 1}"
            )
            time.sleep(0.5)
        if last_error is not None:
            raise RuntimeError(
                f"{model} failed to start after 3 attempts: {last_error}"
            )

    def _wait_ready(self, child: subprocess.Popen, port: int) -> bool:
        """Wait until the engine answers /ready, exits, or the startup
        budget is spent.  The budget is uncapped by default (0), because a
        first-time model download happens inside this window and can run
        far longer than any fixed engine-startup allowance."""
        deadline = (
            time.monotonic() + self.startup_timeout
            if self.startup_timeout > 0
            else None
        )
        while True:
            if self._stop.is_set():
                return False
            if child.poll() is not None:
                return False
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                conn.request("GET", "/ready")
                resp = conn.getresponse()
                conn.close()
                if resp.status == 200:
                    return True
                if resp.status != 503:
                    return False
            except OSError:
                pass
            if deadline is not None and time.monotonic() >= deadline:
                print(
                    f"serve-multi · engine did not become ready within "
                    f"{self.startup_timeout:.0f}s; restarting it",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            time.sleep(READY_POLL_SECONDS)

    def _stop_child(self) -> None:
        with self._lock:
            child = self._child
            self._child = None
            self._child_ready = False
        if child is None or child.poll() is not None:
            return
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def _relay_output(self, child: subprocess.Popen) -> None:
        assert child.stdout is not None
        tail: deque[bytes] = deque()
        tail_bytes = 0
        # Shared with _output_tail so a crash report can quote the last lines.
        child._splash_output_tail = tail
        try:
            # read1, not readline: model-download progress (tqdm) repaints
            # with carriage returns and no newlines, so relaying line by line
            # would hold it back until the whole download finished.
            while not self._stop.is_set():
                chunk = child.stdout.read1(65536)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                tail.append(chunk)
                tail_bytes += len(chunk)
                while tail_bytes > 32768 and len(tail) > 1:
                    tail_bytes -= len(tail.popleft())
        except (OSError, ValueError):
            pass  # the pipe closed; the watcher is the authoritative signal

    def _watch_child(self, child: subprocess.Popen) -> None:
        """Run until the child exits; restart on unexpected death."""
        while True:
            model = self.active_model
            if model is None:
                return
            child.wait()
            if self._stop.is_set():
                return
            with self._lock:
                if self._child is not child:
                    return  # already replaced
            if child.poll() == 0:
                return  # clean exit (shouldn't normally happen)
            # Unexpected crash — restart after settling.  The restart claims
            # the lifecycle (like a switch) before taking _op_lock, so
            # requests queue behind the (re)load instead of killing it; it
            # is re-checked under the lock in case a racing restart already
            # revived the engine.
            if tail := self._output_tail(child):
                print(tail, flush=True)
            try:
                if not self._claim_restart(model):
                    return  # a switch owns the lifecycle; it relaunches
                try:
                    with self._op_lock:
                        with self._lock:
                            if (
                                self._child is not None
                                and self._child.poll() is None
                            ):
                                return  # someone already revived the engine
                        self._launch_child(model, wait_ready=True)
                finally:
                    self._release_restart()
            except (RuntimeError, OSError) as error:
                print(
                    f"serve-multi · restart of {model} failed: {error}",
                    flush=True,
                )
                # Park for a bit before retrying so we don't thrash the OS.
                time.sleep(5)
                continue

    def _output_tail(self, child: subprocess.Popen) -> str | None:
        tail = getattr(child, "_splash_output_tail", None)
        if not tail:
            return None
        try:
            return b"".join(tail).decode("utf-8", errors="replace")
        except (OSError, ValueError):
            return None

    def _reap_loop(self) -> None:
        """Periodically check whether the engine died and relaunch if so."""
        while not self._stop.is_set():
            time.sleep(2)
            self._reap_once()

    def _reap_once(self) -> None:
        with self._lock:
            child = self._child
            model = self._active_model
            if model is None or self._switching:
                return  # a switch/restart owns the lifecycle
            if child is None or child.poll() is not None:
                self._crash_times = [
                    t for t in self._crash_times if time.monotonic() - t < 60
                ]
                self._crash_times.append(time.monotonic())
                if len(self._crash_times) >= 3:
                    self._crash_loop = True
                if self._crash_loop:
                    return
                print(
                    f"serve-multi · no engine serving; (re)launching {model}",
                    flush=True,
                )
        try:
            if not self._claim_restart(model):
                return  # a switch owns the lifecycle; it will (re)launch
            try:
                with self._op_lock:
                    # Re-check under the lifecycle lock: a racing restart
                    # may have revived the engine while we queued.
                    with self._lock:
                        if self._child is not None and self._child.poll() is None:
                            return
                    self._launch_child(model, wait_ready=True)
            finally:
                self._release_restart()
        except (RuntimeError, OSError) as error:
            print(
                f"serve-multi · (re)launch of {model} failed: {error}; will retry",
                flush=True,
            )

    def _do_switch(self, model: str) -> None:
        """Switch the active model to *model*.  Runs in a background thread."""
        try:
            canonical = self._resolve_model(model)
            if canonical not in self._model_names():
                raise ValueError(f"unknown model: {model}")
            with self._op_lock:
                self._wait_for_in_flight()
                self._stop_child()
                self._launch_child(canonical, wait_ready=True)
            if self._stop.is_set():
                return  # shutting down; do not commit an unready model
            with self._lock:
                self._switching = False
                self._switch_target = None
                self._switch_done.set()
            self._drain_queue()
            print(f"serve-multi · switched to {model}", flush=True)
        except Exception as error:  # noqa: BLE001 — deliberate: fall back to previous model

            # Try to restore the previous model before giving up (unless the
            # request was rejected before the current engine was touched).
            # The switching state is kept for the whole restore — new
            # requests queue behind it and forwards are refused, instead of
            # racing the relaunch — and is released only once the restore is
            # done, after which the next queued switch starts.
            with self._lock:
                previous = self._active_model
            restore = (
                not isinstance(error, ValueError) and previous and previous != model
            )
            if restore:
                print(
                    f"serve-multi · switch to {model} failed: {error}; "
                    f"restoring {previous}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    with self._op_lock:
                        self._launch_child(previous, wait_ready=True)
                except (RuntimeError, OSError) as restore_error:
                    print(
                        f"serve-multi · restore of {previous} also failed: "
                        f"{restore_error}",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                print(
                    f"serve-multi · switch to {model} failed: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            with self._lock:
                self._switching = False
                self._switch_target = None
                self._switch_done.set()
                self._failed_switches.add(canonical)
                # Do not retry a model that just failed to load: drop it
                # from the queue so the supervisor moves on to the next
                # queued model, and let held requests for it fail fast
                # (switch_failed) instead of waiting out the hold budget.
                self._queued_targets = [
                    m for m in self._queued_targets if m != canonical
                ]
            self._drain_queue()

    def _resolve_model(self, model: str) -> str:
        """Resolve an alias or full model ID to the canonical model ID."""
        for spec in self.model_specs:
            if spec["model"] == model:
                return model
            if model in (spec.get("aliases") or ()):  # type: ignore[arg-type]
                return spec["model"]
        return model  # pass through; caller should reject if unknown

    def model_specs_by_model(self, model: str | None = None) -> dict:
        specs = {spec["model"]: spec for spec in self.model_specs}
        if model is None:
            return next(iter(specs.values())) if specs else {}
        canonical = self._resolve_model(model)
        return specs.get(canonical, {})

    def _model_names(self) -> set[str]:
        names = {spec["model"] for spec in self.model_specs}
        for spec in self.model_specs:
            names.update(spec.get("aliases") or ())  # type: ignore[arg-type]
        return names


# ---------------------------------------------------------------------------
# HTTP proxy
# ---------------------------------------------------------------------------


# Hop-by-hop / connection-specific headers the proxy owns and never forwards.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "expect",
        "te",
        "trailers",
        "upgrade",
    }
)
# Response headers rewritten or dropped by the proxy when copying upstream.
_SKIP_RESPONSE_HEADERS = _HOP_BY_HOP_HEADERS | {"content-encoding", "content-language"}


class _ForwardingHTTPServer(http_server.ThreadingHTTPServer):
    """HTTP server that accepts connections and proxies them to the active
    ``splash serve`` engine."""

    def __init__(self, *args, supervisor: Supervisor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.supervisor = supervisor


class _ServeMultiHandler(http_server.BaseHTTPRequestHandler):
    """Proxies requests to the active single-model server and triggers
    model switches when the ``model`` field in chat/completions requests
    names a different model."""

    supervisor: Supervisor

    def __init__(self, request, client_address, server, *, supervisor=_marker):
        # Allow supervisor to be set as a class attribute (e.g. in tests)
        # or passed as a keyword argument (e.g. via functools.partial).
        if supervisor is _marker:
            self.supervisor = getattr(type(self), "supervisor", None)
        else:
            self.supervisor = supervisor
        super().__init__(request, client_address, server)

    def log_message(self, format, *args):
        # Suppress default access logs to keep stdout clean.
        pass

    def parse_request(self):
        """Validate Host/Origin and API credentials before any request is
        proxied or can trigger a model switch.  Mirrors the checks in
        ``server/server.py``; a rejection sends the same error shape the
        single-model server would and closes the connection."""
        if not super().parse_request():
            return False
        try:
            allowed_hosts = self.supervisor.allowed_hosts | {
                self.connection.getsockname()[0].lower()
            }
            validate_headers(
                self.headers, allowed_hosts, self.supervisor.allowed_origins
            )
            path = self.path.partition("?")[0]
            public = self.command == "OPTIONS" or (
                self.command in ("GET", "HEAD")
                and path in ("/", "/index.html", "/health", "/ready")
            )
            if not public:
                authenticate(self.headers, self.supervisor.api_key)
        except APIError as error:
            self.close_connection = True
            print(
                f"serve-multi · rejected {error.status} "
                f"{self.command} {self.path.partition('?')[0]}: {error.message}",
                file=sys.stderr,
                flush=True,
            )
            self._send_auth_error(error)
            return False
        return True

    # -- routing ------------------------------------------------------------

    def do_GET(self):
        if self.path == "/ready":
            if self.supervisor.switching or not self.supervisor.child_ready:
                self._json(503, {"status": "switching"})
                return
            self._forward("GET", self.path, None)
        elif self.path == "/status":
            if self.supervisor.switching or not self.supervisor.child_ready:
                self._json(
                    503,
                    {
                        "status": "switching",
                        "active_model": self.supervisor.active_model,
                        "switch_target": self.supervisor.switch_target,
                        "queued_models": self.supervisor.queued_models,
                        "in_flight": self.supervisor.in_flight,
                        "waiting": self.supervisor.waiting_counts(),
                    },
                )
                return
            self._forward("GET", self.path, None)
        elif self.path == "/metrics":
            if not self.supervisor.child_ready:
                self._send(503, "service unavailable\n", "text/plain")
                return
            self._forward("GET", self.path, None)
        elif self.path == "/v1/models" or self.path.startswith("/v1/models/"):
            self._handle_models()
        else:
            self._forward("GET", self.path, None)

    def do_POST(self):
        # Read the body so we can inspect the model field before deciding
        # whether to switch.
        content_length = int(self.headers.get("Content-Length", 0))
        body: bytes | None = None
        if content_length > 0:
            body = self.rfile.read(content_length)

        # Only chat/completions, completions, and responses carry a modifiable
        # ``model`` field that we need to inspect.
        if body and self.path in (
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
        ):
            switch = self._maybe_switch(body)
            if switch is not None:
                return  # already sent a 503

        # Forward (with the original body for POST; GET never has a body).
        self._forward("POST", self.path, body)

    def do_DELETE(self):
        self._forward("DELETE", self.path, None)

    def do_OPTIONS(self):
        self._forward("OPTIONS", self.path, None)

    # -- model inspection ---------------------------------------------------

    def _maybe_switch(self, body: bytes) -> None | int:
        """Inspect the request body for a ``model`` field.  If it differs
        from the currently active model, trigger a switch and hold the
        connection until the switch completes.
        Returns the status code sent (503) or None if no switch was needed."""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON — forward as-is (e.g. binary uploads, malformed
            # requests that the backend will reject).
            return None
        if not isinstance(data, dict):
            return None
        req_model = data.get("model")
        if not isinstance(req_model, str) or not req_model:
            return None
        active = self.supervisor.active_model
        serving = bool(active) and (
            req_model == active or req_model in self._active_aliases()
        )
        # The active model is about to be replaced when a switch is pending,
        # so a matching request joins the queue like any other (held until
        # its model is the one being served again) instead of racing the
        # engine that is about to stop.
        if serving and not self.supervisor.switching:
            return None  # already serving this model, nothing in flight
        if req_model not in self.supervisor._model_names():
            # Not a model this deployment serves; a switch would just fail
            # after cycling the engine, so answer immediately.
            self._send_error(
                503,
                f"unknown model: {req_model} (not in the serve-multi config)",
                "model_not_found",
                code="model_not_found",
            )
            return 503
        # Model mismatch (or the active model is being switched away) —
        # start a switch, or queue behind the one already in flight, then
        # hold the connection until this model is served.
        self.supervisor.switch_to(req_model)
        return self._hold_until_switched(body, req_model)

    def _hold_until_switched(self, body: bytes, req_model: str) -> int:
        """Hold the connection (silently) until *req_model* is the active
        model, then forward the request.  Returns 503 if that does not
        happen within the hold budget (including queued switches).

        The request is tracked as *waiting for* its model from here until
        it is forwarded (atomically), so the supervisor will not start a
        later switch before this request has been processed — a queued
        model switch never starves an older request.
        """
        # Hold the connection silently until the switch completes.  Nothing
        # may be written before the status line, so no keep-alive pings are
        # sent here; a client that gives up early simply retries and finds
        # the new model ready.  (Writing SSE pings before send_response()
        # would put "body" bytes ahead of the HTTP status line and corrupt
        # the response.)
        self.supervisor.begin_hold(req_model)
        handed_off = False
        try:
            switch_timeout = self.supervisor.hold_timeout
            deadline = time.monotonic() + switch_timeout

            while time.monotonic() < deadline:
                if self.supervisor.switch_failed(req_model):
                    break  # the switch to this model failed; stop holding
                if not self.supervisor.switching and self.supervisor.child_ready:
                    active = self.supervisor.active_model
                    if active and (
                        req_model == active
                        or req_model in self._active_aliases()
                    ):
                        break  # this request's model is the one being served
                time.sleep(0.1)

            # The switch to this model failed to load — answer now instead
            # of holding out the budget; the previous model was restored.
            if self.supervisor.switch_failed(req_model):
                self.supervisor.end_hold(req_model)
                self._send_error(
                    503,
                    f"loading model {req_model} failed; see the server log",
                    "server_error",
                    code="model_load_failed",
                    retry_after="2",
                )
                return 503

            # If the model never became active in time, return 503.
            active = self.supervisor.active_model
            if (
                self.supervisor.switching
                or not self.supervisor.child_ready
                or not active
                or (
                    req_model != active
                    and req_model not in self._active_aliases()
                )
            ):
                self.supervisor.end_hold(req_model)
                target = self.supervisor.switch_target
                self._send_error(
                    503,
                    f"model {req_model} was not ready within "
                    f"{switch_timeout:.0f}s"
                    + (f" (switching to {target})" if target else ""),
                    "model_switching",
                    code="model_switching",
                    retry_after="2",
                )
                return 503

            # Switch complete — forward the original request, handing the
            # wait count over to the forward atomically.
            handed_off = True
            self._forward("POST", self.path, body, held_model=req_model)
            return 200
        except Exception as error:  # noqa: BLE001 — deliberate: release the held request

            if not handed_off:
                self.supervisor.end_hold(req_model)
            print(
                f"serve-multi · unexpected error while holding a request: "
                f"{error}",
                file=sys.stderr,
                flush=True,
            )
            self._send_error(
                503, "internal supervisor error", "server_error"
            )
            return 503

    def _active_aliases(self) -> frozenset[str]:
        active = self.supervisor.active_model
        if not active:
            return frozenset()
        spec = self.supervisor.model_specs_by_model(active)
        return frozenset(spec.get("aliases") or ())

    # -- /v1/models synthesis ----------------------------------------------

    def _handle_models(self):
        """Synthesize the /v1/models response from the current model's
        specification when no engine is ready, otherwise forward."""
        ready = self.supervisor.child_ready
        if not ready:
            self._json(
                503,
                {
                    "object": "list",
                    "data": [],
                    "models": [],
                },
            )
            return
        self._forward("GET", self.path, None)

    # -- forwarding ---------------------------------------------------------

    def _forward_headers(self, upstream_port: int | None = None) -> dict:
        """The client's headers, minus the hop-by-hop headers the proxy owns.

        ``Origin`` is rewritten to the upstream's loopback authority: the
        proxy already validated the client's Host/Origin pair, and upstream
        it presents ``Host: 127.0.0.1:<port>`` (http.client sets it), so a
        user-facing Origin would trip the child's own cross-origin check.
        """
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS
        }
        if upstream_port is not None:
            for key in headers:
                if key.lower() == "origin":
                    headers[key] = f"http://127.0.0.1:{upstream_port}"
        return headers

    def _is_streaming_response(self, resp_headers: dict[str, str]) -> bool:
        """Check if the upstream response is a streaming (SSE) response."""
        content_type = self._header_value(resp_headers, "content-type")
        return "text/event-stream" in content_type.lower()

    def _header_value(self, headers: dict[str, str], name: str) -> str:
        """Case-insensitive header lookup."""
        name_lower = name.lower()
        for key, value in headers.items():
            if key.lower() == name_lower:
                return value
        return ""

    def _write_upstream_response(self, resp: http.client.HTTPResponse) -> None:
        """Copy an upstream response to the client, streaming SSE bodies."""
        resp_headers = dict(resp.getheaders())
        self.send_response(resp.status)
        for key, value in resp_headers.items():
            if key.lower() in _SKIP_RESPONSE_HEADERS:
                continue
            self.send_header(key, value)
        if self._is_streaming_response(resp_headers):
            # Stream chunk-by-chunk so tokens arrive in real time.  read1()
            # issues at most one upstream read per call; read() would block
            # until 64 KiB or EOF and defeat streaming.  Keep conn open:
            # resp reads from it directly.
            self.end_headers()
            self.wfile.flush()
            try:
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, BrokenPipeError):
                pass
        else:
            resp_body = resp.read()
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            self.wfile.flush()

    def _forward(
        self,
        method: str,
        path: str,
        body: bytes | None,
        held_model: str | None = None,
    ) -> None:
        port = self.supervisor.begin_forward(held_model)
        if port is None:
            if held_model is not None:
                # The hold→forward handoff was refused (a switch got in
                # first): the request's wait count must still be released.
                self.supervisor.end_hold(held_model)
            if self.supervisor.switching:
                # A switch is pending and this request may not join the
                # engine it is about to stop; standard SDKs retry on the
                # Retry-After hint.
                self._send_error(
                    503,
                    "model switch in progress; retry shortly",
                    "model_switching",
                    code="model_switching",
                    retry_after="2",
                )
            else:
                self._send_error(503, "no engine running", "server_error")
            return
        try:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port)
                conn.request(method, path, body, self._forward_headers(port))
                try:
                    self._write_upstream_response(conn.getresponse())
                except (OSError, BrokenPipeError):
                    # Client disconnected during forwarding; nothing we can do.
                    pass
                finally:
                    conn.close()
            except (OSError, http.client.HTTPException) as error:
                self._send_error(502, f"upstream error: {error}", "upstream_error")
        finally:
            self.supervisor.end_forward()

    # -- response helpers ---------------------------------------------------

    def _send_error(
        self,
        status: int,
        message: str,
        error_type: str,
        *,
        code: str | None = None,
        retry_after: str | None = None,
    ) -> None:
        """Send an OpenAI-style JSON error body."""
        error: dict = {"message": message, "type": error_type}
        if code is not None:
            error["code"] = code
        payload = json.dumps({"error": error}, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if retry_after is not None:
            self.send_header("Retry-After", retry_after)
        self.end_headers()
        self.wfile.write(payload)

    def _error_dialect(self, path: str):
        """How the engine route this request targets answers an error, as
        server.path_errors chooses it; the proxy has no API of its own."""
        if path.startswith("/v1/messages"):
            return ANTHROPIC_ERRORS
        if path == "/v1/systemone":
            return SYSTEMONE_ERRORS
        return OPENAI_ERRORS

    def _send_auth_error(self, error: APIError) -> None:
        """Answer an APIError rejected before any forwarding with the payload
        and status the engine route would have used (including the 401
        ``WWW-Authenticate`` header)."""
        status, _code, payload = self._error_dialect(
            self.path.partition("?")[0]
        ).answer(error)
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            self.send_response(status)
            if status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, body: object) -> None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send(self, status: int, body: str, content_type: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def serve_multi(args) -> int:
    """Launch the multi-model supervisor.

    Args:
        args: Parsed arguments from ``argparse``.  Must include
            ``config`` and the serve-multi flags; ``client_args`` (when
            present) are the raw 'splash serve' flags given after ``--``
            and are passed through to every engine instance.

    Returns:
        0 on success, 1 on error.
    """
    passthrough = list(getattr(args, "client_args", None) or [])
    models = load_config(args.config)
    # Proxy bind settings come from the same serve flags the engines get.
    host = _flag_value(passthrough, "--host", "127.0.0.1")
    raw_port = _flag_value(
        passthrough, "--port", os.environ.get("SPLASH_PORT", str(serve_options.DEFAULT_PORT))
    )
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        raise ValueError(f"invalid --port {raw_port!r}") from None
    if not 0 < port < 2**16:
        raise ValueError(f"--port must be between 1 and 65535, got {raw_port}")
    api_key = _flag_value(passthrough, "--api-key", os.environ.get("SPLASH_API_KEY"))
    if api_key is not None:
        # Fail fast on an unusable key instead of mid-deployment.
        api_key = validate_api_key(api_key)
    shared: dict = {
        "switch_timeout": args.switch_timeout,
        "startup_timeout": getattr(args, "startup_timeout", None),
        "api_key": api_key,
        "allowed_host": _flag_values(passthrough, "--allowed-host"),
        "allowed_origin": _allowed_origins(passthrough),
        "passthrough": _engine_arguments(passthrough),
    }
    shared = {k: v for k, v in shared.items() if v is not None}

    supervisor = Supervisor(models, shared, host, port)

    # Install signal handlers so Ctrl+C / SIGTERM stops the supervisor cleanly.
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda signum, frame: supervisor.shutdown())

    try:
        # Start the proxy on the user-facing port.
        server = _ForwardingHTTPServer(
            (host, port),
            functools.partial(_ServeMultiHandler, supervisor=supervisor),
            supervisor=supervisor,
        )
        supervisor._proxy_server = server
        threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="serve-multi-proxy",
        ).start()
        threading.Thread(
            target=supervisor._reap_loop,
            daemon=True,
            name="serve-multi-reaper",
        ).start()
        print(
            f"serve-multi · proxy listening on "
            f"http://{host}:{port}  (models: "
            f"{', '.join(m['model'] for m in models)})",
            flush=True,
        )
        # A model marked 'preload' loads eagerly instead of waiting for
        # the first request; if several set it, the earliest one wins.
        initial = next(
            (m["model"] for m in models if m.get("preload")), None
        )
        if initial is not None:
            supervisor._load_initial_model(initial)
        # Block until a signal calls supervisor.shutdown().
        while not supervisor._stop.is_set():
            time.sleep(0.5)
        return 0
    except (RuntimeError, OSError) as error:
        print(f"serve-multi: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        supervisor.shutdown()
        # Restore original signal handlers.
        for signum, handler in previous.items():
            signal.signal(signum, handler)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pick_free_port(host: str) -> int:
    """Return a free TCP port by binding and immediately closing a socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def _flag_value(
    passthrough: list[str], name: str, default: str | None = None
) -> str | None:
    """Return the last value of a flag in a raw 'splash serve' argument list.

    Understands both ``--flag value`` and ``--flag=value`` forms.
    """
    value = default
    for index, arg in enumerate(passthrough):
        if arg == name:
            if index + 1 < len(passthrough):
                value = passthrough[index + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def _flag_values(passthrough: list[str], name: str) -> list[str]:
    """Return all values of a repeatable flag in a raw serve argument list."""
    values: list[str] = []
    for index, arg in enumerate(passthrough):
        if arg == name:
            if index + 1 < len(passthrough):
                values.append(passthrough[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    return values


def _allowed_origins(passthrough: list[str]) -> list[str]:
    """Every --allowed-origin value in a raw serve argument list, checked and
    parsed as the server parses its own (fail fast, before any engine runs)."""
    values = _flag_values(passthrough, "--allowed-origin")
    try:
        return [serve_options.parse_allowed_origin(value) for value in values]
    except ValueError as error:
        raise ValueError(f"--allowed-origin: {error}") from None


def _merge_server_arguments(*argument_lists: list[str]) -> list[str]:
    """Merge raw 'splash serve' argument lists, flag by flag.

    Every list is normalized to its ``--flag`` name, so ``--max-context
    64000`` and ``--max-context=64000`` collide correctly; when the same
    flag appears in several lists, the value from the *latest* list wins.
    A flag whose next token is not another flag is taken as a boolean.
    Positional tokens are rejected.
    """
    merged: dict[str, list[str]] = {}
    for arguments in argument_lists:
        index = 0
        while index < len(arguments):
            arg = arguments[index]
            if not arg.startswith("--"):
                raise ValueError(f"expected '--flag' arguments, found {arg!r}")
            if "=" in arg:
                merged[arg.split("=", 1)[0]] = [arg]
            elif (
                index + 1 < len(arguments)
                and not arguments[index + 1].startswith("--")
            ):
                merged[arg] = [arg, arguments[index + 1]]
                index += 1
            else:
                merged[arg] = [arg]
            index += 1
    return [token for tokens in merged.values() for token in tokens]


def _engine_arguments(passthrough: list[str]) -> list[str]:
    """Serve arguments to hand to the engine child process.

    ``--host``/``--port`` are dropped because the child always binds an
    internal free port (the proxy owns the user-facing one), and
    ``--api-key`` travels through SPLASH_API_KEY instead of the command line.
    """
    dropped = {"--host", "--port", "--api-key"}
    out: list[str] = []
    index = 0
    while index < len(passthrough):
        arg = passthrough[index]
        if arg.split("=", 1)[0] in dropped:
            if "=" not in arg and index + 1 < len(passthrough):
                index += 1  # skip the flag's value as well
            index += 1
            continue
        out.append(arg)
        index += 1
    return out


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Serve flags go after '--' and are passed through to every engine.
    passthrough: list[str] = []
    if "--" in argv:
        boundary = argv.index("--")
        argv, passthrough = argv[:boundary], argv[boundary + 1 :]
    parser = argparse.ArgumentParser(
        prog="splash serve-multi",
        description="Serve multiple models; restart the engine when a request "
        "names another. 'splash serve' flags are given after '--'.",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="FILE",
        help="JSON file listing the models to serve",
    )
    parser.add_argument(
        "--switch-timeout",
        type=float,
        default=None,
        help=(
            "seconds a request waits while its model loads "
            "(default 600; 0 = answer 503 immediately)"
        ),
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.client_args = passthrough
    return serve_multi(args) or 0


if __name__ == "__main__":
    raise SystemExit(_main())
