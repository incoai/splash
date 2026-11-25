"""Auto-start of a local Splash server for the nodes.

Applies only to a base_url on 127.0.0.1 or localhost. When nothing listens
there, it starts `splash serve --model MiniMaxAI/MiniMax-H3:<variant>
--offline --port <port>` in a session of its own and waits for /ready. That
server serves every variant of the model installed when it started, the one
named first (videos.hold_variants, server/videos.py), so a node that finds
it ready uses it whatever variant it was started with. The server is left
running afterwards: it is never stopped from here, also not when ComfyUI
exits, and servers on other ports are neither looked at nor stopped. No
ComfyUI imports.
"""

from __future__ import annotations

import asyncio
import collections
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from .options import DEFAULT_VARIANT
from .splash_client import SplashError, SplashInterrupted, probe_ready

MODEL = "MiniMaxAI/MiniMax-H3"
READY_TIMEOUT = 600.0
READY_POLL = 1.0
LOCAL_HOSTS = ("127.0.0.1", "localhost")
FALLBACK_BINS = ("/opt/homebrew/bin/splash", "/usr/local/bin/splash")
LOG_NAME = "splash-serve.log"
LOG_TAIL_LINES = 20


def model_ref(variant: str = DEFAULT_VARIANT) -> str:
    """The `--model` value of a variant."""
    return f"{MODEL}:{variant}"


def setup_help(variant: str = DEFAULT_VARIANT) -> str:
    return (
        "Set up Splash once: `brew install incoai/tap/splash`, then run once in a "
        f"terminal `splash install {model_ref(variant)}` (it downloads and converts "
        "the weights and shows the weights license)."
    )


# One spawn at a time in this process, across threads and event loops.
_spawn_lock = threading.Lock()
# The servers started here, kept referenced (never waited on or stopped).
SPAWNED: list = []


def is_local(base_url: str) -> bool:
    return (urlsplit(base_url).hostname or "") in LOCAL_HOSTS


def port_of(base_url: str) -> int:
    parts = urlsplit(base_url)
    return parts.port or (443 if parts.scheme == "https" else 80)


def find_splash() -> Optional[str]:
    """The splash executable: $SPLASH_BIN, PATH, then Homebrew's locations."""
    override = os.environ.get("SPLASH_BIN")
    if override:
        return override if os.access(override, os.X_OK) else None
    found = shutil.which("splash")
    if found:
        return found
    for path in FALLBACK_BINS:
        if os.access(path, os.X_OK):
            return path
    return None


def serve_argv(binary: str, port: int, variant: str) -> list[str]:
    return [binary, "serve", "--model", model_ref(variant), "--offline", "--port", str(port)]


def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        with open(path, "r", errors="replace") as handle:
            return "".join(collections.deque(handle, lines)).rstrip()
    except OSError:
        return "(no log)"


async def _acquire(lock: threading.Lock, is_interrupted):
    while not lock.acquire(blocking=False):
        if is_interrupted is not None and is_interrupted():
            raise SplashInterrupted("interrupted while waiting for the Splash start")
        await asyncio.sleep(0.05)


async def _wait_ready(
    base_url, api_key, timeout, is_interrupted, on_status, process=None, log_path=None, variant=DEFAULT_VARIANT
):
    started = time.monotonic()
    while True:
        state = await probe_ready(base_url, api_key)
        if state == "ready":
            return
        if process is not None and process.poll() is not None:
            raise SplashError(
                "splash_exited",
                f"`splash serve` exited with status {process.returncode} before it was "
                f"ready. Last lines of {log_path}:\n{_tail(log_path)}\n{setup_help(variant)}",
            )
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            where = f" Log: {log_path}" if log_path else ""
            raise SplashError(
                "start_timeout",
                f"the Splash server at {base_url} was not ready after {timeout:.0f} s.{where}",
            )
        if is_interrupted is not None and is_interrupted():
            raise SplashInterrupted("interrupted while waiting for the Splash server")
        if on_status is not None:
            on_status(f"Starting Splash… {elapsed:.0f}s")
        await asyncio.sleep(READY_POLL)


async def ensure_server(
    base_url: str,
    variant: str,
    *,
    auto_start: bool = True,
    api_key: str = "",
    is_interrupted: Optional[Callable[[], bool]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    log_dir: Optional[str] = None,
    ready_timeout: float = READY_TIMEOUT,
) -> Optional[subprocess.Popen]:
    """Make sure a Splash server answers at base_url; if one is started here,
    it is started with `variant`. A ready server is used as it is.
    Returns the process spawned here, or None when none was. A remote
    base_url is left alone (the client reports it if unreachable)."""
    if not is_local(base_url):
        return None
    state = await probe_ready(base_url, api_key)
    if state == "ready":
        return None
    if state == "starting":
        await _wait_ready(base_url, api_key, ready_timeout, is_interrupted, on_status)
        return None
    if not auto_start:
        raise SplashError(
            "unreachable",
            f"no Splash server at {base_url} and auto_start is off. {setup_help(variant)}",
        )
    await _acquire(_spawn_lock, is_interrupted)
    try:
        # Another node may have started it while this one waited for the lock.
        state = await probe_ready(base_url, api_key)
        if state != "refused":
            if state == "starting":
                await _wait_ready(base_url, api_key, ready_timeout, is_interrupted, on_status)
            return None
        binary = find_splash()
        if binary is None:
            raise SplashError(
                "splash_not_found",
                f"no Splash server at {base_url} and the `splash` command was not found. "
                f"{setup_help(variant)}",
            )
        log_path = Path(log_dir or tempfile.gettempdir()) / LOG_NAME
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if on_status is not None:
            on_status("Starting Splash…")
        argv = serve_argv(binary, port_of(base_url), variant)
        with open(log_path, "ab") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} comfyui-splash starts: "
                      f"{' '.join(argv)}\n".encode())
            log.flush()
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        SPAWNED.append(process)
        await _wait_ready(
            base_url, api_key, ready_timeout, is_interrupted, on_status, process, log_path, variant
        )
        return process
    finally:
        _spawn_lock.release()
