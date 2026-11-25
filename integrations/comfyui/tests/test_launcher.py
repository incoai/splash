"""The auto-start launcher against fake `splash` executables."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from comfyui_splash import launcher
from comfyui_splash.splash_client import SplashError
from conftest import free_port, start_fake
from fake_splash import FakeSplash

TESTS = Path(__file__).resolve().parent

GOOD = f"""#!{sys.executable}
import json, os, sys, time
with open(os.environ["FAKE_ARGV_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
port = sys.argv[sys.argv.index("--port") + 1]
time.sleep(0.5)
sys.path.insert(0, {str(TESTS)!r})
import fake_splash
fake_splash.main(["--port", port])
"""

BAD = f"""#!{sys.executable}
import json, os, sys
with open(os.environ["FAKE_ARGV_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
for i in range(30):
    print(f"log line {{i}}", flush=True)
print("error: MiniMaxAI/MiniMax-H3 is not installed; run without --offline once", file=sys.stderr)
sys.exit(1)
"""


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """Make a fake splash executable; returns (make(kind), argv_log, spawned list)."""
    argv_log = tmp_path / "argv.jsonl"
    monkeypatch.setenv("FAKE_ARGV_LOG", str(argv_log))
    spawned = []

    def make(kind):
        path = tmp_path / f"splash-{kind}"
        path.write_text(GOOD if kind == "good" else BAD)
        path.chmod(0o755)
        monkeypatch.setenv("SPLASH_BIN", str(path))
        return path

    yield make, argv_log, spawned
    for process in launcher.SPAWNED:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(10)
    launcher.SPAWNED.clear()


def argvs(log: Path):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


async def test_already_running_no_spawn(fake, fake_bin, tmp_path):
    make, log, _ = fake_bin
    make("good")
    state, url = fake
    assert await launcher.ensure_server(url, "pika-6", log_dir=str(tmp_path)) is None
    assert argvs(log) == []


async def test_spawn_then_ready(fake_bin, tmp_path):
    make, log, _ = fake_bin
    binary = make("good")
    port = free_port()
    statuses = []
    process = await launcher.ensure_server(
        f"http://127.0.0.1:{port}", "pika-6", log_dir=str(tmp_path), on_status=statuses.append, ready_timeout=30
    )
    assert process is not None and process.poll() is None
    assert argvs(log) == [["serve", "--model", "MiniMaxAI/MiniMax-H3:pika-6", "--offline", "--port", str(port)]]
    assert launcher.serve_argv(str(binary), port, "pika-6")[0] == str(binary)
    assert statuses and statuses[0].startswith("Starting Splash…")
    assert (tmp_path / launcher.LOG_NAME).exists()
    # The server is left running: a second call finds it ready and spawns nothing.
    assert await launcher.ensure_server(f"http://localhost:{port}", "pika-6", log_dir=str(tmp_path)) is None
    assert len(argvs(log)) == 1


async def test_early_exit_reports_log_tail(fake_bin, tmp_path):
    make, log, _ = fake_bin
    make("bad")
    with pytest.raises(SplashError) as info:
        await launcher.ensure_server(
            f"http://127.0.0.1:{free_port()}", "pika-ref2va-6", log_dir=str(tmp_path), ready_timeout=30
        )
    message = str(info.value)
    assert info.value.code == "splash_exited"
    assert "not installed" in message and "log line 29" in message and "log line 5\n" not in message
    assert "brew install incoai/tap/splash" in message
    assert "splash install MiniMaxAI/MiniMax-H3:pika-ref2va-6" in message
    assert len(argvs(log)) == 1


async def test_binary_missing_setup_help(monkeypatch, tmp_path):
    monkeypatch.setenv("SPLASH_BIN", str(tmp_path / "nope"))
    with pytest.raises(SplashError) as info:
        await launcher.ensure_server(f"http://127.0.0.1:{free_port()}", "pika-6", log_dir=str(tmp_path))
    assert info.value.code == "splash_not_found"
    assert "brew install incoai/tap/splash" in str(info.value)
    assert "splash install MiniMaxAI/MiniMax-H3:pika-6" in str(info.value)


async def test_auto_start_off_setup_help(fake_bin, tmp_path):
    make, log, _ = fake_bin
    make("good")
    with pytest.raises(SplashError) as info:
        await launcher.ensure_server(f"http://127.0.0.1:{free_port()}", "vdn-50", auto_start=False)
    assert info.value.code == "unreachable" and "brew install incoai/tap/splash" in str(info.value)
    assert "splash install MiniMaxAI/MiniMax-H3:vdn-50" in str(info.value)
    assert argvs(log) == []


async def test_concurrent_calls_one_spawn(fake_bin, tmp_path):
    make, log, _ = fake_bin
    make("good")
    url = f"http://127.0.0.1:{free_port()}"
    results = await asyncio.gather(
        *[launcher.ensure_server(url, "pika-6", log_dir=str(tmp_path), ready_timeout=30) for _ in range(3)]
    )
    assert len(argvs(log)) == 1
    assert sum(r is not None for r in results) == 1


async def test_remote_host_never_spawns(fake_bin, tmp_path):
    make, log, _ = fake_bin
    make("good")
    assert await launcher.ensure_server("http://192.0.2.1:8000", "pika-6", log_dir=str(tmp_path)) is None
    assert await launcher.ensure_server("http://splash.example:8000", "pika-6", auto_start=True) is None
    assert argvs(log) == []


async def test_starting_server_waited_not_spawned(fake_bin, tmp_path):
    make, log, _ = fake_bin
    make("good")
    state = FakeSplash()
    state.ready = False
    runner, url = await start_fake(state)
    try:
        async def become_ready():
            await asyncio.sleep(1.5)
            state.ready = True

        asyncio.create_task(become_ready())
        assert await launcher.ensure_server(url, "pika-6", log_dir=str(tmp_path), ready_timeout=30) is None
        assert argvs(log) == []
    finally:
        await runner.cleanup()


def test_local_detection():
    assert launcher.is_local("http://127.0.0.1:8000") and launcher.is_local("http://localhost:9")
    assert not launcher.is_local("http://10.0.0.2:8000")
    assert launcher.port_of("http://127.0.0.1:8123") == 8123


def test_model_ref():
    assert launcher.MODEL == "MiniMaxAI/MiniMax-H3"
    assert launcher.model_ref() == "MiniMaxAI/MiniMax-H3:pika-6"
    assert launcher.model_ref("vdn-8") == "MiniMaxAI/MiniMax-H3:vdn-8"
    assert launcher.serve_argv("splash", 8000, "pika-ref2va-6") == [
        "splash", "serve", "--model", "MiniMaxAI/MiniMax-H3:pika-ref2va-6", "--offline", "--port", "8000"]


def test_find_splash_order(monkeypatch, tmp_path):
    """$SPLASH_BIN alone when set; else `splash` on PATH; else Homebrew's locations."""
    def executable(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
        return str(path)

    on_path = executable(tmp_path / "bin" / "splash")
    fallback = executable(tmp_path / "brew" / "splash")
    override = executable(tmp_path / "checkout" / "splash")
    monkeypatch.setattr(launcher, "FALLBACK_BINS", (str(tmp_path / "missing"), fallback))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("SPLASH_BIN", override)
    assert launcher.find_splash() == override
    monkeypatch.setenv("SPLASH_BIN", str(tmp_path / "not-there"))
    assert launcher.find_splash() is None  # a set SPLASH_BIN is not looked past
    monkeypatch.delenv("SPLASH_BIN")
    assert launcher.find_splash() == on_path
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert launcher.find_splash() == fallback


async def test_ready_never_comes_timeout_message(tmp_path):
    state = FakeSplash()
    state.ready = False
    runner, url = await start_fake(state)
    try:
        started = time.monotonic()
        with pytest.raises(SplashError) as info:
            await launcher.ensure_server(url, "pika-6", log_dir=str(tmp_path), ready_timeout=1.0)
        assert time.monotonic() - started >= 1.0
    finally:
        await runner.cleanup()
    assert str(info.value) == f"Splash start_timeout: the Splash server at {url} was not ready after 1 s."


async def test_foreign_server_on_port_fails_fast(foreign, tmp_path):
    """A 404 on /ready is no Splash answer: one probe, no wait for the ready timeout."""
    state, url = await foreign()
    started = time.monotonic()
    with pytest.raises(SplashError) as info:
        await launcher.ensure_server(url, "pika-6", log_dir=str(tmp_path), ready_timeout=5.0)
    assert time.monotonic() - started < 2.0
    assert str(info.value) == (
        f"Splash not_splash: the program at {url} is not a Splash server: it answers GET /ready with HTTP 404 "
        "(a Splash server answers 200, or 503 while it starts). Stop that program, or set base_url to the "
        "address of a Splash server.")
    assert state.paths == [("GET", "/ready")]


async def test_api_key_refused_on_ready_message(tmp_path):
    """A Splash that guards /ready with its api key: the user is told about
    the key, not that the program is no Splash server."""
    state = FakeSplash()
    state.api_key = "sekrit"
    original = state.get_ready

    async def guarded(request):
        refused = state._auth(request)
        return refused if refused is not None else await original(request)

    state.get_ready = guarded  # app() routes to the instance attribute
    runner, url = await start_fake(state)
    try:
        with pytest.raises(SplashError) as info:
            await launcher.ensure_server(url, "pika-6", log_dir=str(tmp_path), ready_timeout=5.0)
        assert await launcher.probe_ready(url, "sekrit") == "ready"
    finally:
        await runner.cleanup()
    assert (info.value.status, info.value.code) == (401, "authentication_error")
    assert str(info.value) == (
        f"Splash authentication_error: the Splash server at {url} refused GET /ready with HTTP 401 "
        "(invalid or missing API key): set the node's api_key to the server's --api-key.")
