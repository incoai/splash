from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

import pytest
from aiohttp import web

from fake_splash import FakeSplash

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "comfyui_splash"


def _register_package():
    """Import the pack folder as package `comfyui_splash`, as ComfyUI imports a
    custom-node directory (its __init__.py guards the comfy import)."""
    if PACKAGE in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)


_register_package()


async def start_fake(state: FakeSplash, port: int = 0):
    runner = web.AppRunner(state.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


@pytest.fixture
async def fake():
    """A running FakeSplash on 127.0.0.1; yields (fake, base_url)."""
    state = FakeSplash()
    runner, url = await start_fake(state)
    try:
        yield state, url
    finally:
        await runner.cleanup()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def closed_port():
    return free_port()


HTML_404 = ("<!DOCTYPE html>\n<html><head><title>404 Not Found</title></head>"
            "<body><h1>Not Found</h1><p>The requested URL was not found on this server.</p></body></html>\n")


class Foreign:
    """Another program on the port: every path 404 HTML, or /ready 200 and
    the rest 404 HTML (`ready_ok`). Records (method, path) of each request."""

    def __init__(self, ready_ok=False):
        self.ready_ok = ready_ok
        self.paths = []

    def app(self):
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self.any)
        return app

    async def any(self, request):
        self.paths.append((request.method, request.path))
        if self.ready_ok and request.path == "/ready":
            return web.Response(text="OK")
        return web.Response(status=404, text=HTML_404, content_type="text/html")


@pytest.fixture
async def foreign():
    """make(ready_ok=False) -> (Foreign, base_url), running on 127.0.0.1."""
    runners = []

    async def make(**kw):
        state = Foreign(**kw)
        runner, url = await start_fake(state)
        runners.append(runner)
        return state, url

    yield make
    for runner in runners:
        await runner.cleanup()
