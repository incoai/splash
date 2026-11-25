"""The nodes with ComfyUI's modules stubbed in sys.modules (comfy_stubs)."""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import logging
import sys
import types

import numpy as np
import pytest
from aiohttp import web

import comfy_stubs
from conftest import ROOT, free_port, start_fake
from fake_splash import FakeSplash
from test_launcher import argvs, fake_bin  # noqa: F401 (fixture)

ADVANCED = ["generate_audio", "preview", "base_url", "api_key", "auto_start", "unload_comfy_models"]


@pytest.fixture
def stubbed():
    mm, utils, server = comfy_stubs.install()
    sys.modules.pop("comfyui_splash.nodes", None)
    nodes = importlib.import_module("comfyui_splash.nodes")
    nodes.POLL_INTERVAL = 0.01
    for cls in nodes.NODES:
        cls.hidden = types.SimpleNamespace(unique_id="7")
    yield nodes, mm, server
    sys.modules.pop("comfyui_splash.nodes", None)
    comfy_stubs.uninstall()


def ids(schema):
    return [i.id for i in schema.inputs]


def test_schemas(stubbed):
    nodes, _, _ = stubbed
    t2v = nodes.SplashH3TextToVideo.define_schema()
    flf = nodes.SplashH3FirstLastFrameToVideo.define_schema()
    ref = nodes.SplashH3ReferenceToVideo.define_schema()
    assert [s.node_id for s in (t2v, flf, ref)] == [
        "SplashH3TextToVideo", "SplashH3FirstLastFrameToVideo", "SplashH3ReferenceToVideo"]
    assert [s.display_name for s in (t2v, flf, ref)] == [
        "Splash H3 Text to Video", "Splash H3 First-Last-Frame to Video", "Splash H3 Reference to Video"]
    assert all(s.category == "video/Splash" for s in (t2v, flf, ref))
    assert all([o.io_type for o in s.outputs] == ["VIDEO"] for s in (t2v, flf, ref))
    assert ids(t2v) == ["prompt", "size", "seconds", "seed", "variant", *ADVANCED]
    assert ids(flf) == ["prompt", "first_frame", "last_frame", "resolution", "seconds", "seed", "variant", *ADVANCED]
    assert ids(ref) == ["prompt", "pictures", "videos", "audio_clips", "size", "seconds", "seed", *ADVANCED]
    by_id = {i.id: i for i in t2v.inputs}
    assert by_id["size"].options == ["1344x768", "768x1344", "832x480", "480x832"]
    assert by_id["size"].default == "1344x768" == {i.id: i for i in ref.inputs}["size"].default
    resolution = {i.id: i for i in flf.inputs}["resolution"]
    assert (resolution.options, resolution.default) == (["768p", "480p"], "768p")
    seconds, seed = by_id["seconds"], by_id["seed"]
    assert (seconds.default, seconds.min, seconds.max, seconds.display_mode.value) == (5, 4, 15, "slider")
    assert (seed.min, seed.max, seed.control_after_generate) == (0, 2**32 - 1, True)
    assert by_id["variant"].options == ["auto", "vdn-8", "vdn-50", "fasth3-8", "fasth3-4", "pika-6"]
    assert {i.id: i for i in flf.inputs}["base_url"].default == "http://127.0.0.1:8000"
    assert {i.id: i for i in ref.inputs}["base_url"].default == "http://127.0.0.1:8000"
    assert all(by_id[name].advanced for name in ["variant", *ADVANCED])
    defaults = {name: by_id[name].default for name in ADVANCED}
    assert defaults == {"generate_audio": True, "preview": False, "base_url": "http://127.0.0.1:8000",
                        "api_key": "", "auto_start": True, "unload_comfy_models": True}
    flf_by_id = {i.id: i for i in flf.inputs}
    assert not flf_by_id["first_frame"].optional and flf_by_id["last_frame"].optional
    grow = {i.id: i for i in ref.inputs}
    assert [(grow[k].template.input.id, len(grow[k].template.names), grow[k].template.min) for k in
            ("pictures", "videos", "audio_clips")] == [("picture", 9, 0), ("video", 3, 0), ("audio", 3, 0)]
    assert [grow[k].template.names[0] for k in ("pictures", "videos", "audio_clips")] == [
        "picture_1", "video_1", "audio_1"]


async def test_extension_lists_nodes(stubbed):
    nodes, _, _ = stubbed
    package = sys.modules["comfyui_splash"]
    assert not hasattr(package, "NODE_CLASS_MAPPINGS")
    extension = nodes.SplashExtension()
    assert await extension.get_node_list() == nodes.NODES


def _video_bytes(output):
    (video,) = output.args
    return video.file.getvalue()


async def test_t2v_execute(stubbed, fake):
    nodes, mm, server = stubbed
    state, url = fake
    out = await nodes.SplashH3TextToVideo.execute(
        prompt="a cat", size="480x832", seconds=6, seed=11,
        variant="auto", generate_audio=False, preview=True, base_url=url, auto_start=False)
    assert _video_bytes(out) == state.video()
    assert state.posts[0].json == {"prompt": "a cat", "seconds": "6", "size": "480x832", "seed": 11,
                                   "audio": False, "preview": True}
    assert mm.unloads == 1 and mm.empties == 1
    texts = [t for node, t in server.texts if node == "7"]
    assert texts[0] == "queued" and any("denoise · 40% · " in t for t in texts)
    bar = comfy_stubs.ProgressBar.instances[-1]
    assert bar.updates[-1][0] == 100
    previews = [p for _, p in bar.updates if p is not None]
    assert len(previews) == 1 and previews[0][0] == "JPEG" and previews[0][2] == 512


async def test_flf_execute_portrait_and_last_frame(stubbed, fake):
    nodes, _, _ = stubbed
    state, url = fake
    first = np.full((1, 1280, 720, 3), 0.25, np.float32)
    last = np.full((1, 1280, 720, 3), 0.75, np.float32)
    await nodes.SplashH3FirstLastFrameToVideo.execute(
        prompt="go", resolution="768p", seconds=5, first_frame=first, seed=3, last_frame=last,
        variant="vdn-50", base_url=url, unload_comfy_models=False)
    post = state.posts[0]
    assert post.fields()["size"] == "768x1344" and post.fields()["variant"] == "vdn-50"
    assert [(p.name, p.content_type) for p in post.files()] == [
        ("input_reference", "image/png"), ("last_frame", "image/png")]


async def test_request_log_line_names_the_schema_node(stubbed, fake, caplog):
    caplog.set_level(logging.INFO, logger="comfyui_splash")
    nodes, _, _ = stubbed
    state, url = fake
    real = nodes.SplashH3FirstLastFrameToVideo
    clone = type(f"{real.__name__}Clone", (real,) + real.__bases__, dict(real.__dict__))  # as ComfyUI runs it
    first = np.full((1, 480, 832, 3), 0.5, np.float32)
    await clone.execute(prompt="go", resolution="480p", seconds=5, first_frame=first, seed=0,
                        base_url=url, unload_comfy_models=False)
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("splash request ")]
    assert line.startswith("splash request node=SplashH3FirstLastFrameToVideo format=multipart ")
    assert " parts=input_reference:first_frame.png:" in line


async def test_flf_landscape_first_only(stubbed, fake):
    nodes, _, _ = stubbed
    state, url = fake
    await nodes.SplashH3FirstLastFrameToVideo.execute(
        prompt="go", resolution="480p", seconds=5, first_frame=np.zeros((1, 360, 640, 3), np.float32),
        seed=3, base_url=url)
    post = state.posts[0]
    assert post.fields()["size"] == "832x480" and "variant" not in post.fields()
    assert [p.name for p in post.files()] == ["input_reference"]


async def test_reference_execute(stubbed, fake):
    nodes, _, _ = stubbed
    state, url = fake

    class Video:
        def get_stream_source(self):
            return io.BytesIO(b"mp4-bytes")

        def get_container_format(self):
            return "mov,mp4,m4a,3gp,3g2,mj2"

        def get_active_trim_window(self):
            return 0.0, 0.0

    audio = {"waveform": np.zeros((1, 2, 32000 * 3), np.float32), "sample_rate": 32000}
    await nodes.SplashH3ReferenceToVideo.execute(
        prompt="<Picture 1>", size="1344x768", seconds=5, seed=1,
        pictures={"picture_1": np.zeros((1, 64, 64, 3), np.float32),
                          "picture_2": np.ones((1, 64, 64, 3), np.float32)},
        videos={"video_1": Video()}, audio_clips={"audio_1": audio}, base_url=url)
    post = state.posts[0]
    assert "variant" not in post.fields() and post.fields()["size"] == "1344x768"
    assert [(p.name, p.content_type) for p in post.files()] == [
        ("references", "image/png"), ("references", "image/png"), ("references", "video/mp4"),
        ("references", "audio/wav")]
    assert post.files()[2].data == b"mp4-bytes"


async def test_reference_requires_image_or_video(stubbed, fake):
    """No reference, or audio alone: refused before any request."""
    nodes, _, _ = stubbed
    state, url = fake
    audio = {"waveform": np.zeros((1, 2, 32000 * 3), np.float32), "sample_rate": 32000}
    for extra in ({}, {"audio_clips": {"audio_1": audio}}):
        with pytest.raises(ValueError, match="Connect a picture or a video to the Reference node"):
            await nodes.SplashH3ReferenceToVideo.execute(
                prompt="x", size="832x480", seconds=5, seed=1, base_url=url, **extra)
    assert state.posts == []


async def test_interrupt_raises_comfy_exception_and_deletes(stubbed, fake):
    nodes, mm, server = stubbed
    state, url = fake
    state.script = [{"status": "in_progress", "phase": "denoise", "progress": p} for p in range(1, 90)]
    real = server.send_progress_text

    def interrupt_on_third(text, node_id, sid=None):
        real(text, node_id, sid)
        if len(server.texts) == 3:
            mm.interrupt_processing = True

    server.send_progress_text = interrupt_on_third
    with pytest.raises(comfy_stubs.InterruptProcessingException):
        await nodes.SplashH3TextToVideo.execute(
            prompt="x", size="832x480", seconds=5, seed=1, base_url=url)
    assert len(state.deletes) == 1 and mm.interrupt_processing is False


def _serve_models(log):
    return [(a[a.index("--model") + 1], a[a.index("--port") + 1]) for a in argvs(log)]


async def test_t2v_auto_starts_default_variant(stubbed, fake_bin, tmp_path):
    nodes, _, _ = stubbed
    make, log, _ = fake_bin
    make("good")
    nodes._log_dir = lambda: str(tmp_path)
    port = free_port()
    await nodes.SplashH3TextToVideo.execute(
        prompt="x", size="832x480", seconds=5, seed=1, base_url=f"http://127.0.0.1:{port}")
    assert _serve_models(log) == [("MiniMaxAI/MiniMax-H3:pika-6", str(port))]


async def test_every_node_defaults_to_port_8000(stubbed, monkeypatch):
    nodes, _, _ = stubbed
    calls = []

    async def ensure_server(base_url, variant, **kwargs):
        calls.append((base_url, variant))
        raise nodes.SplashError("unreachable", "stop here")

    monkeypatch.setattr(nodes.launcher, "ensure_server", ensure_server)
    picture = {"picture_1": np.zeros((1, 64, 64, 3), np.float32)}
    for kwargs in ({}, {"base_url": "  "}):
        with pytest.raises(RuntimeError, match="stop here"):
            await nodes.SplashH3ReferenceToVideo.execute(
                prompt="x", size="832x480", seconds=5, seed=1, pictures=picture, **kwargs)
    with pytest.raises(RuntimeError, match="stop here"):
        await nodes.SplashH3TextToVideo.execute(
            prompt="x", size="832x480", seconds=5, seed=1, base_url="")
    assert calls == [("http://127.0.0.1:8000", "pika-ref2va-6"), ("http://127.0.0.1:8000", "pika-ref2va-6"),
                     ("http://127.0.0.1:8000", "pika-6")]


async def test_variant_not_installed_message_reaches_user(stubbed, fake):
    nodes, _, _ = stubbed
    state, url = fake
    message = ("the vdn-50 variant of MiniMaxAI/MiniMax-H3 is not installed: "
               "splash install MiniMaxAI/MiniMax-H3:vdn-50, then start the server again")
    state.reject = (400, message, "invalid_request_error")
    with pytest.raises(RuntimeError) as info:
        await nodes.SplashH3TextToVideo.execute(
            prompt="x", size="832x480", seconds=5, seed=1, variant="vdn-50", base_url=url)
    assert str(info.value) == f"Splash invalid_request_error: {message}"


@pytest.mark.parametrize("body", [b"", b"<html>proxy error</html>"], ids=["empty", "html"])
async def test_bad_download_reported_at_the_node(stubbed, body):
    """Empty or non-mp4 bytes for the clip are refused at the node."""
    nodes, _, _ = stubbed
    state = FakeSplash(mp4=body)
    runner, url = await start_fake(state)
    try:
        with pytest.raises(RuntimeError) as info:
            await nodes.SplashH3TextToVideo.execute(prompt="x", size="832x480", seconds=5, seed=1, base_url=url)
    finally:
        await runner.cleanup()
    assert str(info.value) == (f"Splash bad_response: the clip from the Splash server at {url} is not an mp4: "
                               f"{len(body)} bytes without an ftyp box at bytes 4-8")


class LanguageModelSplash:
    """A Splash language-model server on the port: /ready 200, POST /v1/videos
    a JSON 404 not_found, /v1/models a model without output modalities."""

    def app(self):
        app = web.Application()
        app.router.add_get("/ready", self.ready)
        app.router.add_post("/v1/videos", self.videos)
        app.router.add_get("/v1/models", self.models)
        return app

    async def ready(self, request):
        return web.json_response({"status": "ready"})

    async def videos(self, request):
        return web.json_response(
            {"error": {"message": "not found", "type": "invalid_request_error", "code": "not_found"}}, status=404)

    async def models(self, request):
        return web.json_response({"object": "list", "data": [{"id": "some/llm", "object": "model"}]})


@pytest.mark.parametrize("kind", ["foreign_html", "language_model"])
async def test_not_video_server_message(stubbed, foreign, kind):
    """/ready answers 200 but POST /v1/videos is a 404: the node names what
    the server lists and the command that starts a video server; no HTML."""
    nodes, _, _ = stubbed
    if kind == "foreign_html":
        state, url = await foreign(ready_ok=True)
        served, code, variant = "no model it lists", "http_404", "auto"
    else:
        runner, url = await start_fake(LanguageModelSplash())
        served, code, variant = "some/llm (no output modality)", "not_found", "vdn-8"
    try:
        with pytest.raises(RuntimeError) as info:
            await nodes.SplashH3TextToVideo.execute(
                prompt="x", size="832x480", seconds=5, seed=1, variant=variant, base_url=url)
    finally:
        if kind == "language_model":
            await runner.cleanup()
    port = url.rsplit(":", 1)[1]
    model = "pika-6" if variant == "auto" else variant
    assert str(info.value) == (
        f"Splash not_video_server: the server at {url} serves {served}, not an H3 video model: it answers "
        f"POST /v1/videos with HTTP 404 ({code}). The video nodes need a Splash video server: stop this one and "
        f"run `splash serve --model MiniMaxAI/MiniMax-H3:{model} --port {port}`, or set base_url to where one runs.")
    if kind == "foreign_html":
        assert ("GET", "/v1/models") in state.paths


def test_stubs_replace_server_module_only_while_installed():
    """The stubs stand in for ComfyUI's `server` module (PromptServer) and
    are removed afterwards; no other `server` package is importable from the
    test run (Splash's own server/ is never on the path)."""
    assert "server" not in sys.modules and importlib.util.find_spec("server") is None
    comfy_stubs.install()
    try:
        assert hasattr(sys.modules["server"], "PromptServer") and not hasattr(sys.modules["server"], "__path__")
    finally:
        comfy_stubs.uninstall()
    assert "server" not in sys.modules


def test_package_imports_without_comfy():
    """Outside ComfyUI the package imports and only the entry point refuses."""
    import subprocess

    code = (
        "import importlib.util, sys, asyncio;"
        f"spec = importlib.util.spec_from_file_location('pkg', {str(ROOT / '__init__.py')!r},"
        f" submodule_search_locations=[{str(ROOT)!r}]);"
        "m = importlib.util.module_from_spec(spec); sys.modules['pkg'] = m; spec.loader.exec_module(m);"
        "print(m.SplashExtension is None, hasattr(m, 'NODE_CLASS_MAPPINGS'));"
        "\ntry:\n asyncio.run(m.comfy_entrypoint())\nexcept ImportError as e:\n print('ImportError', 'comfy' in str(e))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.split() == ["True", "False", "ImportError", "True"], out.stderr


def test_example_workflow_uses_existing_nodes(stubbed):
    nodes, _, _ = stubbed
    known = {cls.define_schema().node_id for cls in nodes.NODES} | {"SaveVideo"}
    workflow = json.loads((ROOT / "example_workflows" / "splash_h3_t2v.json").read_text())
    types_used = {node["type"] for node in workflow["nodes"]}
    assert types_used <= known and "SplashH3TextToVideo" in types_used and "SaveVideo" in types_used
    t2v = next(n for n in workflow["nodes"] if n["type"] == "SplashH3TextToVideo")
    save = next(n for n in workflow["nodes"] if n["type"] == "SaveVideo")
    (link,) = workflow["links"]
    assert link[1] == t2v["id"] and link[3] == save["id"] and link[5] == "VIDEO"
    # widgets_values follow the node's inputs in order; a seed's
    # control_after_generate setting follows the seed.
    names = []
    for item in nodes.SplashH3TextToVideo.define_schema().inputs:
        names += [item.id, "control_after_generate"] if item.control_after_generate else [item.id]
    values = dict(zip(names, t2v["widgets_values"], strict=True))
    assert values["size"] in nodes.options.SIZES and values["variant"] in nodes.options.VARIANTS
    assert (values["seconds"], values["control_after_generate"], values["base_url"]) == (
        5, "randomize", "http://127.0.0.1:8000")
