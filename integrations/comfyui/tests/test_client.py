from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging

import aiohttp
import av
import pytest

from comfyui_splash import options
from comfyui_splash.splash_client import (
    FilePart,
    SplashClient,
    SplashError,
    SplashInterrupted,
)

FAST = 0.01


def client(url, **kw):
    return SplashClient(url, poll_interval=FAST, **kw)


def fields(**kw):
    base = dict(prompt="a dog", seconds=5, size="832x480", seed=7, audio=True, preview=False)
    base.update(kw)
    return options.video_fields(**base)


async def test_happy_path_rising_progress_valid_mp4(fake):
    state, url = fake
    seen = []
    mp4 = await client(url).generate(fields(), on_progress=lambda job: seen.append(job["progress"]))
    assert seen == sorted(seen) and seen[0] == 0 and seen[-1] == 100
    assert len(set(seen)) >= 4
    with av.open(io.BytesIO(mp4)) as container:
        assert container.streams.video and container.streams.audio
    post = state.posts[0]
    assert post.content_type == "application/json"
    assert post.json == {"prompt": "a dog", "seconds": "5", "size": "832x480", "seed": 7,
                         "audio": True, "preview": False}
    assert [v for _, v in state.contents] == ["video"]
    assert not state.deletes


async def test_preview_fetched_once(fake):
    state, url = fake
    state.script = [
        {"status": "in_progress", "phase": "denoise", "progress": 50},
        {"phase": "preview", "progress": 91, "preview": True},
        {"phase": "decode", "progress": 97},
        {"phase": "audio", "progress": 98},
        {"status": "completed", "phase": None, "progress": 100},
    ]
    previews = []
    await client(url).generate(fields(preview=True), on_preview=previews.append)
    assert len(previews) == 1 and previews[0] == state.video()
    assert [v for _, v in state.contents] == ["preview", "video"]


async def test_job_failed(fake):
    state, url = fake
    with pytest.raises(SplashError) as info:
        await client(url).generate(fields(prompt="x [fail]"))
    assert info.value.code == "engine_failed"
    assert str(info.value) == "Splash engine_failed: the engine stopped"


async def test_503_queue_full_no_retry(fake):
    state, url = fake
    state.queue_full = True
    with pytest.raises(SplashError) as info:
        await client(url).generate(fields())
    assert (info.value.status, info.value.code) == (503, "video_queue_full")
    assert len(state.posts) == 1


async def test_connection_refused(closed_port):
    with pytest.raises(SplashError) as info:
        await client(f"http://127.0.0.1:{closed_port}").generate(fields())
    assert info.value.code == "unreachable"


async def test_interrupt_deletes_job(fake):
    state, url = fake
    state.script = [{"status": "in_progress", "phase": "denoise", "progress": p} for p in range(1, 90)]
    polls = []

    def interrupted():
        return len(polls) >= 3

    with pytest.raises(SplashInterrupted):
        await client(url).generate(fields(), on_progress=polls.append, is_interrupted=interrupted)
    assert state.deletes == [state.gets[0]] and len(polls) == 3


async def test_cancelled_deletes_job(fake):
    state, url = fake
    state.script = [{"status": "in_progress", "phase": "denoise", "progress": p} for p in range(1, 90)]
    task = asyncio.create_task(SplashClient(url, poll_interval=0.05).generate(fields()))
    while not state.gets:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.deletes == [state.gets[0]]


async def test_callback_exception_deletes_job(fake):
    """ComfyUI's progress hook raises InterruptProcessingException (a BaseException)."""
    state, url = fake

    class Stop(BaseException):
        pass

    def on_progress(job):
        if job["status"] == "in_progress":
            raise Stop()

    with pytest.raises(Stop):
        await client(url).generate(fields(), on_progress=on_progress)
    assert len(state.deletes) == 1


async def test_404_during_poll(fake):
    state, url = fake
    state.script = [{"status": "in_progress", "progress": 5}, {"_forget": True}]
    with pytest.raises(SplashError) as info:
        await client(url).generate(fields())
    assert (info.value.status, info.value.code) == (404, "not_found")
    assert str(info.value) == "Splash not_found: video not found"


async def test_api_key_header(fake):
    state, url = fake
    state.api_key = "sekrit"
    await client(url, api_key="sekrit").generate(fields())
    assert state.posts[0].headers["Authorization"] == "Bearer sekrit"
    with pytest.raises(SplashError) as info:
        await client(url).generate(fields())
    assert info.value.status == 401
    assert "Authorization" not in state.posts[1].headers


def test_variant_auto_omitted_and_never_model_or_steps():
    assert "variant" not in fields(variant="auto")
    assert fields(variant="fasth3-4")["variant"] == "fasth3-4"
    assert not {"model", "steps"} & set(fields(variant="vdn-50"))


async def test_multipart_post_reaches_server(fake):
    state, url = fake
    files = [FilePart("references", "image_1.png", "image/png", b"img"),
             FilePart("references", "video_1.mp4", "video/mp4", b"vid"),
             FilePart("references", "audio_1.wav", "audio/wav", b"wav")]
    await client(url).generate(fields(audio=False), files)
    post = state.posts[0]
    assert post.content_type.startswith("multipart/form-data; boundary=")
    assert [(p.name, p.content_type, p.data) for p in post.files()] == [
        ("references", "image/png", b"img"), ("references", "video/mp4", b"vid"), ("references", "audio/wav", b"wav")]
    assert post.fields() == {"prompt": "a dog", "seconds": "5", "size": "832x480", "seed": "7",
                             "audio": "false", "preview": "false"}


@pytest.mark.parametrize("resolution,width,height,size", [
    ("768p", 1920, 1080, "1344x768"), ("768p", 1080, 1920, "768x1344"),
    ("480p", 512, 512, "832x480"), ("480p", 480, 832, "480x832"),
])
def test_flf_size_follows_first_frame(resolution, width, height, size):
    assert options.keyframe_size(resolution, width, height) == size


async def test_non_english_prompt_json(fake):
    state, url = fake
    prompt = "一只金毛犬在海边奔跑，夕阳。 🐕 \"quoted\"\nline two"
    await client(url).generate(fields(prompt=prompt))
    post = state.posts[0]
    assert post.content_type == "application/json"
    assert json.loads(post.raw.decode("utf-8"))["prompt"] == prompt
    assert state.jobs[next(iter(state.jobs))].request["prompt"] == prompt


def _request_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "comfyui_splash" and r.getMessage().startswith("splash request ")]


async def test_request_log_line_json(fake, caplog):
    caplog.set_level(logging.INFO, logger="comfyui_splash")
    state, url = fake
    await client(url, node="SplashH3TextToVideo", api_key="secret-key").generate(fields(variant="fasth3-4"))
    [line] = _request_lines(caplog)
    digest = hashlib.sha256(b"a dog").hexdigest()[:8]
    [job_id] = state.jobs
    assert line == (
        "splash request node=SplashH3TextToVideo format=json prompt_chars=5 "
        f"prompt_sha256={digest} seconds=5 size=832x480 seed=7 audio=true preview=false "
        f"variant=fasth3-4 files=0 base_url={url} job={job_id}"
    )
    assert "a dog" not in line and "secret-key" not in line


async def test_request_log_line_multipart_non_ascii(fake, caplog):
    caplog.set_level(logging.INFO, logger="comfyui_splash")
    state, url = fake
    prompt = "一只小狗在草地上奔跑，阳光明媚。"
    files = [FilePart("input_reference", "first_frame.png", "image/png", b"\x89PNG1234"),
             FilePart("last_frame", "last_frame.png", "image/png", b"\x89PNG12")]
    await client(url, node="SplashH3FirstLastFrameToVideo").generate(fields(prompt=prompt), files)
    post = state.posts[0]
    assert post.fields()["prompt"] == prompt
    # Every text part declares UTF-8; the prompt's bytes on the wire are its UTF-8.
    text_parts = len(fields(prompt=prompt))
    assert post.raw.count(b"Content-Type: text/plain; charset=utf-8\r\n\r\n") == text_parts
    assert b"charset=utf-8\r\n\r\n" + prompt.encode("utf-8") + b"\r\n" in post.raw
    [line] = _request_lines(caplog)
    words = dict(word.split("=", 1) for word in line.split()[2:])
    assert words["node"] == "SplashH3FirstLastFrameToVideo" and words["format"] == "multipart"
    assert words["prompt_chars"] == str(len(prompt)) == "16"
    assert words["prompt_sha256"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    assert words["files"] == "2"
    assert words["parts"] == "input_reference:first_frame.png:8,last_frame:last_frame.png:6"
    assert words["job"] in state.jobs
    assert prompt not in line and "小狗" not in line


async def test_request_log_line_on_error(fake, caplog):
    caplog.set_level(logging.INFO, logger="comfyui_splash")
    state, url = fake
    state.reject = (400, "seconds must be a whole number from 5 to 15", "invalid_request_error")
    with pytest.raises(SplashError):
        await client(url, node="SplashH3TextToVideo").generate(fields())
    [line] = _request_lines(caplog)
    assert line.endswith(f"base_url={url} error=invalid_request_error status=400")
    assert "job=" not in line


async def test_non_json_error_body_is_short(foreign):
    """A non-JSON error body: status, URL and a short plain text of it, no tags."""
    _, url = await foreign(ready_ok=True)
    async with aiohttp.ClientSession() as session:
        with pytest.raises(SplashError) as info:
            await SplashClient(url).get(session, "video_x")
    assert (info.value.status, info.value.code) == (404, "http_404")
    assert str(info.value) == (
        f"Splash http_404: HTTP 404 from {url}/v1/videos/video_x, not a Splash error answer: 404 Not Found "
        "Not Found The requested URL was not found on this server.")
