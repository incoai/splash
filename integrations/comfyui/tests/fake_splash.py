"""A stateful fake of Splash's /v1/videos routes (aiohttp), for tests and for
checks against a real ComfyUI.

Each GET of a job applies the next step of its script (a dict of job
fields), so a test controls how the job advances. A prompt containing
"[slow]" holds every in-progress step for SLOW_HOLD polls; "[fail]" makes
the job fail. The mp4 it serves is a small valid H.264 + AAC clip made with
PyAV at start.

Standalone, from this directory: python fake_splash.py --port 8765 [--log requests.jsonl]
"""

from __future__ import annotations

import argparse
import email.parser
import email.policy
import io
import json
import secrets
import time
from dataclasses import dataclass, field
from fractions import Fraction

from aiohttp import web

MODEL = "MiniMaxAI/MiniMax-H3"
FIELDS = {
    "prompt", "model", "seconds", "size", "variant", "seed", "steps",
    "preview", "audio", "input_reference", "last_frame", "references",
}
FILE_FIELDS = ("input_reference", "last_frame", "references")

DEFAULT_SCRIPT = [
    {"status": "in_progress", "phase": "encode", "progress": 2},
    {"phase": "denoise", "progress": 40},
    {"phase": "preview", "progress": 91, "preview": True},
    {"phase": "decode", "progress": 97},
    {"status": "completed", "phase": None, "progress": 100},
]
FAIL_SCRIPT = [
    {"status": "in_progress", "phase": "denoise", "progress": 10},
    {"status": "failed", "phase": None, "error": {"code": "engine_failed", "message": "the engine stopped"}},
]
SLOW_HOLD = 120

_MP4 = None


def make_mp4(seconds: float = 1.0, width: int = 64, height: int = 64) -> bytes:
    """A small valid mp4: H.264 video (24 fps) and an AAC stereo soundtrack."""
    import av
    import numpy as np

    out = io.BytesIO()
    with av.open(out, mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=24)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        audio = container.add_stream("aac", rate=32000)
        audio.layout = "stereo"
        frames = int(seconds * 24)
        for i in range(frames):
            pixels = np.full((height, width, 3), (i * 255 // max(frames - 1, 1)), np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        samples, chunk = int(seconds * 32000), 1024
        t = np.arange(samples) / 32000
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for start in range(0, samples, chunk):
            part = tone[start:start + chunk]
            data = np.stack([part, part])
            frame = av.AudioFrame.from_ndarray(data, format="fltp", layout="stereo")
            frame.sample_rate = 32000
            frame.pts = start
            frame.time_base = Fraction(1, 32000)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return out.getvalue()


def sample_mp4() -> bytes:
    global _MP4
    if _MP4 is None:
        _MP4 = make_mp4()
    return _MP4


@dataclass
class Part:
    name: str
    filename: str | None
    content_type: str
    data: bytes

    @property
    def text(self):
        return self.data.decode("utf-8")


@dataclass
class Post:
    headers: dict
    content_type: str
    raw: bytes
    parts: list
    json: dict | None = None

    def fields(self):
        if self.json is not None:
            return dict(self.json)
        return {p.name: p.text for p in self.parts if p.filename is None}

    def files(self):
        return [p for p in self.parts if p.filename is not None]

    def summary(self):
        return {
            "content_type": self.content_type.split(";")[0],
            "authorization": self.headers.get("Authorization"),
            "fields": self.fields(),
            "files": [
                {"name": p.name, "filename": p.filename, "content_type": p.content_type, "bytes": len(p.data)}
                for p in self.files()
            ],
        }


def error(status, message, code):
    kind = "server_error" if status >= 500 else "invalid_request_error"
    return web.json_response({"error": {"message": message, "type": kind, "code": code}}, status=status)


def parse_multipart(content_type: str, raw: bytes) -> list:
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + raw
    )
    if not message.is_multipart():
        raise ValueError("not multipart")
    parts = []
    for part in message.iter_parts():
        parts.append(Part(
            part.get_param("name", header="content-disposition"),
            part.get_filename(),
            part.get_content_type(),
            part.get_payload(decode=True),
        ))
    return parts


@dataclass
class Job:
    id: str
    request: dict
    script: list
    state: dict = field(default_factory=dict)

    def public(self):
        return {
            "id": self.id,
            "object": "video",
            "status": "queued",
            "progress": 0,
            "phase": None,
            "error": None,
            "preview": False,
            "size": self.request.get("size", "1344x768"),
            "seconds": str(self.request.get("seconds", "5")),
            **self.state,
        }


class FakeSplash:
    def __init__(self, script=None, mp4: bytes | None = None, log_path: str | None = None):
        self.script = list(script or DEFAULT_SCRIPT)
        self.mp4 = mp4
        self.log_path = log_path
        self.ready = True
        self.queue_full = False
        self.reject = None  # (status, message, code) answered to the next POST
        self.api_key = None
        self.jobs: dict[str, Job] = {}
        self.posts: list[Post] = []
        self.gets: list[str] = []
        self.contents: list[tuple[str, str]] = []
        self.deletes: list[str] = []
        self.events: list[dict] = []

    def _log(self, event: dict):
        event = {"t": round(time.time(), 3), **event}
        self.events.append(event)
        if self.log_path:
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(event) + "\n")

    def video(self) -> bytes:
        if self.mp4 is None:
            self.mp4 = sample_mp4()
        return self.mp4

    def app(self) -> web.Application:
        app = web.Application(client_max_size=512 * 1024 * 1024)
        app.router.add_get("/ready", self.get_ready)
        app.router.add_post("/v1/videos", self.post_video)
        app.router.add_get("/v1/videos/{id}", self.get_video)
        app.router.add_get("/v1/videos/{id}/content", self.get_content)
        app.router.add_delete("/v1/videos/{id}", self.delete_video)
        app.router.add_get("/_fake/log", self.get_log)
        app.router.add_get("/v1/models", self.get_models)
        return app

    def _auth(self, request):
        if self.api_key is None:
            return None
        if request.headers.get("Authorization") != f"Bearer {self.api_key}":
            return error(401, "invalid or missing API key", "authentication_error")
        return None

    async def get_ready(self, request):
        return web.json_response(
            {"status": "ready" if self.ready else "unavailable"}, status=200 if self.ready else 503
        )

    async def get_log(self, request):
        return web.json_response(self.events)

    async def get_models(self, request):
        return web.json_response({"object": "list", "data": [
            {"id": MODEL, "object": "model", "created": 0, "owned_by": "splash",
             "input_modalities": ["text", "image"], "output_modalities": ["video", "audio"]}
        ]})

    async def post_video(self, request):
        raw = await request.read()
        content_type = request.headers.get("Content-Type", "")
        post = Post(dict(request.headers), content_type, raw, [])
        if content_type.startswith("multipart/form-data"):
            post.parts = parse_multipart(content_type, raw)
        else:
            post.json = json.loads(raw or b"{}")
        self.posts.append(post)
        self._log({"event": "POST /v1/videos", **post.summary()})
        if denied := self._auth(request):
            return denied
        if self.reject:
            status, message, code = self.reject
            self.reject = None
            return error(status, message, code)
        if self.queue_full:
            return error(503, "the video queue is full; try again later", "video_queue_full")
        names = set(post.fields()) | {p.name for p in post.files()}
        if unknown := sorted(names - FIELDS):
            return error(400, f"unsupported parameter: {unknown[0]}", "unsupported_parameter")
        for p in post.files():
            if p.name not in FILE_FIELDS:
                return error(400, f"{p.name} is not supported", "unsupported_parameter")
        fields = post.fields()
        prompt = str(fields.get("prompt", ""))
        if "[fail]" in prompt:
            script = list(FAIL_SCRIPT)
        elif "[slow]" in prompt:
            script = [step for step in self.script[:-1] for _ in range(SLOW_HOLD)] + self.script[-1:]
        else:
            script = list(self.script)
        if fields.get("preview") not in ("true", True):
            script = [{k: v for k, v in step.items() if k != "preview"} for step in script]
        job = Job("video_" + secrets.token_hex(16), fields, script)
        self.jobs[job.id] = job
        return web.json_response(job.public())

    async def get_video(self, request):
        job_id = request.match_info["id"]
        self.gets.append(job_id)
        if denied := self._auth(request):
            return denied
        job = self.jobs.get(job_id)
        if job is None:
            return error(404, "video not found", "not_found")
        if job.script:
            step = job.script.pop(0)
            if step.get("_forget"):
                del self.jobs[job_id]
                return error(404, "video not found", "not_found")
            job.state.update(step)
        return web.json_response(job.public())

    async def get_content(self, request):
        job_id = request.match_info["id"]
        variant = request.query.get("variant", "video")
        self.contents.append((job_id, variant))
        self._log({"event": "GET content", "id": job_id, "variant": variant})
        if denied := self._auth(request):
            return denied
        job = self.jobs.get(job_id)
        if job is None:
            return error(404, "video not found", "not_found")
        public = job.public()
        available = public["status"] == "completed" if variant == "video" else public["preview"]
        if not available:
            return error(404, f"the {variant} of {job_id} is not available (status {public['status']})", "not_found")
        return web.Response(body=self.video(), content_type="video/mp4")

    async def delete_video(self, request):
        job_id = request.match_info["id"]
        self.deletes.append(job_id)
        self._log({"event": "DELETE /v1/videos", "id": job_id})
        if denied := self._auth(request):
            return denied
        job = self.jobs.pop(job_id, None)
        if job is None:
            return error(404, "video not found", "not_found")
        return web.json_response({"id": job_id, "object": "video.deleted", "deleted": True})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", help="append one JSON line per request to this file")
    parser.add_argument("--ready-delay", type=float, default=0.0, help="seconds to wait before listening")
    args = parser.parse_args(argv)
    if args.ready_delay:
        time.sleep(args.ready_delay)
    fake = FakeSplash(log_path=args.log)
    fake.video()
    print(f"fake splash listening on http://{args.host}:{args.port}", flush=True)
    web.run_app(fake.app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
