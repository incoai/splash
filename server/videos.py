"""A video model's server: OpenAI-shaped /v1/videos jobs on the video engine.

`python -m server.server` of a video model's assembly (install/video.py)
serves these routes alone, beside /health, /ready, /status and /v1/models:
POST /v1/videos, GET /v1/videos, GET /v1/videos/{id},
GET /v1/videos/{id}/content and DELETE /v1/videos/{id}. A language model's
routes answer 404: a language model is served by an instance of its own.
The engine process (runtime/video/VideoRuntime.hpp) runs one job at a time
on the GPU, the others wait in a short queue in their order of arrival. A
job's progress follows the engine's phases, its denoising step by step; its
preview and mp4 stay in a store of bounded bytes, which evicts the oldest
finished job first; DELETE cancels a job, a running one at its next step.

`python -m server.videos`, which `splash generate` starts, runs one clip on
the engine instead and writes its mp4 (main, generate).
"""

from __future__ import annotations

import argparse
import email.parser
import email.policy
import itertools
import json
import os
import re
import secrets
import select
import signal
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs

from install import assembly, models, video

from . import protocol as wire
from .diagnostics import print_status
from .errors import APIError

# OpenAI's `seconds` names a clip; MiniMax-H3 renders 3, 10 and 15 s clips of
# 17 n + 5 frames at 24 fps.
CLIP_FRAMES = {"3": 73, "10": 243, "15": 362}
SIZE = "1344x768"
VARIANTS = {"turbo": wire.VideoVariant.TURBO, "default": wire.VideoVariant.DEFAULT}
FIELDS = {
    "prompt",
    "model",
    "seconds",
    "size",
    "variant",
    "seed",
    "steps",
    "preview",
    "audio",
}
# The jobs that may wait for the engine, and the bytes of finished jobs' mp4s
# kept for download: the oldest finished job goes first.
MAX_QUEUED_JOBS = 4
STORE_BUDGET_BYTES = 4 << 30
MAX_LISTED_JOBS = 100
# A job's progress once each phase has completed; the denoising's grows with
# its steps.
PHASE_PROGRESS = {
    wire.VideoPhase.ENCODE: 2,
    wire.VideoPhase.PREPARE: 4,
    wire.VideoPhase.DENOISE: 90,
    wire.VideoPhase.PREVIEW: 91,
    wire.VideoPhase.DECODE: 97,
    wire.VideoPhase.AUDIO: 98,
    wire.VideoPhase.MUX: 100,
}
# How long a stopping engine may take to end its request, and a starting one
# to announce Ready, as the server waits for a language model's.
CLOSE_SECONDS = 30.0
START_SECONDS = 600.0
JOB_ID = re.compile(r"/v1/videos/(video_[0-9a-f]{32})(/content)?")


def serves(model_root: Path):
    """Whether the assembly at model_root holds a video model."""
    record = assembly.recorded(model_root)
    return record is not None and record["target_format"] == video.FORMAT


class VideoFailed(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    seconds: str
    variant: str
    seed: int
    steps: int
    preview: bool
    audio: bool


def _integer(fields, name, default, maximum):
    value = fields.get(name, default)
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if type(value) is not int or not 0 <= value <= maximum:
        raise APIError(400, f"{name} must be an integer in [0, {maximum}]")
    return value


def _boolean(fields, name, default):
    value = fields.get(name, default)
    if isinstance(value, str) and value in ("true", "false"):
        value = value == "true"
    if not isinstance(value, bool):
        raise APIError(400, f"{name} must be a boolean")
    return value


def parse_request(fields, names):
    """A POST /v1/videos body's fields (JSON, or form fields, all strings)
    as a VideoRequest; `names` are the names the model answers to."""
    if "input_reference" in fields:
        raise APIError(400, "input_reference is not supported", "unsupported_parameter")
    if unknown := sorted(set(fields) - FIELDS):
        raise APIError(
            400, f"unsupported parameter: {unknown[0]}", "unsupported_parameter"
        )
    prompt = fields.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise APIError(400, "prompt must be a non-empty string")
    if fields.get("model", names[0]) not in names:
        raise APIError(
            404, f"model {fields['model']!r} is not served", "model_not_found"
        )
    seconds = str(fields.get("seconds", "3"))
    if seconds not in CLIP_FRAMES:
        raise APIError(400, "seconds must be one of " + ", ".join(CLIP_FRAMES))
    if fields.get("size", SIZE) != SIZE:
        raise APIError(400, f"size must be {SIZE}")
    variant = fields.get("variant", "turbo")
    if variant not in VARIANTS:
        raise APIError(400, "variant must be turbo or default")
    return VideoRequest(
        prompt,
        seconds,
        variant,
        _integer(fields, "seed", 0, 2**31 - 1),
        _integer(fields, "steps", 0, 1000),
        _boolean(fields, "preview", False),
        _boolean(fields, "audio", True),
    )


def prompt_tokens(tokenizer, prompt, ready):
    """The prompt's token ids, as the text encoder of the engine that
    announced ready takes them."""
    tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not 2 <= len(tokens) <= ready.max_context_tokens:
        raise APIError(
            400,
            f"the prompt is {len(tokens)} tokens; the text encoder takes 2 to "
            f"{ready.max_context_tokens}",
        )
    return tuple(tokens)


def clip(request, tokens):
    """A VideoRequestFrame's fields but its id: the clip of request."""
    outputs = wire.VideoOutputFlag(0)
    if request.preview:
        outputs |= wire.VideoOutputFlag.PREVIEW
    if request.audio:
        outputs |= wire.VideoOutputFlag.AUDIO
    width, height = (int(n) for n in SIZE.split("x"))
    return (VARIANTS[request.variant], outputs, CLIP_FRAMES[request.seconds], width,
            height, request.steps, request.seed, tokens)  # fmt: skip


def load(model_root, binary, startup_timeout):
    """The engine of the assembly at model_root, started and ready, and the
    tokenizer its prompts are encoded with."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_root / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    engine = VideoEngine(
        [str(binary), "serve-native", str(model_root)], startup_timeout
    )
    engine.start()
    return engine, tokenizer


def form_fields(content_type, payload):
    """The text fields of a multipart/form-data body, as OpenAI's SDK sends
    a video's; a file field is refused."""
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + bytes(payload)
    )
    if not message.is_multipart():
        raise APIError(400, "invalid multipart/form-data body")
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name is None or part.get_filename() is not None:
            raise APIError(
                400, f"{name or 'a field'} is not supported", "unsupported_parameter"
            )
        fields[name] = part.get_content().strip()
    return fields


@dataclass
class Job:
    id: str
    model: str
    request: VideoRequest
    # The prompt's tokens, as the text encoder takes them.
    tokens: tuple
    created_at: int = field(default_factory=lambda: int(time.time()))
    status: str = "queued"
    progress: int = 0
    phase: wire.VideoPhase | None = None
    completed_at: int | None = None
    error: dict | None = None
    # Each phase's seconds and the engine's peak footprint at its end.
    phases: dict = field(default_factory=dict)
    preview: bytes | None = None
    video: bytes | None = None

    @property
    def finished(self):
        return self.status in ("completed", "failed")

    @property
    def stored_bytes(self):
        return len(self.preview or b"") + len(self.video or b"")

    def public(self):
        """The video object, with OpenAI's fields and Splash's beside them:
        its phase, its request's options and each phase's timing."""
        return {
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "expires_at": None,
            "size": SIZE,
            "seconds": self.request.seconds,
            "remixed_from_video_id": None,
            "error": self.error,
            "phase": self.phase.name.lower() if self.phase else None,
            "variant": self.request.variant,
            "seed": self.request.seed,
            "preview": self.preview is not None,
            "phases": self.phases,
        }


class VideoEngine:
    """The video engine process on the native protocol: started with the
    server, and again for the next job once it has exited."""

    def __init__(self, command, startup_timeout):
        self.command = command
        self.startup_timeout = startup_timeout
        self.process = None
        self.parser = None
        self.pending = b""
        self.ready = None
        self.request_ids = itertools.count(1)
        # Writes to the engine, and the request in flight and whether it is
        # cancelled, which a cancel from another thread reads.
        self.lock = threading.Lock()
        self.current = None
        self.cancelled = False

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        """Start the engine unless it runs, and wait for its Ready."""
        if self.alive():
            return
        if self.process is not None:
            self.close()
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        self.parser, self.pending = wire.FrameParser(), b""
        ready = self._event(time.monotonic() + self.startup_timeout)
        if not isinstance(ready, wire.ReadyEvent) or not ready.video:
            raise VideoFailed("video_engine_failed", "the engine serves no video model")
        self.ready = ready

    def run(self, request, on_event):
        """The clip of request, a VideoRequestFrame's fields but its id: its
        VideoOutputEvent once every other event went to on_event; raises
        VideoFailed with the engine's error. A cancel() before it is sent
        cancels it as it is sent."""
        try:
            self.start()
            frame = wire.VideoRequestFrame(next(self.request_ids), *request)
            with self.lock:
                self.current = frame.request_id
                self._send(frame)
                if self.cancelled:
                    self._send(wire.CancelFrame(frame.request_id))
            while True:
                event = self._event(None)
                if isinstance(event, wire.ErrorEvent):
                    if event.failure_class is not wire.FailureClass.REQUEST_ERROR:
                        self.close()
                    raise VideoFailed(event.code.decode(), event.message.decode())
                if (
                    isinstance(event, wire.VideoOutputEvent)
                    and event.output is wire.VideoOutputKind.VIDEO
                ):
                    return event
                on_event(event)
        finally:
            with self.lock:
                self.current, self.cancelled = None, False

    def cancel(self):
        """Cancel the request in flight, or the next one run() sends."""
        with self.lock:
            self.cancelled = True
            if self.current is not None:
                self._send(wire.CancelFrame(self.current))

    def stop(self):
        """End the engine's input: it cancels its request and exits, or is
        killed after CLOSE_SECONDS."""
        if self.alive():
            with self.lock:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=CLOSE_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def close(self):
        """Stop the engine and close its output, once nothing reads it."""
        self.stop()
        if self.process is not None:
            self.process.stdout.close()

    def _send(self, message):
        """Write a frame; call it with the lock held."""
        try:
            self.process.stdin.write(wire.serialize_message(message))
        except (OSError, ValueError) as error:
            raise VideoFailed(
                "video_engine_failed", f"the engine stopped: {error}"
            ) from None

    def _event(self, deadline):
        """The engine's next event, by deadline (time.monotonic) if given."""
        while True:
            while self.pending:
                step = self.parser.consume(self.pending)
                self.pending = self.pending[step.consumed_bytes :]
                if step.issue:
                    raise VideoFailed("video_engine_failed", step.issue.describe())
                if step.frame:
                    return wire.decode_frame(step.frame)
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not select.select([self.process.stdout], [], [], wait)[0]:
                raise VideoFailed(
                    "video_engine_failed", "the engine did not start in time"
                )
            chunk = os.read(self.process.stdout.fileno(), 1 << 20)
            if not chunk:
                status = self.process.wait()
                raise VideoFailed(
                    "video_engine_failed", f"the engine exited with status {status}"
                )
            self.pending += chunk


class VideoService:
    """Admits jobs, runs them one at a time on the engine, and keeps their
    results within the store's budget."""

    def __init__(self, engine, tokenizer, names, *, budget_bytes=STORE_BUDGET_BYTES):
        self.engine = engine
        self.tokenizer = tokenizer
        self.names = names
        self.budget_bytes = budget_bytes
        self.jobs = OrderedDict()
        self.queue = []
        self.running = None
        self.evictions = 0
        self.closing = False
        self.lock = threading.Condition()
        self.worker = threading.Thread(
            target=self._work, name="splash-videos", daemon=True
        )
        self.worker.start()

    def submit(self, fields):
        request = parse_request(fields, self.names)
        tokens = prompt_tokens(self.tokenizer, request.prompt, self.engine.ready)
        with self.lock:
            if self.closing:
                raise APIError(503, "the server is shutting down", "server_shutdown")
            if len(self.queue) >= MAX_QUEUED_JOBS:
                raise APIError(
                    503,
                    f"{MAX_QUEUED_JOBS} video jobs are already queued",
                    "video_queue_full",
                )
            job = Job("video_" + secrets.token_hex(16), self.names[0], request, tokens)
            self.jobs[job.id] = job
            self.queue.append(job)
            self.lock.notify_all()
            return job.public()

    def get(self, job_id):
        with self.lock:
            return self._job(job_id).public()

    def list(self, query):
        try:
            limit = int(query.get("limit", 20))
        except ValueError:
            raise APIError(400, "limit must be an integer") from None
        if not 1 <= limit <= MAX_LISTED_JOBS:
            raise APIError(400, f"limit must be in [1, {MAX_LISTED_JOBS}]")
        order = query.get("order", "desc")
        if order not in ("asc", "desc"):
            raise APIError(400, "order must be asc or desc")
        with self.lock:
            jobs = list(self.jobs.values())
            if order == "desc":
                jobs.reverse()
            if (after := query.get("after")) is not None:
                ids = [job.id for job in jobs]
                if after not in ids:
                    raise APIError(404, "video not found", "not_found")
                jobs = jobs[ids.index(after) + 1 :]
            data = [job.public() for job in jobs[:limit]]
        return {
            "object": "list",
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": len(jobs) > limit,
        }

    def content(self, job_id, variant):
        """The mp4 of a completed job, or its preview once decoded."""
        if variant not in ("video", "preview"):
            raise APIError(
                400, "variant must be video or preview", "unsupported_parameter"
            )
        with self.lock:
            job = self._job(job_id)
            data = job.preview if variant == "preview" else job.video
        if data is None:
            raise APIError(
                404 if job.finished else 409,
                f"the {variant} of {job_id} is not available (status {job.status})",
                "video_not_ready",
            )
        return data

    def delete(self, job_id):
        """Forgets a job, cancelling it if it has not finished."""
        with self.lock:
            job = self._job(job_id)
            del self.jobs[job_id]
            if job in self.queue:
                self.queue.remove(job)
            elif job is self.running:
                self.engine.cancel()
        return {"id": job_id, "object": "video.deleted", "deleted": True}

    def status(self):
        with self.lock:
            finished = [job for job in self.jobs.values() if job.finished]
            return {
                "queued": len(self.queue),
                "running": self.running.id if self.running else None,
                "stored": len(finished),
                "stored_bytes": sum(job.stored_bytes for job in finished),
                "budget_bytes": self.budget_bytes,
                "evictions": self.evictions,
                "engine": "ready" if self.engine.alive() else "stopped",
            }

    def close(self):
        """Drop the queued jobs and stop the engine, which ends the running
        one."""
        with self.lock:
            self.closing = True
            self.queue.clear()
            self.lock.notify_all()
        self.engine.stop()
        self.worker.join()
        self.engine.close()

    def _job(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            raise APIError(404, "video not found", "not_found")
        return job

    def _work(self):
        while True:
            with self.lock:
                while not self.queue and not self.closing:
                    self.lock.wait()
                if self.closing:
                    return
                job = self.running = self.queue.pop(0)
                job.status = "in_progress"
            self._run(job)
            with self.lock:
                self.running = None
                self._evict()

    def _run(self, job):
        try:
            output = self.engine.run(
                clip(job.request, job.tokens), lambda event: self._report(job, event)
            )
        except VideoFailed as error:
            self._finish(job, None, {"code": error.code, "message": str(error)})
        else:
            self._finish(job, output, None)

    def _report(self, job, event):
        with self.lock:
            if isinstance(event, wire.VideoOutputEvent):
                job.preview = event.mp4
            else:
                job.phase = event.phase
                start = (
                    0
                    if event.phase == 1
                    else PHASE_PROGRESS[wire.VideoPhase(event.phase - 1)]
                )
                share = (PHASE_PROGRESS[event.phase] - start) * event.step / event.steps
                job.progress = max(job.progress, int(start + share))
                if event.step == event.steps:
                    job.phases[event.phase.name.lower()] = {
                        "seconds": round(event.elapsed_micros / 1e6, 3),
                        "peak_footprint_bytes": event.peak_footprint_bytes,
                    }

    def _finish(self, job, output, error):
        with self.lock:
            job.status = "failed" if error else "completed"
            job.error, job.phase = error, None
            job.completed_at = int(time.time())
            if output is not None:
                job.video, job.progress = output.mp4, 100
                job.phases["total"] = {"seconds": round(output.wall_micros / 1e6, 3)}
            print_status(
                f"Video · {job.id} · {job.status}"
                + (
                    f" · {job.phases['total']['seconds']} s"
                    if output
                    else f" · {error['code']}"
                )
            )

    def _evict(self):
        """Drop the oldest finished jobs past the store's budget."""
        finished = [job for job in self.jobs.values() if job.finished]
        total = sum(job.stored_bytes for job in finished)
        for job in finished:
            if total <= self.budget_bytes:
                break
            total -= job.stored_bytes
            del self.jobs[job.id]
            self.evictions += 1


class VideoApp:
    """What a video model's server serves (FrontendServer.app): the model
    under its names, and its jobs on the engine."""

    def __init__(self, model, names, engine, tokenizer):
        self.model = model
        self.names = tuple(dict.fromkeys((model, *names)))
        self.videos = VideoService(engine, tokenizer, self.names)

    @classmethod
    def start(cls, model, names, model_root, binary, startup_timeout):
        """The app of the assembly at model_root, its engine started and
        ready, and its tokenizer loaded."""
        return cls(model, names, *load(model_root, binary, startup_timeout))

    def status(self):
        return {"videos": self.videos.status()}

    def close(self):
        self.videos.close()


def handler(base):
    """The request handler of a video model's server: base's HTTP handling
    (FrontendHandler), its own routes."""

    class VideoHandler(base):
        def do_GET(self):
            path = self.route
            if path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/ready":
                ready = self.app.videos.engine.alive()
                self._json(
                    200 if ready else 503,
                    {"status": "ready" if ready else "unavailable"},
                )
            elif path == "/status":
                self._json(200, self.server.status())
            elif path == "/v1/models" or path.startswith("/v1/models/"):
                self._models(path)
            elif path == "/v1/videos":
                self._answer(
                    lambda: self._json(200, self.app.videos.list(self._query()))
                )
            elif match := JOB_ID.fullmatch(path):
                job_id, content = match.groups()
                self._answer(
                    lambda: (
                        self._send(
                            200,
                            self.app.videos.content(
                                job_id, self._query().get("variant", "video")
                            ),
                            "video/mp4",
                        )
                        if content
                        else self._json(200, self.app.videos.get(job_id))
                    )
                )
            else:
                self._absent()

        def do_POST(self):
            if self.route != "/v1/videos":
                self._absent()
                return
            self._body_reservation = None
            try:
                self._answer(
                    lambda: self._json(200, self.app.videos.submit(self._fields()))
                )
            finally:
                if self._body_reservation is not None:
                    self._body_reservation.release()

        def do_DELETE(self):
            if match := JOB_ID.fullmatch(self.route):
                if not match.group(2):
                    self._answer(
                        lambda: self._json(200, self.app.videos.delete(match.group(1)))
                    )
                    return
            self._absent()

        def _absent(self):
            self._safe_error(
                APIError(
                    404,
                    f"not found: this server serves the video model {self.app.model} "
                    "at /v1/videos; a language model is served by a server of its own",
                    "not_found",
                )
            )

        def _answer(self, respond):
            try:
                respond()
            except APIError as error:
                self._safe_error(error)
            except TimeoutError:
                self._safe_error(APIError(408, "HTTP I/O timed out", "request_timeout"))

        def _query(self):
            return {
                key: values[-1]
                for key, values in parse_qs(self.path.partition("?")[2]).items()
            }

        def _fields(self):
            """The POST body's fields: JSON, or OpenAI's SDK's multipart form."""
            deadline = time.monotonic() + 600.0
            content_type = self.headers.get("Content-Type", "")
            if self.headers.get_content_type() == "multipart/form-data":
                return form_fields(
                    content_type, self._read_body(deadline, ("multipart/form-data",))
                )
            body = self._read_json_body(deadline)
            return {key: value for key, value in body.items() if value is not None}

        def _models(self, path):
            entries = [
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "splash",
                    "input_modalities": ["text"],
                    "output_modalities": ["video", "audio"],
                }
                for name in self.app.names
            ]
            if path == "/v1/models":
                self._json(200, {"object": "list", "data": entries})
            elif entry := next(
                (e for e in entries if path == f"/v1/models/{e['id']}"), None
            ):
                self._json(200, entry)
            else:
                self._safe_error(APIError(404, "model not found", "model_not_found"))

    return VideoHandler


def generate(engine, tokenizer, request, out: Path):
    """Run request on the ready engine as a job runs, its progress on
    stderr, and write its mp4 to out; its preview, when asked, to
    <out stem>.preview.mp4 as soon as it is decoded. Raises VideoFailed with
    the engine's error, APIError for a prompt the encoder refuses."""
    tokens = prompt_tokens(tokenizer, request.prompt, engine.ready)
    preview = out.with_suffix(".preview.mp4")

    def report(event):
        if isinstance(event, wire.VideoOutputEvent):
            preview.write_bytes(event.mp4)
            print_status(f"Video · preview · {preview}", error=True)
        else:
            print_status(
                f"Video · {event.phase.name.lower()} · step {event.step}/"
                f"{event.steps} · {event.elapsed_micros / 1e6:.1f} s",
                error=True,
            )

    output = engine.run(clip(request, tokens), report)
    out.write_bytes(output.mp4)
    print_status(
        f"Video · {out} · {output.frames} frames · {output.wall_micros / 1e6:.1f} s",
        error=True,
    )


def _stop(number, _frame):
    raise KeyboardInterrupt(number)


def main(argv=None):
    """One clip of the video model at MODEL_ROOT: what `splash generate`
    starts, with the POST /v1/videos fields its flags give. Exits 0 once the
    mp4 is written, 1 on a refusal or the engine's error, and 128 + the
    signal's number on SIGINT or SIGTERM, which cancel the clip at its next
    step."""
    parser = argparse.ArgumentParser(prog="python -m server.videos")
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request", type=json.loads, required=True)
    args = parser.parse_args(argv)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, _stop)
    engine = None
    try:
        # The launcher blocks both across its exec.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        request = parse_request(args.request, (args.model,))
        if not args.out.parent.is_dir():
            raise APIError(400, f"{args.out.parent} is not a directory")
        # Nothing reads `held`: it keeps the assembly from collection until
        # this process exits, as the server's does.
        model_root, held = assembly.hold(args.model_root)
        if not serves(model_root):
            raise APIError(
                400,
                f"{args.model} is a language model: splash generate takes a video "
                "model, such as MiniMaxAI/MiniMax-H3; splash serve serves this one",
            )
        print_status(f"Loading · {args.model}", error=True)
        engine, tokenizer = load(model_root, args.binary, START_SECONDS)
        failure = []

        def run():
            try:
                generate(engine, tokenizer, request, args.out)
            except Exception as error:
                failure.append(error)

        # The clip runs beside this thread, which takes the stop signals and
        # cancels it.
        worker = threading.Thread(target=run, name="splash-generate", daemon=True)
        worker.start()
        try:
            worker.join()
        except KeyboardInterrupt:
            print_status("Video · cancelling at the next step", error=True)
            engine.cancel()
            worker.join()
            raise
        if failure:
            raise failure[0]
        return 0
    except (APIError, VideoFailed, models.ModelError, OSError) as error:
        print_status(f"Error · {error}", error=True)
        return 1
    except KeyboardInterrupt as stop:
        return 128 + (stop.args[0] if stop.args else signal.SIGINT)
    finally:
        # A stop during cleanup leaves it to end the engine.
        for number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(number, signal.SIG_IGN)
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
