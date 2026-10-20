#!/usr/bin/env python3

"""Real video-model tests (make test-video-real, test-video-http-real).

engine: the video engine process of a prepared MiniMax-H3 model, driven over
the native protocol with the golden prompt's saved hidden states
(--hidden-states): the 3 s golden clip's final latents and video frames must
equal the references byte for byte; then a clip is cancelled at a step
boundary, and the engine serves on.

http: `python -m server.server` on the model, as `splash serve --model`
starts it: the golden clip through POST /v1/videos with the saved hidden
states, byte-identical as above, its preview and mp4 downloadable, chat
routes absent, a cancelled job, and then the clip of the server's own
tokenizer and text encoder, compared with the golden latents (relative RMS)
and frames (PSNR). With --seconds 15 it also runs one 15 s clip end to end.
Each report goes to --output as JSON: per-phase times and peak memory. With
--converted the test first prepares images converted before, as
prepare_local_model does, and serves them.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import select
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.tests.smoke_real import (  # noqa: E402
    SmokeFailure,
    available_port,
    request_headers,
    require,
)
from install import assembly, local, paths  # noqa: E402
from server import protocol as wire  # noqa: E402

GOLDEN_PROMPT = (
    "A golden retriever runs along a beach at sunset, waves crashing, camera "
    "tracking from the side. Sound of waves and the dog's footsteps on wet sand."
)
WIDTH, HEIGHT = 1344, 768
FRAMES = {3: 73, 10: 243, 15: 362}


class EngineProcess:
    """The video engine on its pipes, read by the native protocol."""

    def __init__(self, command):
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        self.parser = wire.FrameParser()
        self.pending = b""

    def send(self, message):
        self.process.stdin.write(wire.serialize_message(message))

    def event(self, timeout=3600.0):
        deadline = time.monotonic() + timeout
        while True:
            while self.pending:
                step = self.parser.consume(self.pending)
                self.pending = self.pending[step.consumed_bytes :]
                if step.issue:
                    raise SmokeFailure(step.issue.describe())
                if step.frame:
                    return wire.decode_frame(step.frame)
            ready, _, _ = select.select([self.process.stdout], [], [], 1.0)
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 1 << 20)
                require(bool(chunk), "the video engine closed its output")
                self.pending += chunk
            require(time.monotonic() < deadline, "the video engine stopped answering")

    def close(self):
        self.process.stdin.close()
        require(self.process.wait(timeout=120) == 0, "the video engine exited badly")


def clip_request(request_id, tokens, seconds=3, preview=True, seed=0):
    outputs = wire.VideoOutputFlag.AUDIO
    if preview:
        outputs |= wire.VideoOutputFlag.PREVIEW
    return wire.VideoRequestFrame(
        request_id,
        wire.VideoVariant.TURBO,
        outputs,
        FRAMES[seconds],
        WIDTH,
        HEIGHT,
        0,
        seed,
        tuple(tokens),
    )


def run_clip(engine, request, cancel_at_step=None):
    """The request's progress, by phase, and its outputs; the error that
    ended it instead, if any."""
    engine.send(request)
    phases, outputs = {}, {}
    while True:
        event = engine.event()
        if isinstance(event, wire.VideoProgressEvent):
            phase = phases.setdefault(event.phase.name.lower(), {})
            phase.update(
                steps=event.steps,
                seconds=event.elapsed_micros / 1e6,
                peak_footprint_bytes=event.peak_footprint_bytes,
            )
            if (
                cancel_at_step is not None
                and event.phase is wire.VideoPhase.DENOISE
                and event.step == cancel_at_step
            ):
                engine.send(wire.CancelFrame(request.request_id))
        elif isinstance(event, wire.VideoOutputEvent):
            outputs[event.output.name.lower()] = event
            if event.output is wire.VideoOutputKind.VIDEO:
                return phases, outputs, None
        elif isinstance(event, wire.ErrorEvent):
            return phases, outputs, event
        else:
            raise SmokeFailure(f"unexpected event {event}")


def compare_golden(keep, references):
    """Byte identity of a kept request's latents and frames."""
    final = np.load(keep / "final.npz")
    golden = np.load(references / "golden_3s_c5r1_both" / "final.npz")
    for name in ("video", "audio"):
        require(
            final[name].dtype == golden[name].dtype
            and final[name].tobytes() == golden[name].tobytes(),
            f"the {name} latents differ from the golden clip's",
        )
    frames = np.load(references / "golden_ref" / "frames.npy", mmap_mode="r")
    require(
        (keep / "frames.rgb").read_bytes() == np.ascontiguousarray(frames).tobytes(),
        "the video frames differ from the golden clip's",
    )


def run_engine(arguments):
    references = arguments.references
    golden = np.load(references / "prompts" / "golden.npz")
    tokens = [int(token) for token in golden["ids"]]
    with tempfile.TemporaryDirectory(prefix="splash-video-engine-") as keep:
        keep = Path(keep)
        engine = EngineProcess(
            [
                str(arguments.binary),
                "serve-native",
                str(arguments.model_root),
                "--hidden-states",
                str(references / "prompts" / "golden.npz"),
                "--keep",
                str(keep),
            ]
        )
        ready = engine.event(timeout=600)
        require(
            isinstance(ready, wire.ReadyEvent) and ready.video,
            "the engine did not announce a video model",
        )
        started = time.monotonic()
        phases, outputs, error = run_clip(engine, clip_request(1, tokens))
        require(error is None, f"the golden clip failed: {error}")
        require(
            set(outputs) == {"preview", "video"}
            and all(output.mp4[4:8] == b"ftyp" for output in outputs.values()),
            "the golden clip's preview and mp4 did not arrive",
        )
        compare_golden(keep / "1", references)
        wall = time.monotonic() - started
        cancelled_phases, _, error = run_clip(
            engine, clip_request(2, tokens, preview=False), cancel_at_step=2
        )
        require(
            error is not None and error.code == b"cancelled",
            "a cancelled clip did not end as cancelled",
        )
        require(
            cancelled_phases["denoise"]["steps"] == 8
            and "decode" not in cancelled_phases,
            "the cancelled clip went past its denoising",
        )
        engine.close()
    return {
        "golden_3s": {
            "identical": True,
            "wall_seconds": round(wall, 1),
            "phases": phases,
        },
        "cancel": {"cancelled_after_step": 2},
    }


class VideoServer:
    """`python -m server.server` of the model on a port of its own, its engine
    started with `engine_options` (a wrapper of the binary)."""

    def __init__(self, arguments, scratch, engine_options=()):
        binary = scratch / "engine"
        binary.write_text(
            f'#!/bin/sh\nexec "{arguments.binary.resolve()}" "$@" '
            + " ".join(f'"{option}"' for option in engine_options)
            + "\n"
        )
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        self.port = available_port()
        self.log = (scratch / "server.log").open("w")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "server.server", str(arguments.model_root),
             "--model", arguments.model, "--binary", str(binary),
             "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
        )  # fmt: skip
        deadline = time.monotonic() + 600
        while call(self.port, "GET", "/ready")[0] != 200:
            require(self.process.poll() is None, "the video server exited")
            require(
                time.monotonic() < deadline, "the video server did not become ready"
            )
            time.sleep(1)

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=180)
        self.log.close()


def call(port, method, path, body=None, timeout=60):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    payload = None if body is None else json.dumps(body).encode()
    try:
        connection.request(method, path, payload, request_headers(payload is not None))
        response = connection.getresponse()
        data = response.read()
        kind = response.getheader("Content-Type", "")
    except OSError:
        return 0, None
    finally:
        connection.close()
    return response.status, json.loads(data) if kind.startswith(
        "application/json"
    ) else data


class MemorySampler:
    """The host's used memory (active, wired and compressed pages), every
    second while a job runs."""

    def __init__(self):
        self.peak = 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped.wait(1.0):
            output = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = int(re.search(r"page size of (\d+)", output).group(1))
            pages = dict(re.findall(r'"?([A-Za-z -]+)"?:\s+(\d+)\.', output))
            used = sum(int(pages.get(name, 0)) for name in (
                "Pages active", "Pages wired down", "Pages occupied by compressor"))  # fmt: skip
            self.peak = max(self.peak, used * page)

    def stop(self):
        self.stopped.set()
        self.thread.join()
        return self.peak


def generate(port, body, timeout=3600):
    """A job of body, completed: its video object and mp4."""
    status, video = call(port, "POST", "/v1/videos", body)
    require(
        status == 200 and video["status"] == "queued",
        f"POST /v1/videos: {status} {video}",
    )
    progress, deadline = 0, time.monotonic() + timeout
    while video["status"] not in ("completed", "failed"):
        require(time.monotonic() < deadline, "the job did not finish in time")
        time.sleep(2)
        video = call(port, "GET", f"/v1/videos/{video['id']}")[1]
        require(video["progress"] >= progress, "the job's progress went back")
        progress = video["progress"]
    require(video["status"] == "completed", f"the job failed: {video['error']}")
    status, mp4 = call(port, "GET", f"/v1/videos/{video['id']}/content", timeout=120)
    require(status == 200 and mp4[4:8] == b"ftyp", "the mp4 did not download")
    return video, mp4


def relative_rms(got, want):
    got, want = got.astype(np.float64), want.astype(np.float64)
    return float(np.sqrt(np.mean((got - want) ** 2) / np.mean(want**2)))


def psnr(got, want):
    error = np.mean((got.astype(np.float64) - want.astype(np.float64)) ** 2)
    return float("inf") if error == 0 else float(10 * np.log10(255.0**2 / error))


def run_http(arguments):
    references = arguments.references
    golden = references / "prompts" / "golden.npz"
    report = {}
    with tempfile.TemporaryDirectory(prefix="splash-video-http-") as scratch:
        scratch = Path(scratch)
        keep = scratch / "keep"
        # The golden clip, of the saved hidden states: byte-identical.
        server = VideoServer(
            arguments, scratch, ("--hidden-states", golden, "--keep", keep)
        )
        try:
            for path in ("/v1/chat/completions", "/v1/responses"):
                status, answer = call(
                    server.port, "POST", path, {"model": arguments.model}
                )
                require(status == 404 and "video model" in answer["error"]["message"],
                        f"{path} is served by a video model's server")  # fmt: skip
            body = {"prompt": GOLDEN_PROMPT, "seconds": "3", "seed": 0, "preview": True}
            video, _ = generate(server.port, body)
            status, preview = call(
                server.port, "GET", f"/v1/videos/{video['id']}/content?variant=preview"
            )
            require(
                status == 200 and preview[4:8] == b"ftyp",
                "the preview did not download",
            )
            compare_golden(keep / "1", references)
            report["golden_3s"] = {"identical": True, "phases": video["phases"]}
            # A job deleted while it denoises stops at the next step.
            cancelled = call(server.port, "POST", "/v1/videos", body)[1]
            while (
                call(server.port, "GET", f"/v1/videos/{cancelled['id']}")[1]["progress"]
                < 10
            ):
                time.sleep(1)
            started = time.monotonic()
            require(call(server.port, "DELETE", f"/v1/videos/{cancelled['id']}")[0] == 200,
                    "DELETE refused the running job")  # fmt: skip
            while (
                call(server.port, "GET", "/status")[1]["videos"]["running"] is not None
            ):
                require(
                    time.monotonic() - started < 300, "the deleted job kept running"
                )
                time.sleep(1)
            report["cancel"] = {"stopped_seconds": round(time.monotonic() - started, 1)}
        finally:
            server.close()
        # The same clip of the server's tokenizer and the engine's text encoder.
        server = VideoServer(arguments, scratch, ("--keep", keep / "encoded"))
        try:
            sampler = MemorySampler()
            video, _ = generate(server.port, body)
            peak = sampler.stop()
            kept = keep / "encoded" / "1"
            hidden = np.load(kept / "hidden.npz")["hidden"]
            want = np.load(golden)
            require(hidden.shape == want["hidden"].shape, "the prompt's tokens differ")
            final = np.load(kept / "final.npz")
            reference = np.load(references / "golden_3s_c5r1_both" / "final.npz")
            frames = np.frombuffer((kept / "frames.rgb").read_bytes(), np.uint8)
            golden_frames = np.load(references / "golden_ref" / "frames.npy").reshape(
                -1
            )
            report["encoder_3s"] = {
                "hidden_relative_rms": relative_rms(hidden, want["hidden"]),
                "video_latent_relative_rms": relative_rms(
                    final["video"], reference["video"]
                ),
                "audio_latent_relative_rms": relative_rms(
                    final["audio"], reference["audio"]
                ),
                "frames_psnr_db": psnr(frames, golden_frames),
                "phases": video["phases"],
                "host_used_peak_bytes": peak,
            }
            if arguments.seconds == 15:
                sampler = MemorySampler()
                video, mp4 = generate(
                    server.port, {**body, "seconds": "15"}, timeout=7200
                )
                report["clip_15s"] = {
                    "phases": video["phases"],
                    "mp4_bytes": len(mp4),
                    "host_used_peak_bytes": sampler.stop(),
                }
        finally:
            server.close()
    return report


def prepare(arguments):
    """Prepares the converted images as prepare_local_model does, twice: the
    second reads only their metadata."""
    times = []
    for _ in range(2):
        started = time.monotonic()
        arguments.model_root = local.prepare_local_model(
            arguments.converted, models_dir=arguments.models_dir
        )
        times.append(round(time.monotonic() - started, 2))
    record = assembly.verify(arguments.model_root)
    require(record["target_format"] == "h3-video", "the images prepared no video model")
    return {
        "first_seconds": times[0],
        "again_seconds": times[1],
        "files": len(record["files"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("engine", "http"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-root", type=Path, help="a prepared MiniMax-H3")
    source.add_argument(
        "--converted",
        type=Path,
        help="images converted before, prepared (prepare_local_model) in "
        "--models-dir and served",
    )
    parser.add_argument("--models-dir", type=Path, default=paths.MODELS)
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / "splash")
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="the h3 engine's references (golden prompt, latents and frames)",
    )
    parser.add_argument("--seconds", type=int, choices=(3, 15), default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = {}
        if arguments.converted:
            report["prepare"] = prepare(arguments)
        run = run_engine if arguments.mode == "engine" else run_http
        report |= run(arguments)
    except SmokeFailure as error:
        print(f"video {arguments.mode} test: FAIL: {error}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n")
    print(text)
    print(f"video {arguments.mode} test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
