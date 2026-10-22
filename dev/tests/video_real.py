#!/usr/bin/env python3

"""Real video-model tests (make test-video-real, test-video-http-real,
test-video-first-frame-real) on the h3 engine's references (--references,
its H3_REFS), named and compared as its dev/tests/references.sh and
smoke_golden.sh do.

engine: the engine, on GOLDEN_PROMPT_NPZ's hidden states, makes the 3 s
golden clip, GOLDEN_FINAL and GOLDEN_FRAMES byte for byte; a clip cancelled
at denoising step 2 ends there. http: the same clip through POST /v1/videos,
a job deleted while it denoises, and a clip of --seconds from the server's
own tokenizer and text encoder. first-frame: the 3 s golden clip from
GOLDEN_FRAMES' first frame, a PNG sent as input_reference, makes the h3
engine's latents of the same inputs (--first-frame-final) byte for byte.
"""

import argparse
import contextlib
import filecmp
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import openai
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.tests.smoke_real import (  # noqa: E402
    SmokeFailure,
    available_port,
    request_headers,
    require,
)
from install import assembly, local  # noqa: E402
from server import protocol as wire  # noqa: E402
from server import videos  # noqa: E402

GOLDEN_PROMPT_NPZ = "prompts/golden.npz"
GOLDEN_FINAL = "golden_3s_c5r1_both/final.npz"
GOLDEN_FRAMES = "golden_ref/frames.npy"
PREVIEW = {"preview": "true"}
GOLDEN_PROMPT = (
    "A golden retriever runs along a beach at sunset, waves crashing, camera "
    "tracking from the side. Sound of waves and the dog's footsteps on wet sand."
)


def require_golden(kept: Path, references: Path):
    """The latents and frames an engine kept (--keep) are the golden clip's."""
    final = references / GOLDEN_FINAL
    require(filecmp.cmp(kept / "final.npz", final, shallow=False),
            f"the latents differ from {final}")  # fmt: skip
    frames = np.load(references / GOLDEN_FRAMES, mmap_mode="r").tobytes()
    require((kept / "frames.rgb").read_bytes() == frames,
            f"the frames differ from {references / GOLDEN_FRAMES}")  # fmt: skip


def run_engine(arguments, scratch):
    hidden = arguments.references / GOLDEN_PROMPT_NPZ
    engine = videos.VideoEngine(
        [arguments.binary, "serve-native", arguments.model_root,
         "--hidden-states", hidden, "--keep", scratch],
        videos.START_SECONDS,
    )  # fmt: skip
    engine.start()
    tokens = tuple(int(token) for token in np.load(hidden)["ids"])
    phases, previews = {}, []

    def note(event, cancel_at=None):
        if isinstance(event, wire.VideoOutputEvent):
            previews.append(event.mp4)
            return
        phases[event.phase.name.lower()] = event.elapsed_micros / 1e6
        if event.phase is wire.VideoPhase.DENOISE and event.step == cancel_at:
            engine.cancel()

    def clip(preview):
        request = {"prompt": GOLDEN_PROMPT, "preview": preview}
        return videos.clip(videos.parse_request(request, (arguments.model,)), tokens)

    try:
        output = engine.run(clip(True), note)
        require(output.mp4[4:8] == b"ftyp" and previews[0][4:8] == b"ftyp",
                "the golden clip's mp4 or preview did not arrive")  # fmt: skip
        require_golden(scratch / "1", arguments.references)
        report = {"golden_3s": {"identical": True, "phases": dict(phases)}}
        phases.clear()
        try:
            engine.run(clip(False), lambda event: note(event, cancel_at=2))
            cancelled = None
        except videos.VideoFailed as error:
            cancelled = error.code
        require(cancelled == "cancelled" and "decode" not in phases,
                f"the clip cancelled at step 2 went on: {cancelled}, {phases}")  # fmt: skip
    finally:
        engine.close()
    require(engine.process.returncode == 0, "the engine exited badly")
    return report | {"cancel": {"cancelled_after_step": 2}}


@contextlib.contextmanager
def serving(arguments, scratch, *engine_options):
    """An OpenAI client of `python -m server.server` on the model, as
    `splash serve` starts it, and its port; its engine takes engine_options."""
    engine = scratch / "engine"
    command = shlex.join(map(str, engine_options))
    engine.write_text(
        f'#!/bin/sh\nexec "{arguments.binary.resolve()}" "$@" {command}\n'
    )
    engine.chmod(0o755)
    port = available_port()
    log = (scratch / "server.log").open("a")
    process = subprocess.Popen(
        [sys.executable, "-m", "server.server", arguments.model_root,
         "--model", arguments.model, "--binary", engine, "--port", str(port)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 600
        while status(port, "/ready") is None:
            require(process.poll() is None and time.monotonic() < deadline,
                    "the video server did not become ready")  # fmt: skip
            time.sleep(1)
        key = os.environ.get("SPLASH_API_KEY", "unused")
        yield openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=key), port
    finally:
        process.terminate()
        process.wait(timeout=180)
        log.close()


def status(port, path):
    with contextlib.suppress(httpx.HTTPError):
        url, headers = f"http://127.0.0.1:{port}{path}", request_headers(False)
        return httpx.get(url, headers=headers, timeout=60).raise_for_status().json()


def create(client, seconds):
    return client.videos.create(
        prompt=GOLDEN_PROMPT, seconds=seconds, extra_body=PREVIEW
    )


def generate(client, seconds):
    """A completed job of the golden prompt, its mp4 downloaded."""
    video = client.videos.poll(create(client, seconds).id, poll_interval_ms=2000)
    require(video.status == "completed", f"the job failed: {video.error}")
    mp4 = client.videos.download_content(video.id).read()
    require(mp4[4:8] == b"ftyp", "the mp4 did not download")
    return video


def run_http(arguments, scratch):
    references, report = arguments.references, {}
    hidden, kept = references / GOLDEN_PROMPT_NPZ, scratch / "golden"
    golden = ("--hidden-states", hidden, "--keep", kept)
    with serving(arguments, scratch, *golden) as (client, port):
        video = generate(client, "3")
        preview = client.videos.download_content(video.id, variant="preview").read()
        require(preview[4:8] == b"ftyp", "the preview did not download")
        require_golden(kept / "1", references)
        report["golden_3s"] = {"identical": True, "phases": video.phases}
        # A job deleted while it denoises stops at its next step.
        job = create(client, "3")
        while client.videos.retrieve(job.id).progress < 10:
            time.sleep(1)
        started = time.monotonic()
        client.videos.delete(job.id)
        while status(port, "/status")["videos"]["running"] is not None:
            require(time.monotonic() - started < 300, "the deleted job kept running")
            time.sleep(1)
        report["cancel"] = {"stopped_seconds": round(time.monotonic() - started, 1)}
    with serving(arguments, scratch, "--keep", scratch / "encoded") as (client, _):
        video = generate(client, str(arguments.seconds))
        encoded = np.load(scratch / "encoded" / "1" / "hidden.npz")["hidden"]
        want = np.load(hidden)["hidden"]
        require(encoded.shape == want.shape, "the prompt's tokens differ from h3's")
        report[f"encoder_{arguments.seconds}s"] = {"phases": video.phases}
    return report


def run_first_frame(arguments, scratch):
    """The 3 s golden clip from the golden frames' first frame: its latents
    are the h3 engine's of the same prompt, seed and first frame."""
    first = scratch / "first.png"
    frames = np.load(arguments.references / GOLDEN_FRAMES, mmap_mode="r")
    Image.fromarray(frames[0]).save(first)
    hidden, kept = arguments.references / GOLDEN_PROMPT_NPZ, scratch / "first-frame"
    golden = ("--hidden-states", hidden, "--keep", kept)
    with serving(arguments, scratch, *golden) as (client, _):
        with first.open("rb") as image:
            job = client.videos.create(
                prompt=GOLDEN_PROMPT, seconds="3", input_reference=image
            )
        video = client.videos.poll(job.id, poll_interval_ms=2000)
        require(video.status == "completed", f"the job failed: {video.error}")
    final = arguments.first_frame_final
    require(filecmp.cmp(kept / "1" / "final.npz", final, shallow=False),
            f"the latents differ from {final}")  # fmt: skip
    return {"first_frame_3s": {"identical": True, "phases": video.phases}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("engine", "http", "first-frame"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-root", type=Path, help="a prepared MiniMax-H3")
    source.add_argument("--converted", type=Path, help="images converted before")
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / "splash")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--seconds", type=int, choices=(3, 10, 15), default=3)
    parser.add_argument("--first-frame-final", type=Path,
                        help="the h3 engine's final.npz of the 3 s golden clip from golden frame 0")  # fmt: skip
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.converted:
            arguments.model_root = local.prepare_local_model(arguments.converted)
        record = assembly.verify(arguments.model_root)
        require(record["target_format"] == "h3-video", "the model is not MiniMax-H3")
        if arguments.mode == "first-frame":
            require(arguments.first_frame_final and arguments.first_frame_final.is_file(),
                    "first-frame takes --first-frame-final, the h3 engine's latents")  # fmt: skip
        modes = {"engine": run_engine, "http": run_http, "first-frame": run_first_frame}
        run = modes[arguments.mode]
        with tempfile.TemporaryDirectory(prefix="splash-video-") as scratch:
            report = run(arguments, Path(scratch))
    except (SmokeFailure, videos.VideoFailed) as error:
        print(f"video {arguments.mode} test: FAIL: {error}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n")
    print(f"{text}\nvideo {arguments.mode} test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
