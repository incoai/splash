"""A video model's server (server/videos.py) over a stand-in engine
(dev/tests/fake_video_engine.py): a job through OpenAI's SDK, one from its
first and last frames, the bounds of the jobs' queue and store, and the one
clip `splash generate` runs."""

import contextlib
import hashlib
import io
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx
import openai
from PIL import Image

from server import server as api
from server import videos

ENGINE = Path(__file__).resolve().parents[1] / "fake_video_engine.py"
MODEL = "MiniMaxAI/MiniMax-H3"
SEVEN = (7).to_bytes(8, "little")
# A first frame of the canvas's size, which goes to the engine as it is.
CANVAS = bytes(range(256)) * (1344 * 768 * 3 // 256)


def png(width, height, pixels):
    image = io.BytesIO()
    Image.frombytes("RGB", (width, height), pixels).save(image, "PNG")
    return image.getvalue()


def digest(pixels):
    return hashlib.sha256(pixels).hexdigest()


def tokenize(text, add_special_tokens):
    return {"input_ids": [ord(character) for character in text]}


def engine(log, step_seconds=0.0, fail=False):
    command = [sys.executable, ENGINE, log, str(step_seconds)] + ["fail"] * fail
    started = videos.VideoEngine(command, 30)
    started.start()
    return started


def taken(log):
    """The engine's log: a record per request taken and per cancel."""
    return [json.loads(line) for line in log.read_text().splitlines()]


class VideoServerTest(unittest.TestCase):
    def serve(self, step_seconds=0.0):
        self.log = Path(self.enterContext(tempfile.TemporaryDirectory())) / "log"
        app = videos.VideoApp(MODEL, ("h3",), engine(self.log, step_seconds), tokenize)
        self.addCleanup(app.close)
        server = api.FrontendServer(("127.0.0.1", 0), app)
        server.RequestHandlerClass = videos.handler(api.FrontendHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_port}/v1"
        self.client = openai.OpenAI(
            base_url=self.url,
            api_key="unused",
            max_retries=0,
        )
        return app

    def test_a_job_runs_and_its_clip_and_preview_download_until_deleted(self):
        self.serve()
        videos_api = self.client.videos
        created = videos_api.create(
            prompt="a cat", model=MODEL, extra_body={"seed": 7, "preview": "true"}
        )
        self.assertEqual((created.status, created.model), ("queued", MODEL))
        video = videos_api.poll(created.id, poll_interval_ms=20)
        self.assertEqual((video.status, video.progress), ("completed", 100))
        self.assertEqual(set(video.phases), {"denoise", "total"})
        self.assertEqual(videos_api.download_content(video.id).read(), SEVEN)
        preview = videos_api.download_content(video.id, variant="preview").read()
        self.assertEqual(preview, b"preview" + SEVEN)
        self.assertEqual(taken(self.log), [{"request": 1, "tokens": [97, 32, 99, 97, 116],
                                            "frames": 73, "outputs": 3, "seed": 7}])  # fmt: skip
        self.assertTrue(videos_api.delete(video.id).deleted)
        with self.assertRaises(openai.NotFoundError):
            videos_api.retrieve(video.id)

    def test_a_job_starts_from_its_input_reference_and_ends_on_its_last_frame(self):
        self.serve()
        first = ("first.png", png(1344, 768, CANVAS), "image/png")
        video = self.client.videos.create(prompt="a cat", input_reference=first)
        self.assertEqual(
            self.client.videos.poll(video.id, poll_interval_ms=20).status, "completed"
        )
        # A last frame of another shape is scaled to cover the canvas and cropped.
        last = ("last.png", png(300, 100, bytes([10, 200, 30]) * 30000), "image/png")
        both = httpx.post(f"{self.url}/videos", data={"prompt": "a cat"},
                          files={"input_reference": first, "last_frame": last})  # fmt: skip
        self.assertEqual(both.status_code, 200)
        self.client.videos.poll(both.json()["id"], poll_interval_ms=20)
        (one, two) = taken(self.log)
        self.assertEqual(
            (one["first_frame"], "last_frame" in one), (digest(CANVAS), False)
        )
        self.assertEqual((two["first_frame"], two["last_frame"]),
                         (digest(CANVAS), digest(bytes([10, 200, 30]) * 1344 * 768)))  # fmt: skip

    def test_keyframes_are_refused_unless_image_files_of_a_first_frame(self):
        self.serve()
        prompt = {"prompt": "a cat"}
        image = ("frame.png", png(4, 2, bytes(24)), "image/png")
        for body, refused in (
            ({"data": prompt, "files": {"last_frame": image}}, "last_frame needs input_reference"),
            ({"data": prompt, "files": {"input_reference": ("frame.png", b"no image", "image/png")}},
             "input_reference: image could not be decoded"),
            ({"json": prompt | {"input_reference": "frame.png"}}, "input_reference must be an image file"),
        ):  # fmt: skip
            with self.subTest(refused=refused):
                answer = httpx.post(f"{self.url}/videos", **body)
                self.assertEqual(answer.status_code, 400)
                self.assertIn(refused, answer.json()["error"]["message"])
        self.assertEqual(taken(self.log), [])

    def test_an_invalid_request_is_refused_in_openais_shape(self):
        self.serve()
        with self.assertRaises(openai.BadRequestError) as refused:
            self.client.videos.create(prompt="a cat", seconds="4")
        self.assertEqual(refused.exception.code, "invalid_request_error")
        self.assertEqual(taken(self.log), [])

    def test_the_queue_and_the_store_are_bounded(self):
        app = self.serve(step_seconds=0.2)
        app.videos.budget_bytes = 12  # one 8-byte clip
        videos_api = self.client.videos
        jobs = [videos_api.create(prompt="job 0")]
        while videos_api.retrieve(jobs[0].id).status != "in_progress":
            time.sleep(0.02)
        jobs += [videos_api.create(prompt=f"job {n}", extra_body={"seed": n})
                 for n in range(1, videos.MAX_QUEUED_JOBS + 1)]  # fmt: skip
        with self.assertRaises(openai.APIStatusError) as refused:
            videos_api.create(prompt="one too many")
        self.assertEqual((refused.exception.status_code, refused.exception.code),
                         (503, "video_queue_full"))  # fmt: skip
        videos_api.poll(jobs[-1].id, poll_interval_ms=20)
        listed = videos_api.list(order="asc", limit=100)
        self.assertEqual([video.id for video in listed.data], [jobs[-1].id])
        self.assertEqual(app.videos.status()["evictions"], videos.MAX_QUEUED_JOBS)


class GenerateTest(unittest.TestCase):
    """`python -m server.videos`, which `splash generate` runs: one clip of a
    held assembly's engine, written to --out."""

    def setUp(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.addCleanup(signal.signal, number, signal.getsignal(number))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.log = self.directory / "log"
        self.log.touch()
        # The assembly holds the turbo variant alone.
        (self.directory / "weights" / "dit-turbo").mkdir(parents=True)

    def generate(
        self, video=True, step_seconds=0.0, fail=False, keyframes=(), request=None
    ):
        def load(*_):
            return engine(self.log, step_seconds, fail), tokenize

        with (
            mock.patch.object(videos.assembly, "hold", lambda path: (path, None)),
            mock.patch.object(videos, "serves", return_value=video),
            mock.patch.object(videos, "load", load),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            status = videos.main([
                str(self.directory), "--model", MODEL, "--binary", "splash",
                "--out", str(self.directory / "clip.mp4"),
                "--request", request or '{"prompt": "a cat", "seed": 7, "preview": true}',
                *keyframes,
            ])  # fmt: skip
        return status, stderr.getvalue()

    def test_the_clip_and_its_preview_are_written(self):
        status, stderr = self.generate()
        self.assertEqual(status, 0)
        self.assertEqual((self.directory / "clip.mp4").read_bytes(), SEVEN)
        preview = (self.directory / "clip.preview.mp4").read_bytes()
        self.assertEqual(preview, b"preview" + SEVEN)
        self.assertIn("Video · denoise · step 4/4", stderr)

    def test_the_first_frame_is_read_from_its_file(self):
        first = self.directory / "first.png"
        first.write_bytes(png(1344, 768, CANVAS))
        status, _ = self.generate(keyframes=("--first-frame", str(first)))
        self.assertEqual(status, 0)
        self.assertEqual(taken(self.log)[0]["first_frame"], digest(CANVAS))

    def test_a_variant_not_installed_and_other_steps_are_refused(self):
        status, stderr = self.generate(
            request='{"prompt": "a cat", "variant": "default"}'
        )
        self.assertEqual(status, 1)
        self.assertIn(
            "the default variant of MiniMaxAI/MiniMax-H3 is not installed", stderr
        )
        status, stderr = self.generate(request='{"prompt": "a cat", "steps": 4}')
        self.assertEqual(status, 1)
        self.assertIn("steps must be 0 or 8, the turbo variant's", stderr)
        self.assertEqual(taken(self.log), [])

    def test_a_language_model_is_refused(self):
        status, stderr = self.generate(video=False)
        self.assertEqual(status, 1)
        self.assertIn("is a language model", stderr)
        self.assertEqual(taken(self.log), [])

    def test_an_engine_error_fails_the_clip(self):
        status, stderr = self.generate(fail=True)
        self.assertEqual(status, 1)
        self.assertIn("Error · the video VAE failed", stderr)
        self.assertFalse((self.directory / "clip.mp4").exists())

    def test_a_stop_cancels_the_clip_at_its_next_step(self):
        def interrupt():
            deadline = time.monotonic() + 10
            while not self.log.read_text() and time.monotonic() < deadline:
                time.sleep(0.01)
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=interrupt, daemon=True).start()
        status, stderr = self.generate(step_seconds=0.5)
        self.assertEqual(status, 130)
        self.assertIn("cancelling at the next step", stderr)
        self.assertEqual([next(iter(record)) for record in taken(self.log)],
                         ["request", "cancel"])  # fmt: skip
        self.assertFalse((self.directory / "clip.mp4").exists())


if __name__ == "__main__":
    unittest.main()
