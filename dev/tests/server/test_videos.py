"""A video model's server (server/videos.py) over a stand-in engine
(dev/tests/fake_video_engine.py): a job through OpenAI's SDK, and the one clip
`splash generate` runs."""

import contextlib
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

import openai

from server import server as api
from server import videos

ENGINE = Path(__file__).resolve().parents[1] / "fake_video_engine.py"
MODEL = "MiniMaxAI/MiniMax-H3"
SEVEN = (7).to_bytes(8, "little")


def tokenize(text, add_special_tokens):
    return {"input_ids": [ord(character) for character in text]}


def engine(log, step_seconds=0.0):
    started = videos.VideoEngine([sys.executable, ENGINE, log, str(step_seconds)], 30)
    started.start()
    return started


def taken(log):
    """The engine's log: a record per request taken and per cancel."""
    return [json.loads(line) for line in log.read_text().splitlines()]


class VideoServerTest(unittest.TestCase):
    def setUp(self):
        self.log = Path(self.enterContext(tempfile.TemporaryDirectory())) / "log"
        app = videos.VideoApp(MODEL, ("h3",), engine(self.log), tokenize)
        self.addCleanup(app.close)
        server = api.FrontendServer(("127.0.0.1", 0), app)
        server.RequestHandlerClass = videos.handler(api.FrontendHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.client = openai.OpenAI(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="unused"
        )

    def test_a_job_runs_and_its_clip_and_preview_download_until_deleted(self):
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

    def test_an_invalid_request_is_refused_in_openais_shape(self):
        with self.assertRaises(openai.BadRequestError) as refused:
            self.client.videos.create(prompt="a cat", seconds="4")
        self.assertEqual(refused.exception.code, "invalid_request_error")
        self.assertEqual(taken(self.log), [])


class GenerateTest(unittest.TestCase):
    """`python -m server.videos`, which `splash generate` runs: one clip of a
    held assembly's engine, written to --out."""

    def setUp(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.addCleanup(signal.signal, number, signal.getsignal(number))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.log = self.directory / "log"
        self.log.touch()

    def generate(self, video=True, step_seconds=0.0):
        def load(*_):
            return engine(self.log, step_seconds), tokenize

        with (
            mock.patch.object(videos.assembly, "hold", lambda path: (path, None)),
            mock.patch.object(videos, "serves", return_value=video),
            mock.patch.object(videos, "load", load),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            status = videos.main([
                str(self.directory), "--model", MODEL, "--binary", "splash",
                "--out", str(self.directory / "clip.mp4"),
                "--request", '{"prompt": "a cat", "seed": 7, "preview": true}',
            ])  # fmt: skip
        return status, stderr.getvalue()

    def test_the_clip_and_its_preview_are_written(self):
        status, stderr = self.generate()
        self.assertEqual(status, 0)
        self.assertEqual((self.directory / "clip.mp4").read_bytes(), SEVEN)
        preview = (self.directory / "clip.preview.mp4").read_bytes()
        self.assertEqual(preview, b"preview" + SEVEN)
        self.assertIn("Video · denoise · step 4/4", stderr)

    def test_a_language_model_is_refused(self):
        status, stderr = self.generate(video=False)
        self.assertEqual(status, 1)
        self.assertIn("is a language model", stderr)
        self.assertEqual(taken(self.log), [])

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
