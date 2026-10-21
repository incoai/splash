"""A video model's server (server/videos.py) over a stand-in engine
(dev/tests/fake_video_engine.py): OpenAI's video API, through its own SDK
too, the jobs' queue, cancellation and store, and a language model's routes
absent; and the one clip `splash generate` runs."""

import contextlib
import http.client
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

from openai import OpenAI

from server import server as api
from server import videos

ROOT = Path(__file__).resolve().parents[3]
MODEL = "MiniMaxAI/MiniMax-H3"


class FakeTokenizer:
    """One token per character."""

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def fake_engine(log, step_seconds, *options):
    return videos.VideoEngine(
        [
            sys.executable,
            str(ROOT / "dev/tests/fake_video_engine.py"),
            str(log),
            str(step_seconds),
            *options,
        ],  # fmt: skip
        30.0,
    )


class VideoServerTest(unittest.TestCase):
    def serve(self, step_seconds=0.02, budget_bytes=videos.STORE_BUDGET_BYTES):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.log = Path(temporary.name) / "engine.log"
        engine = fake_engine(self.log, step_seconds)
        engine.start()
        app = videos.VideoApp(MODEL, ("h3",), engine, FakeTokenizer())
        app.videos.budget_bytes = budget_bytes
        self.addCleanup(app.close)
        server = api.FrontendServer(("127.0.0.1", 0), app)
        server.RequestHandlerClass = videos.handler(api.FrontendHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.port = server.server_port
        return app

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            connection.request(
                method, path, None if body is None else json.dumps(body), headers
            )
            response = connection.getresponse()
            data = response.read()
            kind = response.getheader("Content-Type", "")
        finally:
            connection.close()
        return response.status, json.loads(data) if kind.startswith(
            "application/json"
        ) else data

    def wait(self, job_id, status="completed"):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            code, video = self.request("GET", f"/v1/videos/{job_id}")
            if code != 200 or video["status"] == status:
                return video
            time.sleep(0.02)
        self.fail(f"{job_id} did not reach {status}")

    def engine_log(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_a_job_runs_through_openais_sdk(self):
        self.serve()
        client = OpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="unused")
        created = client.videos.create(
            prompt="a cat", model=MODEL, seconds="3", size="1344x768",
            extra_body={"seed": 7, "preview": "true"},
        )  # fmt: skip
        self.assertEqual(
            (created.object, created.status, created.model), ("video", "queued", MODEL)
        )
        video = self.wait(created.id)
        self.assertEqual(
            (video["progress"], video["error"], video["preview"]), (100, None, True)
        )
        self.assertEqual(
            set(video["phases"]),
            {"encode", "prepare", "denoise", "preview", "decode", "mux", "total"},
        )
        retrieved = client.videos.retrieve(created.id)
        self.assertEqual(retrieved.status, "completed")
        mp4 = client.videos.download_content(created.id).read()
        self.assertEqual(mp4[4:8], b"ftyp")
        self.assertEqual(mp4[-8:], (7).to_bytes(8, "little"))
        self.assertEqual(
            self.request("GET", f"/v1/videos/{created.id}/content?variant=preview")[1],
            b"preview" + mp4,
        )
        (taken,) = self.engine_log()
        self.assertEqual(
            taken,
            {"request": 1, "tokens": [ord(c) for c in "a cat"], "frames": 73, "variant": 0,
             "outputs": 3, "seed": 7},
        )  # fmt: skip
        listed = client.videos.list()
        self.assertEqual([video.id for video in listed.data], [created.id])
        self.assertTrue(client.videos.delete(created.id).deleted)
        self.assertEqual(self.request("GET", f"/v1/videos/{created.id}")[0], 404)

    def test_requests_are_checked_and_refused_in_openais_shape(self):
        self.serve()
        for body, status, code in (
            ({"prompt": "a cat", "seconds": "4"}, 400, "invalid_request_error"),
            ({"prompt": "a cat", "size": "720x1280"}, 400, "invalid_request_error"),
            ({"prompt": "a cat", "input_reference": "x"}, 400, "unsupported_parameter"),
            ({"prompt": "a cat", "quality": "high"}, 400, "unsupported_parameter"),
            ({"prompt": "a cat", "model": "other"}, 404, "model_not_found"),
            ({"prompt": "a"}, 400, "invalid_request_error"),
            ({"prompt": ""}, 400, "invalid_request_error"),
        ):
            with self.subTest(body=body):
                answer = self.request("POST", "/v1/videos", body)
                self.assertEqual(
                    (answer[0], answer[1]["error"]["code"]), (status, code)
                )
        self.assertEqual(self.engine_log(), [])

    def test_a_language_models_routes_are_absent(self):
        self.serve()
        for method, path in (
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/messages"),
            ("GET", "/v1/responses/resp_x"),
        ):
            with self.subTest(path=path):
                status, answer = self.request(
                    method, path, {} if method == "POST" else None
                )
                self.assertEqual(status, 404)
                self.assertIn("serves the video model", answer["error"]["message"])
        status, models = self.request("GET", "/v1/models")
        self.assertEqual([model["id"] for model in models["data"]], [MODEL, "h3"])
        self.assertEqual(self.request("GET", "/ready"), (200, {"status": "ready"}))

    def test_a_deleted_job_is_cancelled_at_a_step_and_the_next_one_runs(self):
        self.serve(step_seconds=0.5)
        first = self.request("POST", "/v1/videos", {"prompt": "slow job"})[1]
        second = self.request("POST", "/v1/videos", {"prompt": "next job"})[1]
        deadline = time.monotonic() + 10
        while self.request("GET", f"/v1/videos/{first['id']}")[1]["phase"] != "denoise":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.02)
        self.assertEqual(self.request("DELETE", f"/v1/videos/{first['id']}")[0], 200)
        self.assertEqual(self.wait(second["id"])["status"], "completed")
        self.assertEqual(
            [next(iter(record.items())) for record in self.engine_log()],
            [("request", 1), ("cancel", 1), ("request", 2)],
        )

    def test_the_queue_and_the_store_are_bounded(self):
        app = self.serve(step_seconds=0.2, budget_bytes=30)
        jobs = [self.request("POST", "/v1/videos", {"prompt": "job 0"})[1]]
        self.wait(jobs[0]["id"], "in_progress")
        jobs += [self.request("POST", "/v1/videos", {"prompt": f"job {n}", "seed": n})[1]
                 for n in range(1, videos.MAX_QUEUED_JOBS + 1)]  # fmt: skip
        status, refused = self.request("POST", "/v1/videos", {"prompt": "one too many"})
        self.assertEqual((status, refused["error"]["code"]), (503, "video_queue_full"))
        self.wait(jobs[-1]["id"])
        # Each mp4 is 20 bytes: the store keeps the newest finished one.
        listed = self.request("GET", "/v1/videos?order=asc&limit=100")[1]
        self.assertEqual([video["id"] for video in listed["data"]], [jobs[-1]["id"]])
        self.assertEqual(app.videos.status()["evictions"], videos.MAX_QUEUED_JOBS)


class GenerateTest(unittest.TestCase):
    """`python -m server.videos`, which `splash generate` executes: one clip
    of a held assembly's engine, written to --out."""

    def setUp(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.addCleanup(signal.signal, number, signal.getsignal(number))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.log = self.directory / "engine.log"
        self.log.touch()

    def generate(self, request, *, video=True, step_seconds=0.0, options=()):
        """main's status and stderr, its engine the stand-in."""

        def load(model_root, binary, startup_timeout):
            engine = fake_engine(self.log, step_seconds, *options)
            engine.start()
            return engine, FakeTokenizer()

        stderr = io.StringIO()
        with (
            mock.patch.object(videos.assembly, "hold", lambda path: (path, None)),
            mock.patch.object(videos, "serves", return_value=video),
            mock.patch.object(videos, "load", side_effect=load),
            contextlib.redirect_stderr(stderr),
        ):
            status = videos.main([
                str(self.directory), "--model", MODEL, "--binary", "splash",
                "--out", str(self.directory / "clip.mp4"),
                "--request", json.dumps(request),
            ])  # fmt: skip
        return status, stderr.getvalue()

    def test_the_clip_and_its_preview_are_written(self):
        status, stderr = self.generate(
            {"prompt": "a cat", "seed": "7", "preview": True, "audio": False}
        )
        self.assertEqual(status, 0)
        mp4 = (self.directory / "clip.mp4").read_bytes()
        self.assertEqual(mp4[-8:], (7).to_bytes(8, "little"))
        self.assertEqual(
            (self.directory / "clip.preview.mp4").read_bytes(), b"preview" + mp4
        )
        (taken,) = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(
            taken,
            {"request": 1, "tokens": [ord(c) for c in "a cat"], "frames": 73, "variant": 0,
             "outputs": 1, "seed": 7},
        )  # fmt: skip
        self.assertIn("Video · denoise · step 4/4", stderr)
        self.assertIn("· 73 frames ·", stderr)

    def test_without_preview_only_the_clip_is_written(self):
        self.assertEqual(self.generate({"prompt": "a cat", "seconds": "10"})[0], 0)
        self.assertEqual(
            sorted(path.name for path in self.directory.glob("*.mp4")), ["clip.mp4"]
        )

    def test_a_language_model_and_an_invalid_request_are_refused(self):
        status, stderr = self.generate({"prompt": "a cat"}, video=False)
        self.assertEqual(status, 1)
        self.assertIn("is a language model", stderr)
        status, stderr = self.generate({"prompt": "a cat", "seconds": "4"})
        self.assertEqual(status, 1)
        self.assertIn("seconds must be one of 3, 10, 15", stderr)
        self.assertEqual(self.log.read_text(), "")
        self.assertFalse((self.directory / "clip.mp4").exists())

    def test_an_engine_error_fails_the_clip(self):
        status, stderr = self.generate({"prompt": "a cat"}, options=["fail"])
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
        status, stderr = self.generate({"prompt": "a cat"}, step_seconds=0.5)
        self.assertEqual(status, 130)
        self.assertIn("cancelling at the next step", stderr)
        self.assertEqual(
            [
                next(iter(record))
                for record in map(json.loads, self.log.read_text().splitlines())
            ],
            ["request", "cancel"],
        )
        self.assertFalse((self.directory / "clip.mp4").exists())


if __name__ == "__main__":
    unittest.main()
