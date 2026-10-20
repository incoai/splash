"""A video model's server (server/videos.py) over a stand-in engine
(dev/tests/fake_video_engine.py): OpenAI's video API, through its own SDK
too, the jobs' queue, cancellation and store, and a language model's routes
absent."""

import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

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


class VideoServerTest(unittest.TestCase):
    def serve(self, step_seconds=0.02, budget_bytes=videos.STORE_BUDGET_BYTES):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.log = Path(temporary.name) / "engine.log"
        engine = videos.VideoEngine(
            [
                sys.executable,
                str(ROOT / "dev/tests/fake_video_engine.py"),
                str(self.log),
                str(step_seconds),
            ],  # fmt: skip
            30.0,
        )
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


if __name__ == "__main__":
    unittest.main()
