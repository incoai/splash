"""A video model's server (server/videos.py) over a stand-in engine
(dev/tests/fake_video_engine.py): a job through OpenAI's SDK, one from its
first and last frames, the bounds of the jobs' queue and store, a job deleted
while queued or running, and the one clip `splash generate` runs."""

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

from dev.tests import native_peer
from dev.tests.server_fixtures import console_written, main_args
from server import protocol as wire
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


def engine(log, step_seconds=0.0, fail=False, hold=False):
    command = [sys.executable, ENGINE, log, str(step_seconds)]
    command += ["fail"] * fail + ["hold"] * hold
    started = videos.VideoEngine(command, 30)
    started.start()
    return started


def taken(log):
    """The engine's log: a record per request taken and per cancel."""
    return [json.loads(line) for line in log.read_text().splitlines()]


class VideoServerTest(unittest.TestCase):
    def serve(self, step_seconds=0.0, hold=False):
        self.log = Path(self.enterContext(tempfile.TemporaryDirectory())) / "log"
        app = videos.VideoApp(
            MODEL, ("h3",), engine(self.log, step_seconds, hold=hold), tokenize
        )
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
        self.assertEqual((video.size, video.seconds), ("1344x768", "5"))
        self.assertEqual(set(video.phases), {"denoise", "total"})
        self.assertEqual(videos_api.download_content(video.id).read(), SEVEN)
        preview = videos_api.download_content(video.id, variant="preview").read()
        self.assertEqual(preview, b"preview" + SEVEN)
        self.assertEqual(taken(self.log), [{"request": 1, "tokens": [97, 32, 99, 97, 116],
                                            "frames": 124, "size": "1344x768", "outputs": 3,
                                            "seed": 7}])  # fmt: skip
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
        # Both keyframes' vision blocks at 1344x768 leave the byte-level
        # tokenizer 2 tokens of the 2,048 (docs/video.md, Video engine): 832x480.
        last = ("last.png", png(300, 100, bytes([10, 200, 30]) * 30000), "image/png")
        files = {"input_reference": first, "last_frame": last}
        refused = httpx.post(
            f"{self.url}/videos", data={"prompt": "a cat"}, files=files
        )
        self.assertEqual(refused.status_code, 400)
        self.assertIn(f"takes 2 to {videos.H3.max_text_tokens}", refused.text)
        both = httpx.post(f"{self.url}/videos", data={"prompt": "a cat", "size": "832x480"},
                          files=files)  # fmt: skip
        self.assertEqual(both.status_code, 200)
        self.client.videos.poll(both.json()["id"], poll_interval_ms=20)
        (one, two) = taken(self.log)
        self.assertEqual(
            (one["first_frame"], "last_frame" in one), (digest(CANVAS), False)
        )
        stretched = videos.images.keyframe(first[1], 832, 480, True)
        self.assertEqual((two["first_frame"], two["last_frame"]),
                         (digest(stretched), digest(bytes([10, 200, 30]) * 832 * 480)))  # fmt: skip

    def test_a_480p_job_renders_its_canvas_and_length(self):
        self.serve()
        first = ("first.png", png(1344, 768, CANVAS), "image/png")
        answer = httpx.post(f"{self.url}/videos", data={"prompt": "a cat", "size": "832x480",
                            "seconds": "5"}, files={"input_reference": first})  # fmt: skip
        self.assertEqual(answer.status_code, 200)
        created = answer.json()
        self.assertEqual((created["size"], created["seconds"]), ("832x480", "5"))
        video = self.client.videos.poll(created["id"], poll_interval_ms=20)
        self.assertEqual(video.status, "completed")
        tall = self.client.videos.create(prompt="a cat", size="480x832", seconds="10")
        self.client.videos.poll(tall.id, poll_interval_ms=20)
        # the first frame stretched onto 832 x 480, as the port puts it there
        stretched = videos.images.keyframe(first[1], 832, 480, True)
        (wide, portrait) = taken(self.log)
        self.assertEqual((wide["frames"], wide["size"], wide["first_frame"]),
                         (124, "832x480", digest(stretched)))  # fmt: skip
        self.assertEqual((portrait["frames"], portrait["size"]), (243, "480x832"))

    def test_a_canvas_outside_both_tiers_is_refused(self):
        self.serve()
        refused = ("1280x720", "1600x768", "832x464", "1024x576", "0832x480", "832", 832)  # fmt: skip
        for size in refused:
            with self.subTest(size=size):
                body = {"prompt": "a cat", "size": size}
                answer = httpx.post(f"{self.url}/videos", json=body)
                self.assertEqual(answer.status_code, 400)
                self.assertIn("size must be WIDTHxHEIGHT of a 768p or 480p canvas",
                              answer.json()["error"]["message"])  # fmt: skip
        self.assertEqual(taken(self.log), [])

    def test_seconds_are_a_whole_number_from_4_to_15(self):
        # Pika's align_num_frames of seconds x 24: the smallest 17 n + 5.
        self.assertEqual(
            [videos.clip_frames(seconds) for seconds in range(4, 16)],
            [107, 124, 158, 175, 192, 226, 243, 277, 294, 328, 345, 362],
        )
        self.assertEqual(
            [videos.clip_seconds(s) for s in ("4", 4, "15")], ["4", "4", "15"]
        )
        self.serve()
        for seconds in ("3", "16", 3, 16, "0", "-5", "07", "7.0", 7.0, True, ""):
            with self.subTest(seconds=seconds):
                body = {"prompt": "a cat", "seconds": seconds}
                answer = httpx.post(f"{self.url}/videos", json=body)
                self.assertEqual(answer.status_code, 400)
                self.assertIn("seconds must be a whole number from 4 to 15",
                              answer.json()["error"]["message"])  # fmt: skip
        self.assertEqual(taken(self.log), [])

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
            self.client.videos.create(prompt="a cat", seconds="3")
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

    def serve_held(self):
        """A server whose engine holds its first job at its first step until
        the job is cancelled, and the event set once the server has that
        step."""
        app = self.serve(hold=True)
        reported, report = threading.Event(), app.videos._report

        def step(job, event):
            report(job, event)
            reported.set()

        self.enterContext(mock.patch.object(app.videos, "_report", step))
        return app, reported

    def test_a_running_job_deleted_is_cancelled_and_the_next_one_runs(self):
        app, reported = self.serve_held()
        videos_api = self.client.videos
        running = videos_api.create(
            prompt="job 1", extra_body={"seed": 1, "preview": "true"}
        )
        self.assertTrue(reported.wait(10))
        self.assertEqual(videos_api.retrieve(running.id).status, "in_progress")
        process = app.videos.engine.process
        queued = videos_api.create(prompt="job 2", extra_body={"seed": 2})
        self.assertTrue(videos_api.delete(running.id).deleted)
        with self.assertRaises(openai.NotFoundError):
            videos_api.retrieve(running.id)
        with self.assertRaises(openai.NotFoundError):
            videos_api.download_content(running.id, variant="preview")
        video = videos_api.poll(queued.id, poll_interval_ms=20)
        self.assertEqual(video.status, "completed")
        self.assertEqual(
            videos_api.download_content(video.id).read(), (2).to_bytes(8, "little")
        )
        # The engine took the cancel, then the next job in the same process,
        # and the store keeps the next job's clip alone.
        records = [next(iter(record.items())) for record in taken(self.log)]
        self.assertEqual(records, [("request", 1), ("cancel", 1), ("request", 2)])
        self.assertIs(app.videos.engine.process, process)
        self.assertTrue(app.videos.engine.alive())
        listed = videos_api.list(order="asc")
        self.assertEqual([video.id for video in listed.data], [queued.id])
        status = app.videos.status()
        self.assertEqual((status["stored"], status["stored_bytes"]), (1, 8))

    def test_a_queued_job_deleted_never_reaches_the_engine(self):
        app, reported = self.serve_held()
        videos_api = self.client.videos
        running = videos_api.create(prompt="job 1")
        self.assertTrue(reported.wait(10))
        queued = videos_api.create(prompt="job 2")
        last = videos_api.create(prompt="job 3")
        self.assertEqual(videos_api.retrieve(queued.id).status, "queued")
        self.assertTrue(videos_api.delete(queued.id).deleted)
        with self.assertRaises(openai.NotFoundError):
            videos_api.retrieve(queued.id)
        self.assertEqual(app.videos.status()["queued"], 1)
        videos_api.delete(running.id)
        self.assertEqual(
            videos_api.poll(last.id, poll_interval_ms=20).status, "completed"
        )
        self.assertEqual([record.get("tokens") for record in taken(self.log)],
                         [[*b"job 1"], None, [*b"job 3"]])  # fmt: skip

    def test_a_job_deleted_as_its_clip_arrives_cancels_no_other(self):
        app = self.serve()
        finished, deleted = threading.Event(), threading.Event()
        run = app.videos.engine.run

        def held_run(request, on_event):
            """The engine's run, the first held once its clip is back until
            the test has deleted its job."""
            try:
                return run(request, on_event)
            finally:
                if not finished.is_set():
                    finished.set()
                    deleted.wait(10)

        self.enterContext(mock.patch.object(app.videos.engine, "run", held_run))
        videos_api = self.client.videos
        first = videos_api.create(prompt="job 1")
        self.assertTrue(finished.wait(10))
        second = videos_api.create(prompt="job 2")
        self.assertTrue(videos_api.delete(first.id).deleted)
        deleted.set()
        video = videos_api.poll(second.id, poll_interval_ms=20)
        self.assertEqual((video.status, video.error), ("completed", None))
        self.assertEqual([next(iter(record)) for record in taken(self.log)],
                         ["request", "request"])  # fmt: skip


class SeedTest(unittest.TestCase):
    def test_a_seed_is_an_unsigned_32_bit_integer(self):
        from dev.tests import native_peer

        for seed in (4294967295, "4294967295"):
            with self.subTest(seed=seed):
                request = videos.parse_request(
                    {"prompt": "a cat", "seed": seed}, (MODEL,)
                )
                frame = videos.wire.VideoRequestFrame(7, *videos.clip(request, (1, 2)))
                ((message, _),) = native_peer.ClientFrameReader().feed(
                    bytes(videos.wire.serialize_message(frame))
                )
                self.assertEqual((request.seed, message.seed), (4294967295, 4294967295))
        for seed in (4294967296, "4294967296", -1, "-1"):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(
                    videos.APIError, r"seed must be an integer in \[0, 4294967295\]"
                ):
                    videos.parse_request({"prompt": "a cat", "seed": seed}, (MODEL,))


class PresentationTest(unittest.TestCase):
    def test_token_ids_are_the_golden(self):
        # Literal ids: the vision tokens of Qwen3-VL and a 832x480 keyframe's 26 x 15 merged pads.
        block = [151652, *[151655] * 390, 151653]
        self.assertEqual(videos.presentation(tokenize, "a cat"), [*b"a cat"])
        self.assertEqual(
            videos.presentation(tokenize, "a cat", 2, (832, 480)),
            [*b"<Picture 1>: ", *block, *b"<Picture 2>: ", *block, *b"a cat"],
        )


class GenerateTest(unittest.TestCase):
    """`python -m server.videos`, which `splash generate` runs: one clip of a
    held assembly's engine, written to --out."""

    def setUp(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.addCleanup(signal.signal, number, signal.getsignal(number))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.log = self.directory / "log"
        self.log.touch()
        # The assembly holds the vdn-8 variant alone.
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
            console_written()
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

    def test_the_clip_takes_its_size_and_seconds(self):
        status, _ = self.generate(
            request='{"prompt": "a cat", "size": "864x480", "seconds": "15"}'
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            (taken(self.log)[0]["size"], taken(self.log)[0]["frames"]), ("864x480", 362)
        )

    def test_a_clip_of_7_seconds_has_175_frames(self):
        status, _ = self.generate(
            request='{"prompt": "a cat", "size": "832x480", "seconds": 7}'
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            (taken(self.log)[0]["size"], taken(self.log)[0]["frames"]), ("832x480", 175)
        )

    def test_a_variant_not_installed_and_other_steps_are_refused(self):
        status, stderr = self.generate(
            request='{"prompt": "a cat", "variant": "vdn-50"}'
        )
        self.assertEqual(status, 1)
        self.assertIn(
            "the vdn-50 variant of MiniMaxAI/MiniMax-H3 is not installed", stderr
        )
        status, stderr = self.generate(request='{"prompt": "a cat", "steps": 4}')
        self.assertEqual(status, 1)
        self.assertIn("steps must be 0 or 8, the vdn-8 variant's", stderr)
        self.assertEqual(taken(self.log), [])
        # FastH3's variants run their trained rungs.
        fast = videos.parse_request(
            {"prompt": "a cat", "variant": "fasth3-4"}, (MODEL,)
        )
        self.assertEqual(videos.clip(fast, (1, 2))[0], "fasth3-4")
        with self.assertRaisesRegex(
            videos.APIError, "steps must be 0 or 4, the fasth3-4"
        ):
            videos.parse_request(
                {"prompt": "a cat", "variant": "fasth3-4", "steps": 8}, (MODEL,)
            )

    def test_a_language_model_is_refused(self):
        status, stderr = self.generate(video=False)
        self.assertEqual(status, 1)
        self.assertIn("is a language model", stderr)
        self.assertEqual(taken(self.log), [])

    def test_an_engine_that_cannot_start_reports_why(self):
        log = self.directory / "refused"
        command = [sys.executable, ENGINE, log, "0", "refuse"]
        with self.assertRaisesRegex(videos.VideoStartFailed, "no Apple10 GPU"):
            videos.VideoEngine(command, 30).start()

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


# An engine that sends the events of its hex argument, then waits for its
# input to end.
SENDS = (
    "import os, sys; os.write(1, bytes.fromhex(sys.argv[1])); sys.stdin.buffer.read()"
)
DEVICE_REFUSAL = wire.StartFailedEvent(
    wire.StartFailureKind.DEVICE,
    wire.ShortfallBound.DEVICE_MEMORY,
    0,
    0,
    0,
    b"the video engine runs on an Apple10 GPU (M5) or later",
)


class StartTest(unittest.TestCase):
    """A video engine's start as the engine announces it, and a video
    server's start as its serve record tells the app."""

    def started(self, *events):
        sent = b"".join(native_peer.serialize_event(event) for event in events)
        engine = videos.VideoEngine([sys.executable, "-c", SENDS, sent.hex()], 30)
        self.addCleanup(engine.close)
        engine.start()
        return engine

    def test_an_engine_is_ready_after_its_warm_up(self):
        ready = wire.ReadyEvent(1, 2048, False, 0, 0, (), video=True)
        engine = self.started(wire.WarmingEvent(0, 2), wire.WarmingEvent(1, 2), ready)
        self.assertEqual(engine.ready, ready)

    def test_an_engine_that_refuses_its_start_says_why(self):
        with self.assertRaises(videos.VideoStartFailed) as raised:
            self.started(DEVICE_REFUSAL)
        self.assertIs(raised.exception.event.kind, wire.StartFailureKind.DEVICE)
        self.assertIn("Apple10", str(raised.exception))

    def test_an_engine_that_does_not_start_in_time_is_stopped(self):
        engine = videos.VideoEngine([sys.executable, "-c", SENDS, ""], 0.5)
        self.addCleanup(engine.close)
        with self.assertRaisesRegex(videos.VideoFailed, "did not start in time"):
            engine.start()
        self.assertIsNotNone(engine.process.poll())

    def test_a_language_models_engine_is_refused(self):
        with self.assertRaisesRegex(videos.VideoFailed, "serves no video model"):
            self.started(wire.ReadyEvent(1, 2048, False, 1024, 1, (2,)))

    def main(self, start):
        """main() of a video model's assembly, its app started by `start`
        (VideoApp.start's side effect): the serve record each serve_forever
        found, and the record main() left."""
        record = Path(self.enterContext(tempfile.TemporaryDirectory())) / "s.json"
        seen = []
        with (
            mock.patch.object(
                api, "parse_args", return_value=main_args(serve_record=record)
            ),
            mock.patch.object(api.videos, "serves", return_value=True),
            mock.patch.object(api.assembly, "hold", lambda path: (path, None)),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(api.videos.VideoApp, "start", side_effect=start),
            mock.patch.object(
                api, "FrontendServer", return_value=mock.Mock(server_port=8000)
            ),
            mock.patch.object(api.signal, "signal"),
            mock.patch.object(api.os, "getpid", return_value=4242),
            mock.patch.object(api, "print_status"),
        ):
            server = api.FrontendServer.return_value
            server.serve_forever.side_effect = lambda: seen.append(
                (server.server_activate.called, json.loads(record.read_text()))
            )
            with contextlib.suppress(SystemExit):
                api.main()
        return seen, json.loads(record.read_text())

    def test_a_video_server_says_ready_in_its_record_once_it_answers(self):
        app = mock.Mock()
        seen, _ = self.main(lambda *_: app)
        self.assertEqual(seen, [(True, {"pid": 4242, "phase": "ready"})])
        app.close.assert_called_once()

    def test_a_video_start_the_engine_refuses_records_the_engines_kind(self):
        def start(*_):
            raise videos.VideoStartFailed(DEVICE_REFUSAL)

        seen, record = self.main(start)
        self.assertEqual(seen, [])
        self.assertEqual(
            record["failure"],
            {
                "kind": "device",
                "message": "the video engine runs on an Apple10 GPU (M5) or later",
            },
        )

    def test_a_video_engine_that_fails_otherwise_is_an_internal_failure(self):
        def start(*_):
            raise videos.VideoFailed("bootstrap_failed", "no Apple10 GPU")

        _, record = self.main(start)
        self.assertEqual(
            record["failure"], {"kind": "internal", "message": "no Apple10 GPU"}
        )


if __name__ == "__main__":
    unittest.main()
