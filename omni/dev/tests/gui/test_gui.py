import http.client
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from dev.tests.server_fixtures import keep_stop_signals
from gui import app, progress, proxy, session
from install import families, launcher, models, upstream
from server import serve_options

TTS_4BIT = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
ASR_4BIT = "mlx-community/Qwen3-ASR-1.7B-4bit"
ASR_BF16 = "mlx-community/Qwen3-ASR-1.7B-bf16"
ASR_SMALL = "mlx-community/Qwen3-ASR-0.6B-4bit"
DRAFT = serve_options.DEFAULT_DRAFT_MODEL


def selection(**body):
    return session.Selection.from_request({"task": "tts", "model": TTS_4BIT, **body})


def wait_for(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    raise AssertionError("condition not reached")


class SelectionTests(unittest.TestCase):
    def test_offers_exactly_the_supported_checkpoints(self):
        self.assertEqual(
            [checkpoint.model for checkpoint in session.CHECKPOINTS],
            list(families.SUPPORTED_MODELS),
        )
        self.assertEqual(
            {c.model for c in session.CHECKPOINTS if c.task == "asr"},
            {ASR_4BIT, ASR_BF16, ASR_SMALL},
        )
        # Each transcription model decodes with the DFlash draft of its size.
        self.assertEqual(
            {c.model: c.draft for c in session.CHECKPOINTS if c.task == "asr"},
            {
                ASR_4BIT: DRAFT,
                ASR_BF16: DRAFT,
                ASR_SMALL: serve_options.SMALL_DRAFT_MODEL,
            },
        )

    def test_speech_models_are_offered_by_type_then_precision(self):
        speech = [c for c in session.CHECKPOINTS if c.task == "tts"]
        self.assertEqual(
            [variant.id for variant in session.VARIANTS],
            ["base", "voice_design", "custom_voice"],
        )
        names = {
            "base": "Base",
            "voice_design": "VoiceDesign",
            "custom_voice": "CustomVoice",
        }
        # Each 1.7B type at each precision, once, and the 0.6B Base and
        # CustomVoice at 4-bit (there is no 0.6B VoiceDesign); the type is
        # the repository's.
        self.assertEqual(
            sorted((c.size, c.variant, c.precision) for c in speech),
            sorted(
                [("1.7B", v, p) for v in names for p in ("4-bit", "bf16")]
                + [("0.6B", v, "4-bit") for v in ("base", "custom_voice")]
            ),
        )
        for checkpoint in speech:
            with self.subTest(model=checkpoint.model):
                self.assertIn(f"-{names[checkpoint.variant]}-", checkpoint.model)
                self.assertTrue(
                    checkpoint.model.endswith(checkpoint.precision.replace("-", ""))
                )
                self.assertEqual(
                    checkpoint.label,
                    f"Qwen3-TTS {checkpoint.size} {names[checkpoint.variant]} · "
                    f"{checkpoint.precision}",
                )
        for checkpoint in session.CHECKPOINTS:
            if checkpoint.task == "asr":
                self.assertIsNone(checkpoint.variant)
                self.assertTrue(
                    checkpoint.label.startswith(f"Qwen3-ASR {checkpoint.size} · ")
                )
                self.assertIn(f"-{checkpoint.size}-", checkpoint.model)
        # No two choices read alike.
        labels = [checkpoint.label for checkpoint in session.CHECKPOINTS]
        self.assertEqual(len(set(labels)), len(labels))

    def test_refuses_what_serve_would_refuse(self):
        for body, message in (
            ({"task": "chat", "model": TTS_4BIT}, "task"),
            ({"task": "tts", "model": "community/other"}, "one of"),
            ({"task": "tts", "model": ASR_4BIT}, "does not serve"),
            ({"task": "asr", "model": TTS_4BIT}, "does not serve"),
            ({"task": "tts", "model": TTS_4BIT, "port": 0}, "port"),
            ({"task": "tts", "model": TTS_4BIT, "port": "8000"}, "port"),
            ({"task": "tts", "model": TTS_4BIT, "lanes": 17}, "16"),
            ({"task": "tts", "model": TTS_4BIT, "lanes": 0}, "lanes"),
            ({"task": "asr", "model": ASR_4BIT, "lanes": 33}, "32"),
            ({"task": "tts", "model": TTS_4BIT, "max_memory": "lots"}, "max_memory"),
            ({"task": "tts", "model": TTS_4BIT, "disable_ane": "yes"}, "disable_ane"),
            ({"task": "asr", "model": ASR_4BIT, "speculative": 1}, "speculative"),
        ):
            with self.subTest(body=body):
                with self.assertRaisesRegex(session.SelectionError, message):
                    session.Selection.from_request(body)
        with self.assertRaises(session.SelectionError):
            session.Selection.from_request(["tts"])

    def test_defaults_are_serves(self):
        chosen = selection()
        self.assertEqual(chosen.port, serve_options.DEFAULT_PORT)
        self.assertIsNone(chosen.lanes)
        self.assertIsNone(chosen.max_memory)
        self.assertFalse(chosen.disable_ane)
        self.assertTrue(
            session.Selection.from_request(
                {"task": "asr", "model": ASR_4BIT}
            ).speculative
        )
        self.assertIsNone(selection(max_memory="auto").max_memory)
        self.assertIsNone(selection(max_memory="  ").max_memory)
        self.assertEqual(selection(max_memory=" 28G ").max_memory, "28G")
        self.assertEqual(
            session.Selection.from_request(
                {"task": "asr", "model": ASR_4BIT, "lanes": 32}
            ).lanes,
            32,
        )

    def test_downloads_only_the_selected_checkpoint_and_its_draft(self):
        self.assertEqual(selection().downloads(), (TTS_4BIT,))
        # Speech models have no draft, whatever the page sent.
        self.assertEqual(selection(speculative=True).downloads(), (TTS_4BIT,))
        for model in (ASR_4BIT, ASR_BF16):
            with self.subTest(model=model):
                on = session.Selection.from_request({"task": "asr", "model": model})
                off = session.Selection.from_request(
                    {"task": "asr", "model": model, "speculative": False}
                )
                self.assertEqual(on.downloads(), (model, DRAFT))
                self.assertEqual(off.downloads(), (model,))

    def test_a_voice_model_downloads_only_itself(self):
        voice_models = [
            c.model
            for c in session.CHECKPOINTS
            if c.task == "tts" and c.variant != "base"
        ]
        self.assertEqual(len(voice_models), 5)
        for model in voice_models:
            with self.subTest(model=model):
                chosen = session.Selection.from_request(
                    {"task": "tts", "model": model, "speculative": True}
                )
                self.assertEqual(chosen.downloads(), (model,))
                argv = chosen.serve_arguments()
                self.assertEqual(argv[:3], ["serve", "--model", model])
                self.assertNotIn("--no-draft-model", argv)
                # Missing: serve fetches it alone. Here: offline, nothing fetched.
                self.assertEqual(
                    session.plan(chosen, model_present=lambda repo: False),
                    session.Plan((model,), (), False),
                )
                asked = []
                self.assertEqual(
                    session.plan(
                        chosen,
                        model_present=lambda repo: asked.append(repo) or True,
                        draft_present=lambda repo: self.fail("no draft"),
                    ),
                    session.Plan((), (), True),
                )
                self.assertEqual(asked, [model])

    def test_command_is_the_launchers_serve_with_the_chosen_options(self):
        cases = (
            (selection(), {}),
            (
                selection(
                    port=8123, lanes=4, max_memory="28G", disable_ane=True, offline=True
                ),
                {
                    "port": 8123,
                    "lanes": 4,
                    "max_memory": 28 << 30,
                    "disable_ane": True,
                    "offline": True,
                },
            ),
            (
                session.Selection.from_request(
                    {"task": "asr", "model": ASR_BF16, "speculative": False}
                ),
                {"no_draft_model": True},
            ),
            (
                session.Selection.from_request({"task": "asr", "model": ASR_4BIT}),
                {"no_draft_model": False, "draft_model": None},
            ),
        )
        origins = app.page_origins(7860)
        for chosen, expected in cases:
            with self.subTest(selection=chosen):
                command = chosen.command(origins, python="/python")
                self.assertEqual(
                    command[:3],
                    ["/python", "-u", str(launcher.ROOT / "install/launcher.py")],
                )
                with mock.patch.dict(os.environ, {}, clear=True):
                    args = launcher.parse_args(command[3:])
                self.assertEqual(args.command, "serve")
                self.assertEqual(args.model, chosen.model)
                self.assertTrue(args.explicit_port)
                expected = {
                    "port": 8000,
                    "lanes": None,
                    "max_memory": None,
                    "disable_ane": False,
                    "offline": False,
                    "no_draft_model": False,
                    **expected,
                }
                for name, value in expected.items():
                    self.assertEqual(getattr(args, name), value, name)
                self.assertEqual(
                    args.allowed_origin,
                    [serve_options.parse_allowed_origin(origin) for origin in origins],
                )
                self.assertNotIn("--allowed-origin", chosen.shown_command())
                self.assertTrue(
                    chosen.shown_command().startswith("splash-omni serve --model ")
                )


class OutputTests(unittest.TestCase):
    def test_classifies_serves_output(self):
        cases = {
            "Fetching 1 file(s), 2.20 GB, from x/y@abc; cached files are reused.": "download",
            "Downloading mlx-community/Qwen3-ASR-0.6B-4bit, the speculative draft": "download",
            "Installing x/y as qwen3-tts (mlx-affine).": "install",
            "Splash model x/y is already installed in /m": "install",
            "12:00:01 Loading · x/y · qwen3-tts": "loading",
            "12:00:09 Ready · x/y · qwen3-tts · 8 lanes · http://127.0.0.1:8000": "ready",
            "12:00:09 Error · native runtime did not become ready": "error",
            "12:00:02 Downloading · draft d/e for speculative transcription": "download",
            "error: Splash is already serving (PID 1, model x/y, port 8000)": "error",
            "Another Splash build is running; waiting...": "build",
        }
        for line, kind in cases.items():
            with self.subTest(line=line):
                self.assertEqual(session.classify(line)[0], kind)
        self.assertEqual(
            session.classify("error: cannot bind 127.0.0.1:8000: in use")[1],
            "cannot bind 127.0.0.1:8000: in use",
        )
        self.assertIsNone(session.classify("12:00:03 Draft · d/e · speculative"))
        self.assertIsNone(session.classify("Warning: something"))

    def test_redrawn_lines_are_live_until_a_newline(self):
        log = session.OutputLog()
        self.assertEqual(log.feed("one\ntw"), ["one"])
        self.assertEqual(log.current, "tw")
        self.assertEqual(log.feed("o\r\n"), ["two"])
        self.assertEqual(log.feed("\r 10%|#"), [])
        log.feed("\r 50%|###\r")
        self.assertEqual(log.current, " 50%|###")
        self.assertEqual(log.feed("100%|#####\r"), [])
        # \r\n split across reads ends the line once.
        self.assertEqual(log.feed("\n"), ["100%|#####"])
        self.assertEqual(
            [line["text"] for line in log.since(0)], ["one", "two", "100%|#####"]
        )
        self.assertEqual([line["text"] for line in log.since(2)], ["100%|#####"])
        bounded = session.OutputLog(limit=2)
        bounded.feed("a\nb\nc\n")
        self.assertEqual([line["text"] for line in bounded.since(0)], ["b", "c"])
        self.assertEqual(bounded.sequence, 3)


# A stand-in for `splash serve`: what it prints, then it serves until
# SIGINT, or exits as the test tells it.
FAKE_SERVE = r"""
import os, signal, sys, time
signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
print("Splash model x/y is already installed in /m", flush=True)
print("Fetching 1 file(s), 2.20 GB, from x/y@abc; cached files are reused.", flush=True)
sys.stdout.write(" 40%|####\r"); sys.stdout.flush()
mode = sys.argv[1]
if mode == "fail":
    print("error: cannot bind 127.0.0.1:8000: [Errno 48] Address already in use", flush=True)
    sys.exit(1)
print("12:00:01 Loading · x/y · qwen3-tts", flush=True)
try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt:
    print("12:00:02 Stopping · releasing engine resources", flush=True)
    sys.exit(130)
"""


class FakeMonitor:
    instances = []

    def __init__(self, repos, *, offline=False, listed=None):
        self.repos, self.offline, self.listed = repos, offline, listed
        self.started = self.stopped = False
        FakeMonitor.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def snapshot(self):
        return [{"repo": repo, "state": "cached"} for repo in self.repos]


class SessionTests(unittest.TestCase):
    def setUp(self):
        FakeMonitor.instances = []
        self.commands = []
        self.ready = threading.Event()

    def session(self, mode="serve", plan=None, fetch="ok"):
        def popen(command, **options):
            self.commands.append((command, options))
            if session.FETCH in command:
                # The draft's fetch: what it prints, and how it ends.
                script = (
                    "import sys; print('Downloading ' + sys.argv[1], flush=True);"
                    + (
                        "sys.exit(0)"
                        if fetch == "ok"
                        else "print('error: cannot download ' + sys.argv[1]);"
                        "sys.exit(1)"
                    )
                )
                return subprocess.Popen(
                    [sys.executable, "-c", script, command[-1]], **options
                )
            return subprocess.Popen([sys.executable, "-c", FAKE_SERVE, mode], **options)

        result = session.ServerSession(
            allowed_origins=app.page_origins(7860),
            popen=popen,
            ready=lambda base_url: self.ready.is_set(),
            monitor=FakeMonitor,
            poll_seconds=0.01,
            planner=lambda chosen: plan or session.Plan(chosen.downloads(), (), False),
        )
        self.addCleanup(result.shutdown, 5)
        return result

    def test_start_load_ready_stop(self):
        served = self.session()
        chosen = session.Selection.from_request(
            {"task": "asr", "model": ASR_4BIT, "port": 8123}
        )
        with mock.patch.dict(os.environ, {"SPLASH_PORT": "9999"}):
            served.start(chosen)
        command, options = self.commands[0]
        self.assertEqual(command[3:5], ["serve", "--model"])
        self.assertTrue(options["start_new_session"])
        self.assertNotIn("SPLASH_PORT", options["env"])
        self.assertEqual(FakeMonitor.instances[0].repos, (ASR_4BIT, DRAFT))
        self.assertTrue(FakeMonitor.instances[0].started)
        wait_for(lambda: served.phase == session.LOADING)
        state = served.snapshot()
        self.assertEqual(state["base_url"], "http://127.0.0.1:8123")
        self.assertEqual(state["selection"]["model"], ASR_4BIT)
        self.assertIn("Loading", state["detail"])
        texts = [line["text"] for line in state["log"]]
        # As in a terminal, the line after a bare \r overwrites it.
        self.assertNotIn(" 40%|####", texts)
        self.assertEqual(texts[-1], "12:00:01 Loading · x/y · qwen3-tts")
        with self.assertRaisesRegex(RuntimeError, "already running"):
            served.start(chosen)
        self.ready.set()
        wait_for(lambda: served.phase == session.READY)
        self.assertIsNotNone(served.snapshot()["ready_at"])
        served.stop()
        self.assertEqual(served.phase, session.STOPPING)
        wait_for(lambda: served.phase == session.IDLE)
        state = served.snapshot(since=len(texts))
        self.assertEqual(state["exit_status"], 130)
        self.assertIsNone(state["error"])
        self.assertTrue(FakeMonitor.instances[0].stopped)
        self.assertIn("Stopping", state["log"][-1]["text"])
        # A new start after a stop.
        self.ready.clear()
        served.start(selection(port=8124))
        wait_for(lambda: served.phase == session.LOADING)
        self.assertEqual(FakeMonitor.instances[1].repos, (TTS_4BIT,))

    def test_a_process_that_ends_on_its_own_failed_with_its_error(self):
        served = self.session("fail")
        served.start(selection())
        wait_for(lambda: served.phase == session.FAILED)
        state = served.snapshot()
        self.assertEqual(state["exit_status"], 1)
        self.assertIn("Address already in use", state["error"])
        self.assertEqual(state["step"], "download")
        # Back to setup clears it, and a new start is allowed.
        served.stop()
        self.assertEqual(served.phase, session.IDLE)

    def test_a_present_model_starts_offline_after_fetching_only_its_draft(self):
        missing = session.Plan((DRAFT,), (DRAFT,), True)
        served = self.session(plan=missing)
        served.start(session.Selection.from_request({"task": "asr", "model": ASR_4BIT}))
        wait_for(lambda: served.phase == session.LOADING)
        (fetch, _), (serve, options) = self.commands
        self.assertEqual(fetch[-1], DRAFT)
        self.assertEqual(fetch[0], str(session.paths.PYTHON))
        self.assertEqual(options["env"]["PYTHONPATH"], str(session.paths.ROOT))
        self.assertIn("--offline", serve)
        self.assertNotIn("--no-draft-model", serve)
        monitor = FakeMonitor.instances[0]
        self.assertEqual((monitor.offline, monitor.listed), (False, (DRAFT,)))
        state = served.snapshot()
        self.assertEqual(state["fetch"], [DRAFT])
        self.assertIn("--offline", state["command"])
        self.assertIn(f"Downloading {DRAFT}", [line["text"] for line in state["log"]])

    def test_a_failed_fetch_starts_no_server(self):
        served = self.session(plan=session.Plan((DRAFT,), (DRAFT,), True), fetch="fail")
        served.start(session.Selection.from_request({"task": "asr", "model": ASR_4BIT}))
        wait_for(lambda: served.phase == session.FAILED)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(served.snapshot()["error"], f"cannot download {DRAFT}")

    def test_everything_here_lists_nothing_on_the_hub(self):
        served = self.session(plan=session.Plan((), (), True))
        served.start(selection())
        wait_for(lambda: served.phase == session.LOADING)
        self.assertEqual(len(self.commands), 1)
        self.assertIn("--offline", self.commands[0][0])
        self.assertEqual(FakeMonitor.instances[0].offline, True)

    def test_stop_while_preparing_and_idle_stop_is_harmless(self):
        served = self.session()
        served.stop()
        self.assertEqual(served.phase, session.IDLE)
        served.start(selection())
        served.stop()
        wait_for(lambda: served.phase == session.IDLE)
        self.assertIsNone(served.snapshot()["error"])


class PlanTests(unittest.TestCase):
    def plan(self, chosen, present=(), **options):
        return session.plan(
            chosen,
            model_present=lambda repo: repo in present,
            draft_present=lambda repo: repo in present,
            **options,
        )

    def test_never_asks_the_hub_about_what_is_here(self):
        asr = session.Selection.from_request({"task": "asr", "model": ASR_4BIT})
        plain = session.Selection.from_request(
            {"task": "asr", "model": ASR_4BIT, "speculative": False}
        )
        with mock.patch.object(session.paths, "PYTHON", Path(sys.executable)):
            cases = (
                (selection(), (TTS_4BIT,), session.Plan((), (), True)),
                (selection(), (), session.Plan((TTS_4BIT,), (), False)),
                (asr, (ASR_4BIT, DRAFT), session.Plan((), (), True)),
                (asr, (ASR_4BIT,), session.Plan((DRAFT,), (DRAFT,), True)),
                (asr, (DRAFT,), session.Plan((ASR_4BIT,), (), False)),
                (asr, (), session.Plan((ASR_4BIT, DRAFT), (), False)),
                (plain, (ASR_4BIT,), session.Plan((), (), True)),
                (plain, (), session.Plan((ASR_4BIT,), (), False)),
            )
            for chosen, present, expected in cases:
                with self.subTest(selection=chosen, present=present):
                    self.assertEqual(self.plan(chosen, present), expected)
        # Without .venv to fetch the draft with, serve sets it up and does.
        with mock.patch.object(session.paths, "PYTHON", Path("/nonexistent/python")):
            self.assertEqual(
                self.plan(asr, (ASR_4BIT,)), session.Plan((DRAFT,), (), False)
            )

    def test_offline_start_needs_everything_here(self):
        chosen = session.Selection.from_request(
            {"task": "asr", "model": ASR_4BIT, "offline": True}
        )
        self.assertEqual(
            self.plan(chosen, (ASR_4BIT, DRAFT)), session.Plan((), (), True)
        )
        with self.assertRaisesRegex(session.SelectionError, "Offline off"):
            self.plan(chosen, (ASR_4BIT,))

    def test_offline_command_is_serves(self):
        command = selection().command(offline=True)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(launcher.parse_args(command[3:]).offline)
        self.assertIn("--offline", selection().shown_command(offline=True))
        self.assertNotIn("--offline", selection().shown_command(offline=False))

    def test_present_is_the_installers_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "hub"
            folder = cache / "models--o--r"
            for commit in ("a" * 40, "b" * 40):
                snapshot = folder / "snapshots" / commit
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text("{}")
                (snapshot / "model.safetensors").write_text("w")
            (folder / "refs").mkdir()
            with (
                mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)}),
                mock.patch.object(session.paths, "MODELS", root / "models"),
            ):
                self.assertFalse(session.model_present("o/r"))
                self.assertTrue(session.draft_present("o/r"))
                (folder / "refs" / "main").write_text("a" * 40)
                self.assertTrue(session.model_present("o/r"))
                self.assertFalse(session.model_present("o/other"))
                installed = root / "models" / "o" / "other"
                installed.mkdir(parents=True)
                # A damaged installation is reinstalled, from the Hub here.
                (installed / "model.json").write_text("{}")
                self.assertFalse(session.model_present("o/other"))
                record = {
                    "version": 1,
                    "model": "o/other",
                    "family": "qwen3-tts",
                    "target_format": "mlx-affine",
                    "sources": {"target": {"repo": "o/other", "revision": "c" * 40}},
                    "files": {},
                }
                (installed / "model.json").unlink()
                (installed / "model.json").write_text(json.dumps(record))
                self.assertTrue(session.model_present("o/other"))
        # The installer's own functions answer, for the installation's
        # selection.
        with (
            mock.patch.object(upstream, "present", return_value=True) as present,
            mock.patch.object(upstream, "draft_present", return_value=False),
        ):
            self.assertTrue(session.model_present(TTS_4BIT))
            self.assertFalse(session.draft_present(DRAFT))
        self.assertEqual(
            present.call_args.args,
            (models.Selection.of(session.paths.MODELS, TTS_4BIT),),
        )

    def test_the_installers_cache_words_are_an_install_step(self):
        line = "Using x/y@abc from the Hugging Face cache; the Hub is not asked."
        self.assertEqual(session.classify(line), ("install", line))


class ProgressTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def folder(self, repo):
        return self.root / repo.replace("/", "--")

    def blob(self, repo, name, size):
        path = self.folder(repo) / "blobs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    def monitor(
        self, files, *, complete=lambda repo: False, offline=False, listed=None
    ):
        resolved = []

        def resolve(repo):
            resolved.append(repo)
            if isinstance(files[repo], Exception):
                raise files[repo]
            return files[repo]

        watched = progress.DownloadMonitor(
            tuple(files),
            offline=offline,
            resolve=resolve,
            folder=self.folder,
            complete=complete,
            listed=listed,
        )
        return watched, resolved

    def test_counts_complete_and_incomplete_blobs(self):
        files = {
            "a/model": {"model.safetensors": (1000, "b1"), "config.json": (10, "b2")}
        }
        watched, resolved = self.monitor(files)
        watched.list()
        self.assertEqual(resolved, ["a/model"])
        self.assertEqual(
            watched.snapshot(),
            [{"repo": "a/model", "total": 1010, "present": 0, "state": "pending"}],
        )
        self.blob("a/model", "b2", 10)
        self.blob("a/model", "b1.1234abcd.incomplete", 400)
        self.assertEqual(watched.snapshot()[0]["present"], 410)
        self.assertEqual(watched.snapshot()[0]["state"], "downloading")
        (self.folder("a/model") / "blobs" / "b1.1234abcd.incomplete").rename(
            self.folder("a/model") / "blobs" / "b1"
        )
        self.assertEqual(watched.snapshot()[0]["state"], "complete")
        self.assertEqual(watched.snapshot()[0]["present"], 1010)

    def test_a_repository_the_cache_held_is_reused(self):
        self.blob("a/model", "b1", 5)
        watched, _ = self.monitor({"a/model": {"m.safetensors": (5, "b1")}})
        watched.list()
        self.assertEqual(watched.snapshot()[0]["state"], "cached")

    def test_without_a_listing_the_snapshot_tells(self):
        watched, resolved = self.monitor(
            {"a/model": OSError("offline"), "b/draft": OSError("offline")},
            complete=lambda repo: repo == "a/model",
        )
        self.assertEqual(
            [entry["state"] for entry in watched.snapshot()], ["cached", "listing"]
        )
        watched.list()
        self.blob("b/draft", "x.ab.incomplete", 7)
        self.assertEqual(
            watched.snapshot(),
            [
                {"repo": "a/model", "total": None, "present": None, "state": "cached"},
                {
                    "repo": "b/draft",
                    "total": None,
                    "present": 7,
                    "state": "downloading",
                },
            ],
        )
        offline, resolved = self.monitor({"a/model": {}}, offline=True)
        offline.list()
        self.assertEqual(resolved, [])
        self.assertEqual(offline.snapshot()[0]["state"], "unknown")

    def test_only_missing_repositories_are_listed(self):
        # The model is installed (not in this cache); the draft is missing.
        watched, resolved = self.monitor(
            {"a/model": {}, "b/draft": {"d.safetensors": (9, "b9")}},
            listed=("b/draft",),
        )
        watched.list()
        self.assertEqual(resolved, ["b/draft"])
        self.assertEqual(
            [(entry["repo"], entry["state"]) for entry in watched.snapshot()],
            [("a/model", "cached"), ("b/draft", "pending")],
        )

    def test_snapshot_complete_needs_every_file(self):
        cache = self.root / "hub"
        snapshot = cache / "models--o--r" / "snapshots" / ("c" * 40)
        snapshot.mkdir(parents=True)
        (cache / "models--o--r" / "refs").mkdir()
        (cache / "models--o--r" / "refs" / "main").write_text("c" * 40)
        (snapshot / "config.json").write_text("{}")
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)}):
            self.assertFalse(progress.snapshot_complete("o/r"))
            (snapshot / "model.safetensors").symlink_to(cache / "missing")
            self.assertFalse(progress.snapshot_complete("o/r"))
            (cache / "missing").write_text("weights")
            self.assertTrue(progress.snapshot_complete("o/r"))
            self.assertFalse(progress.snapshot_complete("o/absent"))


class Upstream(BaseHTTPRequestHandler):
    """A server answering as the API might: an echo of the request, and a
    chunked stream."""

    protocol_version = "HTTP/1.1"
    release = None

    def log_message(self, *args):
        pass

    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path.startswith("/api/v1/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Splash-Time-To-First-Audio-Ms", "12.5")
            self.end_headers()
            self.wfile.write(b"5\r\nfirst\r\n")
            self.wfile.flush()
            Upstream.release.wait(5)
            self.wfile.write(b"6\r\nsecond\r\n0\r\n\r\n")
            return
        data = json.dumps(
            {
                "method": self.command,
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body.decode("latin-1"),
            }
        ).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Splash-Audio-Seconds", "1.500")
        self.send_header("Set-Cookie", "upstream=1")
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = _answer


class FakeSession:
    def __init__(self, allowed_origins=()):
        self.allowed_origins = allowed_origins
        self.started = []
        self.phase = session.IDLE

    def start(self, chosen):
        if chosen.offline:
            raise session.SelectionError("not here")
        if self.started:
            raise RuntimeError("a server is already running")
        self.started.append(chosen)

    def stop(self):
        pass

    def shutdown(self):
        pass

    def snapshot(self, since=0):
        return {"phase": self.phase, "since": since}


class PageTests(unittest.TestCase):
    def setUp(self):
        Upstream.release = threading.Event()
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.upstream.daemon_threads = True
        self.page = app.PageServer(0, session_factory=FakeSession)
        for server in (self.upstream, self.page):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        self.base = f"http://127.0.0.1:{self.upstream.server_port}/api"
        self.port = self.page.server_port
        self.origin = f"http://127.0.0.1:{self.port}"

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(connection.close)
        connection.request(method, path, body=body, headers=headers or {})
        return connection.getresponse()

    def test_proxy_forwards_requests_and_answers_unchanged(self):
        answer = self.request(
            "POST",
            "/proxy/v1/audio/speech?x=1",
            body=b'{"input": "hi"}',
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer key",
                "Origin": self.origin,
                "Cookie": "page=1",
                proxy.UPSTREAM_HEADER: self.base,
            },
        )
        self.assertEqual(answer.status, 201)
        self.assertEqual(answer.getheader("X-Splash-Audio-Seconds"), "1.500")
        self.assertIsNone(answer.getheader("Set-Cookie"))
        echo = json.loads(answer.read())
        self.assertEqual(echo["method"], "POST")
        self.assertEqual(echo["path"], "/api/v1/audio/speech?x=1")
        self.assertEqual(echo["body"], '{"input": "hi"}')
        headers = echo["headers"]
        self.assertEqual(headers["authorization"], "Bearer key")
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(headers["host"], f"127.0.0.1:{self.upstream.server_port}")
        for name in ("origin", "cookie", proxy.UPSTREAM_HEADER.lower()):
            self.assertNotIn(name, headers)

    def test_proxy_streams_as_the_server_answers(self):
        answer = self.request(
            "GET", "/proxy/v1/stream", headers={proxy.UPSTREAM_HEADER: self.base}
        )
        self.assertEqual(answer.status, 200)
        self.assertEqual(answer.getheader("X-Splash-Time-To-First-Audio-Ms"), "12.5")
        # The first chunk arrives while the server still holds the second.
        self.assertEqual(answer.read1(5), b"first")
        Upstream.release.set()
        self.assertEqual(answer.read(), b"second")

    def test_proxy_refusals(self):
        unreachable = self.request(
            "GET",
            "/proxy/v1/models",
            headers={proxy.UPSTREAM_HEADER: "http://127.0.0.1:9"},
        )
        self.assertEqual(unreachable.status, 502)
        self.assertEqual(json.loads(unreachable.read())["error"]["code"], 502)
        for base in (
            None,
            "ftp://host",
            "http://user:pw@host",
            "http://host/?q=1",
            "nonsense",
        ):
            with self.subTest(base=base):
                headers = {} if base is None else {proxy.UPSTREAM_HEADER: base}
                answer = self.request("GET", "/proxy/v1/models", headers=headers)
                self.assertEqual(answer.status, 400)
                answer.read()

    def test_only_the_page_on_loopback_may_call(self):
        foreign = self.request(
            "POST",
            "/proxy/v1/audio/speech",
            body=b"{}",
            headers={"Origin": "http://evil.example", proxy.UPSTREAM_HEADER: self.base},
        )
        self.assertEqual(foreign.status, 403)
        foreign.read()
        rebound = self.request(
            "GET", "/api/state", headers={"Host": f"evil.example:{self.port}"}
        )
        self.assertEqual(rebound.status, 403)
        rebound.read()
        localhost = self.request(
            "GET",
            "/api/state?since=4",
            headers={
                "Host": f"localhost:{self.port}",
                "Origin": f"http://localhost:{self.port}",
            },
        )
        self.assertEqual(json.loads(localhost.read()), {"phase": "idle", "since": 4})

    def test_start_validates_and_starts_once(self):
        def start(body, content_type="application/json"):
            answer = self.request(
                "POST",
                "/api/start",
                body=json.dumps(body).encode(),
                headers={"Content-Type": content_type, "Origin": self.origin},
            )
            return answer.status, json.loads(answer.read())

        self.assertEqual(start({"task": "tts"})[0], 400)
        self.assertEqual(
            start({"task": "tts", "model": TTS_4BIT}, "text/plain")[0], 415
        )
        status, payload = start({"task": "tts", "model": TTS_4BIT, "offline": True})
        self.assertEqual((status, payload["error"]), (400, "not here"))
        status, payload = start({"task": "tts", "model": TTS_4BIT, "port": self.port})
        self.assertEqual(status, 400)
        self.assertIn("this page's", payload["error"])
        self.assertEqual(start({"task": "tts", "model": TTS_4BIT})[0], 200)
        self.assertEqual(self.page.session.started[0].model, TTS_4BIT)
        self.assertEqual(start({"task": "tts", "model": TTS_4BIT})[0], 409)
        self.assertEqual(
            self.page.session.allowed_origins,
            (self.origin, f"http://localhost:{self.port}"),
        )

    def test_port_probe_is_the_launchers(self):
        def probe(port):
            answer = self.request("GET", f"/api/port?port={port}")
            return answer.status, json.loads(answer.read())

        # The upstream fixture listens; the launcher would refuse its port.
        status, taken = probe(self.upstream.server_port)
        self.assertEqual((status, taken["free"]), (200, False))
        self.assertEqual(probe(self.port)[1]["reason"], "this page")
        self.assertEqual(probe(0)[0], 400)
        self.assertEqual(probe("x")[0], 400)
        with mock.patch.object(app.launcher, "_check_port") as check:
            self.assertTrue(probe(8999)[1]["free"])
        check.assert_called_once_with(app.HOST, 8999)

    def test_serves_the_page_and_nothing_else(self):
        index = self.request("GET", "/")
        self.assertEqual(index.status, 200)
        html = index.read().decode()
        for name in ("app.js", "style.css"):
            self.assertIn(f"/static/{name}", html)
            asset = self.request("GET", f"/static/{name}")
            self.assertEqual(asset.status, 200)
            asset.read()
        for path in ("/static/../app.py", "/static/.hidden", "/static/", "/app.py"):
            with self.subTest(path=path):
                answer = self.request("GET", path)
                self.assertEqual(answer.status, 404)
                answer.read()
        self.assertIsNone(app.static_file("../app.py"))

    def test_page_is_splash_omni_with_one_player_and_a_live_page(self):
        html = (app.STATIC / "index.html").read_text()
        self.assertIn("<title>Splash Omni</title>", html)
        # Voice chat is one of the start page's tasks, not a link of its
        # own; the voice chat page has no setup and returns there.
        self.assertIn('name="task" value="voice"', html)
        self.assertNotIn('href="/voice"', html)
        voice_page = (app.STATIC / "voice.html").read_text()
        self.assertNotIn('id="setup-form"', voice_page)
        self.assertNotIn('href="/"', voice_page)
        # The technical options sit under Advanced settings.
        advanced = html[html.index('id="advanced"') :]
        for name in ("port", "lanes", "max_memory", "disable_ane", "offline"):
            self.assertIn(f'name="{name}"', advanced)
        # logic.js before the page script that uses it.
        self.assertLess(
            html.index('src="/static/logic.js"'), html.index('src="/static/app.js"')
        )
        # One player for speech: no audio element beside it.
        self.assertNotIn('id="tts-audio"', html)
        self.assertIn('id="tts-player"', html)
        for element in (
            'id="variant-choices"',
            'id="tts-instructions"',
            'id="tts-speaker"',
            'id="paragraph-unavailable"',
            'id="tab-live"',
            'id="live-toggle"',
            'id="live-transcript"',
            'id="live-language"',
        ):
            self.assertIn(element, html)
        script = (app.STATIC / "app.js").read_text()
        self.assertNotIn("new Audio(", script)
        self.assertNotIn('"splash", "serve"', script)
        self.assertIn('"splash-omni", "serve"', script)

    def test_catalog_reports_installation_and_cache(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(app.paths, "MODELS", Path(directory)),
            mock.patch.object(
                upstream,
                "cached_commit",
                side_effect=lambda chosen: (
                    "a" * 40 if chosen.model == ASR_4BIT else None
                ),
            ),
            mock.patch.object(upstream, "draft_present", return_value=True),
        ):
            installed = Path(directory) / TTS_4BIT
            installed.mkdir(parents=True)
            (installed / "model.json").write_text("{}")
            catalog = app.catalog()
        by_id = {entry["id"]: entry for entry in catalog["models"]}
        self.assertEqual(list(by_id), list(families.SUPPORTED_MODELS))
        self.assertTrue(by_id[TTS_4BIT]["installed"])
        self.assertFalse(by_id[ASR_4BIT]["installed"])
        self.assertTrue(by_id[ASR_4BIT]["cached"])
        self.assertFalse(by_id[ASR_BF16]["cached"])
        self.assertEqual(
            by_id[ASR_4BIT]["draft"],
            {"id": DRAFT, "gigabytes": 0.7, "cached": True},
        )
        self.assertEqual(
            by_id[ASR_SMALL]["draft"],
            {"id": serve_options.SMALL_DRAFT_MODEL, "gigabytes": 0.25, "cached": True},
        )
        self.assertIsNone(by_id[TTS_4BIT]["draft"])
        self.assertEqual(catalog["max_lanes"], {"tts": 16, "asr": 32})
        self.assertEqual(
            [variant["id"] for variant in catalog["variants"]],
            ["base", "voice_design", "custom_voice"],
        )
        for entry in catalog["models"]:
            checkpoint = session.CHECKPOINT[entry["id"]]
            self.assertEqual(entry["variant"], checkpoint.variant)
            self.assertEqual(entry["label"], checkpoint.label)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        keep_stop_signals(self)

    def test_ui_command(self):
        args = launcher.parse_args(["ui"])
        self.assertEqual(
            (args.command, args.port, args.no_browser), ("ui", None, False)
        )
        args = launcher.parse_args(["ui", "--port", "7999", "--no-browser"])
        self.assertEqual((args.port, args.no_browser), (7999, True))
        with mock.patch.object(app, "main", return_value=0) as page:
            self.assertEqual(launcher.main(["ui", "--no-browser"]), 0)
        page.assert_called_once_with(None, open_browser=False, exit_with_parent=False)
        with mock.patch.object(app, "main", return_value=0) as page:
            launcher.main(["ui", "--no-browser", "--exit-with-parent"])
        page.assert_called_once_with(None, open_browser=False, exit_with_parent=True)
        with (
            mock.patch.object(app, "main", side_effect=OSError("in use")),
            mock.patch("sys.stderr", io.StringIO()) as error,
        ):
            self.assertEqual(launcher.main(["ui"]), 1)
        self.assertIn("cannot serve the page", error.getvalue())

    def test_page_port_falls_forward_when_taken(self):
        taken = app.PageServer(0)
        self.addCleanup(taken.server_close)
        with mock.patch.object(app, "DEFAULT_PORT", taken.server_port):
            page = app.bind()
        self.addCleanup(page.server_close)
        self.assertNotEqual(page.server_port, taken.server_port)
        with self.assertRaises(OSError):
            app.bind(taken.server_port)

    def test_page_stops_once_its_parent_exits(self):
        # The parent runs for two checks, then the process is launchd's.
        with (
            mock.patch.object(app.os, "getppid", side_effect=[41, 41, 1]),
            mock.patch.object(app.time, "sleep") as sleep,
            mock.patch.object(app.os, "kill") as kill,
        ):
            app._watch_parent(41)
        self.assertEqual(sleep.call_count, 2)
        kill.assert_called_once_with(app.os.getpid(), signal.SIGTERM)

    def test_page_stops_its_server_on_exit(self):
        page = app.PageServer(0, session_factory=FakeSession)
        page.session.shutdown = mock.Mock()

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        page.serve_forever = interrupt
        handlers = {
            number: signal.getsignal(number)
            for number in (signal.SIGINT, signal.SIGTERM)
        }
        for number, handler in handlers.items():
            self.addCleanup(signal.signal, number, handler)
        with (
            mock.patch.object(app, "bind", return_value=page),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(app.main(open_browser=False), 0)
        page.session.shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
