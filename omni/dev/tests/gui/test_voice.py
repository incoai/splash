"""The voice chat's servers (gui/voice.py) and the page's routes for them."""

import http.client
import json
import os
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from gui import app, session, voice


def executable(directory, name="splash"):
    path = Path(directory) / name
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.splash = executable(self.directory.name)
        # An empty Hub cache, not this machine's.
        cache = mock.patch.dict(
            os.environ, {"HF_HUB_CACHE": self.directory.name + "/empty"}
        )
        cache.start()
        self.addCleanup(cache.stop)

    def test_defaults_name_three_servers_on_their_own_ports(self):
        chosen = voice.selections({"splash": self.splash, "llm_max_memory": "30G"})
        self.assertEqual(chosen[voice.ASR].model, voice.DEFAULT_ASR)
        self.assertEqual(chosen[voice.ASR].task, session.ASR)
        self.assertTrue(chosen[voice.ASR].speculative)
        self.assertEqual(chosen[voice.TTS].model, voice.DEFAULT_TTS)
        llm = chosen[voice.LLM]
        self.assertEqual(llm.model, voice.DEFAULT_LLM)
        self.assertEqual(
            [chosen[role].port for role in voice.ROLES],
            [voice.DEFAULT_PORTS[role] for role in voice.ROLES],
        )
        self.assertEqual(
            llm.command(("http://127.0.0.1:7860",), offline=True),
            [
                self.splash,
                "serve",
                "--model",
                voice.DEFAULT_LLM,
                "--port",
                str(voice.DEFAULT_PORTS[voice.LLM]),
                "--language-only",
                "--idle-release",
                "off",
                "--offline",
                "--max-memory",
                "30G",
                "--allowed-origin",
                "http://127.0.0.1:7860",
            ],
        )
        self.assertTrue(llm.shown_command().startswith("splash serve --model "))
        self.assertEqual(llm.downloads(), ())

    def test_splash_runs_from_its_checkout_without_this_ones_path(self):
        llm = voice.selections({"splash": self.splash})[voice.LLM]
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/elsewhere"}):
            environment, cwd = llm.process_environment()
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(cwd, str(Path(self.splash).resolve().parent))

    def test_refused_setups(self):
        cases = (
            (
                {"asr_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"},
                "does not serve",
            ),
            ({"tts_port": voice.DEFAULT_PORTS[voice.ASR]}, "different ports"),
            ({"llm_port": 0}, "llm_port"),
            ({"llm_model": " "}, "llm_model"),
            ({"llm_max_memory": "lots"}, "llm_max_memory"),
            ({"splash": "/nonexistent/splash"}, "splash command"),
        )
        for change, message in cases:
            with self.subTest(change=change):
                body = {"splash": self.splash, **change}
                with self.assertRaises(session.SelectionError) as refused:
                    voice.selections(body)
                self.assertIn(message, str(refused.exception))

    def test_speech_servers_take_the_memory_their_models_need(self):
        # The engine refuses a --max-memory its model does not fit in, and
        # fits the DFlash draft only where the budget holds it.
        cases = (
            ({}, "8G", "5G"),
            ({"speculative": False}, "5G", "5G"),
            (
                {
                    "asr_model": "mlx-community/Qwen3-ASR-1.7B-bf16",
                    "tts_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                },
                "10G",
                "7G",
            ),
            (
                {
                    "asr_model": "mlx-community/Qwen3-ASR-1.7B-bf16",
                    "speculative": False,
                },
                "7G",
                "5G",
            ),
            ({"speech_max_memory": "12G"}, "12G", "12G"),
        )
        for change, asr, tts in cases:
            with self.subTest(change=change):
                chosen = voice.selections(
                    {"splash": self.splash, "llm_max_memory": "30G", **change}
                )
                self.assertEqual(
                    (chosen[voice.ASR].max_memory, chosen[voice.TTS].max_memory),
                    (asr, tts),
                )

    def test_the_language_model_takes_what_the_speech_servers_leave(self):
        # 48 GiB: the GPU's three quarters, 36 GiB, less both speech budgets
        # and the 1.5 GiB margin.
        with mock.patch.object(voice, "total_memory_gib", return_value=48):
            self.assertEqual(voice.default_llm_memory(), "21G")
            self.assertEqual(voice.default_llm_memory(speculative=False), "24G")
            chosen = voice.selections(
                {
                    "splash": self.splash,
                    "asr_model": "mlx-community/Qwen3-ASR-1.7B-bf16",
                    "tts_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                }
            )
            self.assertEqual(chosen[voice.LLM].max_memory, "17G")
        with mock.patch.object(voice, "total_memory_gib", return_value=16):
            self.assertIsNone(voice.default_llm_memory())

    def test_splash_is_found_by_environment_then_checkout_then_path(self):
        checkout = Path(self.directory.name) / "checkout"
        (checkout / "omni").mkdir(parents=True)
        launcher = executable(str(checkout))
        with (
            mock.patch.object(voice.paths, "ROOT", checkout / "omni"),
            mock.patch.object(voice.shutil, "which", return_value=self.splash),
        ):
            with mock.patch.dict(os.environ, {"SPLASH_BIN": self.splash}):
                self.assertEqual(voice.find_splash(), self.splash)
            with mock.patch.dict(os.environ, {"SPLASH_BIN": ""}):
                self.assertEqual(voice.find_splash(), launcher)
                os.unlink(launcher)
                self.assertEqual(voice.find_splash(), self.splash)

    def test_the_splash_apps_runtime_runs_without_a_command(self):
        runtime = Path(self.directory.name) / "Splash.app/Contents/Resources/splash"
        (runtime / "install").mkdir(parents=True)
        (runtime / "python/bin").mkdir(parents=True)
        (runtime / "install/launcher.py").write_text("")
        executable(str(runtime / "python/bin"), "python3")
        llm = voice.selections({"splash": str(runtime)})[voice.LLM]
        self.assertEqual(
            llm.command()[:4],
            [
                str(runtime / "python/bin/python3"),
                "-u",
                str(runtime / "install/launcher.py"),
                "serve",
            ],
        )
        environment, cwd = llm.process_environment()
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(cwd, str(runtime.resolve()))
        with self.assertRaises(session.SelectionError):
            voice.selections({"splash": str(runtime.parent)})

    def test_cached_language_models_are_found_in_the_hub_cache(self):
        hub = Path(self.directory.name) / "hub"
        gguf = hub / "models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/abc"
        gguf.mkdir(parents=True)
        (gguf / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf").write_bytes(b"")
        mlx = hub / "models--mlx-community--Qwen3.8-27B-4bit/snapshots/def"
        mlx.mkdir(parents=True)
        (mlx / "config.json").write_text("{}")
        draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/0123"
        draft.mkdir(parents=True)
        (draft / "config.json").write_text("{}")
        (draft / "model.safetensors").write_bytes(b"")
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
            self.assertTrue(voice.llm_cached("unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"))
            self.assertFalse(voice.llm_cached("unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0"))
            self.assertTrue(voice.llm_cached("mlx-community/Qwen3.8-27B-4bit"))
            self.assertFalse(voice.llm_cached("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"))
            self.assertEqual(
                voice.draft_directory("mlx-community/Qwen3.8-27B-4bit"), str(draft)
            )
            self.assertIsNone(
                voice.draft_directory("unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M")
            )
            chosen = voice.selections(
                {"splash": self.splash, "llm_model": "unsloth/Qwen3.8-27B-GGUF:Q4"}
            )
            self.assertEqual(chosen[voice.LLM].draft, str(draft))
            self.assertIn("--draft-model", chosen[voice.LLM].command())
            self.assertTrue(voice._llm_plan(chosen[voice.LLM]).offline is False)
            plan = voice._llm_plan(
                voice.LlmSelection(
                    "mlx-community/Qwen3.8-27B-4bit", self.splash, draft=str(draft)
                )
            )
            self.assertTrue(plan.offline)
            # Without its draft, Splash resolves and fetches it online.
            plan = voice._llm_plan(
                voice.LlmSelection("mlx-community/Qwen3.8-27B-4bit", self.splash)
            )
            self.assertFalse(plan.offline)


class FakeSession:
    def __init__(self, allowed_origins=(), planner=None):
        self.planner = planner
        self.phase = session.IDLE
        self.selection = None
        self.stopped = 0

    def start(self, chosen):
        if getattr(chosen, "model", "").endswith("bf16") and chosen.task == session.TTS:
            raise session.SelectionError("not here")
        self.selection = chosen
        self.phase = session.PREPARING

    def stop(self):
        self.stopped += 1
        self.phase = session.IDLE

    def shutdown(self):
        self.stop()

    def snapshot(self, since=0):
        return {"phase": self.phase, "since": since}


class StackTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.splash = executable(directory.name)

    def test_start_runs_all_three_and_refuses_a_second_start(self):
        stack = voice.VoiceStack(session_factory=FakeSession)
        self.assertIs(stack.sessions[voice.LLM].planner, voice._llm_plan)
        stack.start(voice.selections({"splash": self.splash}))
        self.assertTrue(stack.running)
        self.assertEqual(
            {role: entry.selection.model for role, entry in stack.sessions.items()},
            {
                voice.ASR: voice.DEFAULT_ASR,
                voice.TTS: voice.DEFAULT_TTS,
                voice.LLM: voice.DEFAULT_LLM,
            },
        )
        with self.assertRaises(RuntimeError):
            stack.start(voice.selections({"splash": self.splash}))
        stack.stop()
        self.assertFalse(stack.running)
        self.assertEqual(stack.snapshot({voice.TTS: 4})[voice.TTS]["since"], 4)

    def test_a_failed_start_stops_what_it_started(self):
        stack = voice.VoiceStack(session_factory=FakeSession)
        chosen = voice.selections(
            {
                "splash": self.splash,
                "tts_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            }
        )
        with self.assertRaises(session.SelectionError):
            stack.start(chosen)
        self.assertEqual(stack.sessions[voice.ASR].stopped, 1)
        self.assertFalse(stack.running)


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.02)


class RouteTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.splash = executable(directory.name)
        self.page = app.PageServer(
            0,
            session_factory=FakeSession,
            voice_factory=lambda allowed_origins: voice.VoiceStack(
                allowed_origins=allowed_origins, session_factory=FakeSession
            ),
        )
        threading.Thread(target=self.page.serve_forever, daemon=True).start()
        self.addCleanup(self.page.server_close)
        self.addCleanup(self.page.shutdown)

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.page.server_port, timeout=10
        )
        self.addCleanup(connection.close)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        data = None if body is None else json.dumps(body).encode()
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()

    def test_page_catalog_start_state_and_stop(self):
        status, page = self.request("GET", "/voice")
        self.assertEqual(status, 200)
        self.assertIn(b"voice.js", page)
        status, data = self.request("GET", "/api/voice/catalog")
        self.assertEqual(status, 200)
        catalog = json.loads(data)
        self.assertEqual(
            [entry["id"] for entry in catalog["llm_models"]], list(voice.LLM_MODELS)
        )
        self.assertIn("speech", catalog)
        with mock.patch.object(app, "port_status", return_value={"free": True}):
            status, data = self.request(
                "POST", "/api/voice/start", {"splash": self.splash}
            )
        self.assertEqual(status, 200, data)
        self.assertEqual(json.loads(data)[voice.LLM]["phase"], session.PREPARING)
        status, data = self.request("GET", "/api/voice/state?asr=3&tts=x")
        state = json.loads(data)
        self.assertEqual((state[voice.ASR]["since"], state[voice.TTS]["since"]), (3, 0))
        status, data = self.request("POST", "/api/voice/stop", {})
        self.assertEqual(json.loads(data)[voice.ASR]["phase"], session.IDLE)

    def test_one_features_servers_at_a_time(self):
        speech = {
            "task": "tts",
            "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
            "port": 8040,
        }
        with mock.patch.object(app, "port_status", return_value={"free": True}):
            status, data = self.request(
                "POST", "/api/voice/start", {"splash": self.splash}
            )
        self.assertEqual(status, 200, data)
        status, data = self.request("POST", "/api/start", speech)
        self.assertEqual((status, json.loads(data)["error"]), (409, app.VOICE_RUNNING))
        self.request("POST", "/api/voice/stop", {})
        status, data = self.request("POST", "/api/start", speech)
        self.assertEqual(status, 200, data)
        status, data = self.request("POST", "/api/voice/start", {"splash": self.splash})
        self.assertEqual((status, json.loads(data)["error"]), (409, app.SPEECH_RUNNING))

    def test_a_start_on_a_taken_port_is_refused(self):
        body = {"splash": self.splash, "asr_port": self.page.server_port}
        status, data = self.request("POST", "/api/voice/start", body)
        self.assertEqual(status, 400)
        self.assertIn("this page's", json.loads(data)["error"])
        with mock.patch.object(
            app,
            "port_status",
            return_value={"free": False, "reason": "Address already in use"},
        ):
            status, data = self.request(
                "POST", "/api/voice/start", {"splash": self.splash}
            )
        self.assertEqual(status, 400)
        self.assertIn("in use", json.loads(data)["error"])


if __name__ == "__main__":
    unittest.main()
