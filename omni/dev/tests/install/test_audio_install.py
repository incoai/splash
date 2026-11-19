"""Installing and launching the speech and transcription models: the whole
snapshot under target/, the engine's omni-check, and the audio server."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import (
    ASR_MODEL,
    TTS_MODEL,
    FakeHub,
    asr_config,
    audio_repo,
    omni_check,
    selection,
    tts_config,
)
from dev.tests.server_fixtures import keep_stop_signals
from install import assembly, families, launcher, models, upstream

COMMIT = "b" * 40


class AudioFamilyTests(unittest.TestCase):
    def test_the_audio_families_are_served_by_the_audio_server(self):
        self.assertEqual(
            [(family.name, family.model_type) for family in families.FAMILIES],
            [("qwen3-tts", "qwen3_tts"), ("qwen3-asr", "qwen3_asr")],
        )
        self.assertEqual(
            families.AUDIO_MODEL_TYPES,
            {"qwen3_tts": "qwen3-tts", "qwen3_asr": "qwen3-asr"},
        )
        self.assertEqual(families.SERVER, "server.audio_server")
        self.assertEqual(families.named("qwen3-asr"), families.FAMILIES[1])
        self.assertIsNone(families.named("Qwen3.8-27B"))
        self.assertEqual(
            families.SUPPORTED_MODELS,
            (
                "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
                "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                "mlx-community/Qwen3-ASR-1.7B-4bit",
                "mlx-community/Qwen3-ASR-1.7B-bf16",
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
                "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
                "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            ),
        )
        self.assertEqual(
            families.supported_models_text(), ", ".join(families.SUPPORTED_MODELS)
        )


class AudioInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"
        self.fake = FakeHub(self, self.cache)

    def prepare(self, chosen):
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()) as warnings,
        ):
            upstream.prepare(chosen)
        return output.getvalue(), warnings.getvalue()

    def test_a_speech_model_installs_its_whole_snapshot(self):
        self.fake.publish(
            TTS_MODEL,
            COMMIT,
            lambda p: audio_repo(p, tts_config(), speech_tokenizer=True),
        )
        calls, run = omni_check("qwen3-tts")
        chosen = selection(self.root, TTS_MODEL)
        with mock.patch.object(models, "run_engine", side_effect=run):
            output, warnings = self.prepare(chosen)
        self.assertEqual(warnings, "")
        self.assertIn("Installing " + TTS_MODEL + " as qwen3-tts (mlx-affine).", output)
        # One Hub request, the target's.
        self.assertEqual(self.fake.requests, [(TTS_MODEL, None)])
        snapshot = self.fake.snapshot(TTS_MODEL, COMMIT)
        self.assertEqual(calls, [["omni-check", str(snapshot)]])
        record = assembly.verify(chosen.link)
        self.assertEqual(record["family"], "qwen3-tts")
        self.assertEqual(record["target_format"], "mlx-affine")
        self.assertNotIn("vision_format", record)
        self.assertEqual(
            record["sources"], {"target": {"repo": TTS_MODEL, "revision": COMMIT}}
        )
        self.assertFalse((chosen.link / "draft").exists())
        self.assertFalse((chosen.link / "vision").exists())
        # Every file of the snapshot under target/, the speech tokenizer's
        # subdirectory included; the tokenizer files; config.json at the root.
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
            "preprocessor_config.json",
            "generation_config.json",
            "README.md",
            "config.json",
        ):
            self.assertEqual(
                (chosen.link / "target" / name).readlink(), snapshot / name
            )
        for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
            self.assertEqual(
                (chosen.link / "tokenizer" / name).readlink(), snapshot / name
            )
        for name in ("config.json", "tokenizer/config.json"):
            self.assertEqual((chosen.link / name).readlink(), snapshot / "config.json")
        self.assertEqual(
            (chosen.link / "target/speech_tokenizer/model.safetensors").read_bytes(),
            b"codec",
        )
        # A second start finds the installation, without asking the Hub.
        self.fake.clear()
        with mock.patch.object(models, "run_engine", side_effect=run):
            output, _ = self.prepare(chosen)
        self.assertIn("already installed", output)
        self.assertEqual(self.fake.calls(), [])

    def test_an_unquantized_transcription_model_records_bf16(self):
        self.fake.publish(ASR_MODEL, COMMIT, lambda p: audio_repo(p, asr_config()))
        _, run = omni_check("qwen3-asr")
        chosen = selection(self.root, ASR_MODEL)
        with mock.patch.object(models, "run_engine", side_effect=run):
            self.prepare(chosen)
        record = assembly.verify(chosen.link)
        self.assertEqual(record["family"], "qwen3-asr")
        self.assertEqual(record["target_format"], "mlx-bf16")
        self.assertEqual(set(record["sources"]), {"target"})

    def test_an_engine_without_the_omni_check_takes_the_family_from_config(self):
        self.fake.publish(TTS_MODEL, COMMIT, lambda p: audio_repo(p, tts_config()))
        chosen = selection(self.root, TTS_MODEL)
        stale = models.UnknownCheck("the engine splash does not know this omni check")

        def run(arguments, check):
            raise stale

        with mock.patch.object(models, "run_engine", side_effect=run):
            _, warnings = self.prepare(chosen)
        self.assertIn("does not know this omni check", warnings)
        self.assertIn("taking the family from config.json", warnings)
        self.assertEqual(assembly.verify(chosen.link)["family"], "qwen3-tts")

    def test_the_engines_refusal_stops_the_install_before_weights(self):
        self.fake.publish(TTS_MODEL, COMMIT, lambda p: audio_repo(p, tts_config()))
        chosen = selection(self.root, TTS_MODEL)

        def run(arguments, check):
            raise models.ModelError("talker hidden size 1024 is not supported")

        with (
            mock.patch.object(models, "run_engine", side_effect=run),
            self.assertRaisesRegex(models.ModelError, "talker hidden size"),
        ):
            self.prepare(chosen)
        self.assertEqual(self.fake.downloads, [f"{TTS_MODEL}/config.json"])
        with (
            mock.patch.object(
                models, "run_engine", return_value=json.dumps({"family": "Qwen3.8-27B"})
            ),
            self.assertRaisesRegex(models.ModelError, "no audio model family"),
        ):
            self.prepare(chosen)

    def test_a_record_of_the_target_alone_verifies_and_pins(self):
        self.fake.publish(TTS_MODEL, COMMIT, lambda p: audio_repo(p, tts_config()))
        _, run = omni_check("qwen3-tts")
        chosen = selection(self.root, TTS_MODEL)
        with mock.patch.object(models, "run_engine", side_effect=run):
            self.prepare(chosen)
        record = assembly.verify(chosen.link)
        self.assertEqual(
            assembly.pins(record), [(self.fake.snapshot(TTS_MODEL, COMMIT), TTS_MODEL)]
        )
        self.assertTrue(assembly._well_formed(record))
        # A draft, which the language families had, is no source of an
        # audio model.
        record["sources"]["draft"] = {"repo": "x/y", "revision": None}
        self.assertFalse(assembly._well_formed(record))


class AudioLaunchTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop("SPLASH_PORT", None)
        keep_stop_signals(self)

    def test_serve_executes_the_audio_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            root = Path(temporary) / "assembly"
            root.mkdir()
            (root / "model.json").write_text(json.dumps({"family": "qwen3-asr"}))
            with (
                mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                mock.patch.object(launcher.socket, "socket"),
                mock.patch.object(launcher, "_ensure_installed"),
                mock.patch.object(launcher.assembly, "hold", return_value=(root, None)),
                mock.patch.object(launcher.os, "execve") as execute,
            ):
                launcher.main(["serve", "--model", ASR_MODEL, "--lanes", "4"])
            argv = execute.call_args.args[1]
            self.assertIn("server.audio_server", argv)
            self.assertNotIn("server.server", argv)
            self.assertEqual(argv[argv.index("-m") + 1], "server.audio_server")
            self.assertEqual(argv[argv.index("server.audio_server") + 1], str(root))
            self.assertIn("--lanes=4", argv)
            self.assertIn("--tokenizer", argv)
            self.assertEqual(
                argv[argv.index("--tokenizer") + 1], str(root / "tokenizer")
            )


if __name__ == "__main__":
    unittest.main()
