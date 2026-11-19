import contextlib
import errno
import fcntl
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from dev.tests.installer_fixtures import (
    ASR_MODEL,
    COMMIT,
    MODEL,
    REPO_FILES,
    TTS_MODEL,
    FakeHub,
    asr_config,
    audio_repo,
    cached_model,
    cached_snapshot,
    engine_patch,
    fake_hub,
    http_error,
    hub_cache,
    listing,
    no_hub,
    pins,
    selection,
    tts_config,
)
from install import assembly, families, hub, models, paths, upstream

# The installer's refusal of a repository Splash Omni does not serve, which
# names the supported checkpoints.
SUPPORTED = "the supported checkpoints are " + ", ".join(families.SUPPORTED_MODELS)
FOLDER = "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-4bit"
# The engine check as the installer runs it, before a test stubs it.
RUN_ENGINE = models.run_engine


class UpstreamTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"
        # The engine's omni-check names the TTS family unless a test says
        # otherwise.
        self.enterContext(engine_patch("qwen3-tts"))

    @staticmethod
    def prepare(chosen):
        """upstream.prepare's output and warnings."""
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()) as warnings,
        ):
            upstream.prepare(chosen)
        return output.getvalue(), warnings.getvalue()

    def test_the_supported_checkpoints_are_the_two_families(self):
        self.assertEqual(len(families.SUPPORTED_MODELS), 8)
        for model in families.SUPPORTED_MODELS:
            with self.subTest(model=model):
                self.assertEqual(models.validate_repo_id(model), model)
        self.assertEqual(
            [f.name for f in families.FAMILIES], ["qwen3-tts", "qwen3-asr"]
        )
        self.assertEqual(
            families.AUDIO_MODEL_TYPES,
            {"qwen3_tts": "qwen3-tts", "qwen3_asr": "qwen3-asr"},
        )
        self.assertEqual(families.SERVER, "server.audio_server")
        self.assertIsNone(families.named("Qwen3.8-27B"))

    def test_a_model_of_another_family_is_refused_before_weight_downloads(self):
        # A language model, whatever its repository is called, and a
        # transformers release of the audio model with no quantization: the
        # configuration tells them apart, before anything but config.json
        # is fetched.
        fake = FakeHub(self, self.cache)
        for model, config, reason in (
            (
                "mlx-community/Qwen3.8-27B-4bit",
                {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
                "its model_type is 'qwen3'",
            ),
            (
                "someone/fine-tune",
                {"architectures": ["Qwen3TTSForConditionalGeneration"]},
                "its model_type is None",
            ),
        ):
            fake.publish(model, "a" * 40, lambda p, c=config: audio_repo(p, c))
            fake.requests.clear(), fake.downloads.clear()
            with (
                self.subTest(model=model),
                self.assertRaises(models.ModelError) as refused,
            ):
                self.prepare(selection(self.root, model))
            self.assertEqual(
                str(refused.exception),
                f"{model} is not a model Splash Omni serves ({reason}); {SUPPORTED}",
            )
            self.assertEqual(fake.requests, [(model, None)])
            self.assertEqual(fake.downloads, [f"{model}/config.json"])
            self.assertFalse(selection(self.root, model).link.exists())

    def test_a_repository_without_a_checkpoint_is_refused_before_any_download(self):
        # A GGUF repository, which the language engine served, and a package
        # of another kind.
        fake = FakeHub(self, self.cache)
        for model, names in (
            ("unsloth/Qwen3.8-27B-GGUF", ("Qwen3.8-27B-UD-Q4_K_M.gguf", "config.json")),
            ("incoai/Qwen3.8-27B-Splash", ("manifest.json", "target/layer-0.bin")),
        ):
            fake.publish(
                model,
                "a" * 40,
                lambda p, names=names: [
                    (p / n).parent.mkdir(parents=True, exist_ok=True)
                    or (p / n).write_text("{}")
                    for n in names
                ],
            )
            fake.downloads.clear()
            with (
                self.subTest(model=model),
                self.assertRaises(models.ModelError) as refused,
            ):
                self.prepare(selection(self.root, model))
            self.assertEqual(
                str(refused.exception),
                f"{model} is not a model Splash Omni serves (it has no safetensors "
                f"checkpoint); {SUPPORTED}",
            )
            self.assertEqual(fake.downloads, [])

    def test_the_engines_refusal_stops_the_install_before_weights(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)

        def run(arguments, check):
            raise models.ModelError("talker hidden size 1024 is not supported")

        with (
            mock.patch.object(models, "run_engine", side_effect=run),
            self.assertRaisesRegex(models.ModelError, "talker hidden size"),
        ):
            self.prepare(chosen)
        self.assertEqual(fake.downloads, [f"{MODEL}/config.json"])
        with (
            mock.patch.object(
                models, "run_engine", return_value=json.dumps({"family": "Qwen3.8-27B"})
            ),
            self.assertRaisesRegex(models.ModelError, "no audio model family"),
        ):
            self.prepare(chosen)
        self.assertFalse(chosen.link.exists())

    def test_an_engine_without_the_omni_check_takes_the_family_from_config(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        stale = models.UnknownCheck("the engine splash does not know this omni check")
        with mock.patch.object(models, "run_engine", side_effect=stale):
            _, warnings = self.prepare(chosen)
        self.assertIn("does not know this omni check", warnings)
        self.assertIn("taking the family from config.json", warnings)
        self.assertEqual(assembly.verify(chosen.link)["family"], "qwen3-tts")

    def test_an_engine_that_cannot_run_is_named(self):
        # install/models.py prepare, run by hand, may find no engine.
        fake_hub(self, self.cache)
        missing = self.root / "missing/splash"
        with (
            mock.patch.object(models, "run_engine", RUN_ENGINE),
            mock.patch.object(paths, "BINARY", missing),
            self.assertRaises(models.ModelError) as refused,
        ):
            self.prepare(selection(self.root))
        self.assertEqual(
            str(refused.exception),
            f"cannot run the engine {missing} (No such file or directory); "
            "rebuild it with make",
        )

    def test_missing_metadata_never_falls_back_to_another_repository(self):
        required = ("config.json", "tokenizer.json", "tokenizer_config.json")
        fake = FakeHub(self, self.cache)
        for missing in required:
            with self.subTest(missing=missing):
                fake.publish(
                    "user/custom",
                    "a" * 40,
                    lambda p: (
                        p.mkdir(parents=True, exist_ok=True),
                        [
                            (p / name).write_text(json.dumps(tts_config()))
                            for name in ("model.safetensors", *required)
                            if name != missing
                        ],
                    ),
                )
                chosen = selection(self.root, "user/custom")
                with self.assertRaisesRegex(
                    models.ModelError, "must come from the target repository"
                ):
                    self.prepare(chosen)
                self.assertEqual(fake.downloads, [])
                self.assertFalse(chosen.models_root.exists())
                shutil.rmtree(fake.remote)

    def test_a_speech_model_installs_its_whole_snapshot(self):
        fake = FakeHub(self, self.cache)
        fake.publish(MODEL, COMMIT, lambda p: audio_repo(p, speech_tokenizer=True))
        chosen = selection(self.root)
        output, warnings = self.prepare(chosen)
        self.assertEqual(warnings, "")
        self.assertIn(f"Installing {MODEL} as qwen3-tts (mlx-affine).", output)
        self.assertEqual(fake.requests, [(MODEL, None)])
        snapshot = fake.snapshot(MODEL, COMMIT)
        record = assembly.verify(chosen.link)
        self.assertEqual(record["family"], "qwen3-tts")
        self.assertEqual(record["target_format"], "mlx-affine")
        self.assertEqual(
            record["sources"], {"target": {"repo": MODEL, "revision": COMMIT}}
        )
        self.assertEqual(
            set(record) - {"files"},
            {"version", "model", "family", "target_format", "sources"},
        )
        # Every file of the snapshot under target/, the speech tokenizer's
        # subdirectory included; the tokenizer files; config.json at the root.
        for name in (
            *REPO_FILES,
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
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
        self.assertFalse((chosen.link / "draft").exists())
        self.assertFalse((chosen.link / "vision").exists())
        self.assertEqual(listing(chosen.link.resolve()), listing(chosen.link))

    def test_an_unquantized_transcription_model_records_bf16(self):
        fake = FakeHub(self, self.cache)
        fake.publish(ASR_MODEL, COMMIT, lambda p: audio_repo(p, asr_config()))
        chosen = selection(self.root, ASR_MODEL)
        with engine_patch("qwen3-asr"):
            self.prepare(chosen)
        record = assembly.verify(chosen.link)
        self.assertEqual(record["family"], "qwen3-asr")
        self.assertEqual(record["target_format"], "mlx-bf16")
        self.assertEqual(set(record["sources"]), {"target"})

    def test_a_cached_file_is_read_from_the_cache(self):
        fake = fake_hub(self, self.cache)
        repo = hub.Repository.resolve(MODEL)
        with contextlib.redirect_stdout(io.StringIO()):
            repo.download({"config.json"})
        with repo.open("config.json") as stream:
            self.assertEqual(json.load(stream)["quantization_config"]["bits"], 4)
        with repo.open("tokenizer_config.json") as stream:
            self.assertEqual(stream.read(), b"{}")
        self.assertEqual(fake.range_reads, [f"{MODEL}/tokenizer_config.json"])

    def test_an_installed_model_starts_without_asking_the_hub(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        output, _ = self.prepare(chosen)
        count = len(REPO_FILES) + 2
        self.assertIn(f"Fetching {count} file(s), 0.00 GB, from " + MODEL, output)
        installed = chosen.link.resolve()
        # The branch moves on the Hub, and the Hub fails every request: the
        # installation starts as it is, and nothing asks the Hub.
        fake.publish(MODEL, "b" * 40, lambda p: audio_repo(p))
        fake.failure = fake.download_failure = httpx.ConnectTimeout("")
        fake.clear()
        with no_hub():
            output, warnings = self.prepare(chosen)
        self.assertEqual(
            output, f"Splash model {MODEL} is already installed in {chosen.link}\n"
        )
        self.assertEqual(warnings, "")
        self.assertEqual(fake.calls(), [])
        self.assertEqual(chosen.link.resolve(), installed)
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["target"]["revision"], COMMIT
        )

    def test_a_model_in_the_hub_cache_is_installed_without_asking_the_hub(self):
        # As `hf download` leaves it: refs/main names a whole snapshot. The
        # Hub has another commit, which is never asked for.
        fake = fake_hub(self, self.cache, commit="b" * 40)
        cached_model(self.cache, MODEL, COMMIT, lambda p: audio_repo(p))
        chosen = selection(self.root)
        with no_hub():
            output, warnings = self.prepare(chosen)
        self.assertIn(
            f"Using {MODEL}@{'a' * 12} from the Hugging Face cache; the Hub is "
            "not asked.\n",
            output,
        )
        self.assertIn(f"Installing {MODEL} as qwen3-tts (mlx-affine).", output)
        self.assertNotIn("Fetching", output)
        self.assertEqual(warnings, "")
        self.assertEqual(fake.calls(), [])
        record = assembly.verify(chosen.link)
        self.assertEqual(
            record["sources"]["target"], {"repo": MODEL, "revision": COMMIT}
        )
        self.assertEqual(pins(self.cache), [COMMIT])
        # The next start finds it installed.
        with no_hub():
            output, _ = self.prepare(chosen)
        self.assertIn("is already installed", output)

    def test_a_model_not_here_is_downloaded_alone(self):
        # Another supported model is on the Hub and a third in the cache;
        # only the selected one is listed and downloaded.
        fake = fake_hub(self, self.cache)
        other = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        fake.publish(other, "c" * 40, lambda p: audio_repo(p, tts_config(False)))
        cached_model(
            self.cache, TTS_MODEL.replace("4bit", "8bit"), "d" * 40, audio_repo
        )
        chosen = selection(self.root)
        self.prepare(chosen)
        self.assertEqual(fake.requests, [(MODEL, None)])
        self.assertEqual(
            sorted(fake.downloads),
            sorted(f"{MODEL}/{name}" for name in listing(fake.remote / MODEL / COMMIT)),
        )
        self.assertEqual(
            sorted(p.name for p in self.cache.iterdir()),
            sorted(
                [
                    hub.folder_name(MODEL),
                    hub.folder_name(TTS_MODEL.replace("4bit", "8bit")),
                ]
            ),
        )

    def test_a_partial_snapshot_in_the_cache_is_downloaded(self):
        # An interrupted download: refs/main names a snapshot that lacks a
        # shard its index names, or whose file has no blob.
        fake = fake_hub(self, self.cache)
        snapshot = cached_model(self.cache, MODEL, COMMIT, lambda p: audio_repo(p))
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"w": "model-00002.safetensors"}})
        )
        self.assertIsNone(hub.cached_commit(MODEL))
        (snapshot / "model.safetensors.index.json").write_text("{}")
        self.assertEqual(hub.cached_commit(MODEL), COMMIT)
        (snapshot / "README.md").unlink()
        (snapshot / "README.md").symlink_to(self.root / "missing-blob")
        self.assertIsNone(hub.cached_commit(MODEL))
        chosen = selection(self.root)
        self.prepare(chosen)
        self.assertEqual(fake.requests, [(MODEL, None)])
        assembly.verify(chosen.link)

    def test_publishing_removes_what_no_installation_uses(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        first = chosen.link.resolve()
        # A server holds the assembly it serves.
        _, server = assembly.hold(chosen.link, chosen.models_root)
        self.addCleanup(server.close)
        # Staging an interrupted installation left behind.
        stale = [
            chosen.models_root / ".resolved/.loading-x",
            chosen.models_root
            / "mlx-community/.prepare-Qwen3-TTS-12Hz-1.7B-Base-4bit-z",
            chosen.models_root / ".selections/.prepare-w",
        ]
        for path in stale:
            path.mkdir(parents=True)
        snapshot = hub.snapshot(MODEL, COMMIT)

        def touch():
            # A changed source file: the assembly no longer verifies, and
            # its reinstallation records the file anew.
            stat = (snapshot / "README.md").stat()
            os.utime(
                snapshot / "README.md", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1)
            )

        touch()
        output, _ = self.prepare(chosen)
        self.assertIn("Reinstalling", output)
        second = chosen.link.resolve()
        self.assertEqual(
            sorted(p.name for p in (chosen.models_root / ".resolved").iterdir()),
            sorted([first.name, second.name]),
        )
        self.assertFalse(any(path.exists() for path in stale))
        # Released by its server and linked by no selection, it goes next.
        server.close()
        touch()
        self.prepare(chosen)
        self.assertEqual(
            [p.name for p in (chosen.models_root / ".resolved").iterdir()],
            [chosen.link.resolve().name],
        )
        assembly.verify(chosen.link)

    def test_unreachable_hub_without_the_model_here_is_the_error(self):
        fake = fake_hub(self, self.cache)
        for failure, words in (
            (http_error(404), "404 Client Error"),
            (http_error(401), "401 Client Error.*; set HF_TOKEN"),
            (httpx.ConnectTimeout("timed out"), "timed out"),
        ):
            fake.failure = failure
            with (
                self.subTest(failure=failure),
                self.assertRaisesRegex(
                    models.ModelError,
                    "cannot download someone/other, which is neither installed "
                    f"nor in the Hugging Face cache: {words}",
                ),
            ):
                self.prepare(selection(self.root, "someone/other"))
        self.assertFalse(selection(self.root, "someone/other").link.exists())

    def test_offline_starts_what_is_here_and_refuses_what_is_not(self):
        fake = fake_hub(self, self.cache)
        cached_model(self.cache, ASR_MODEL, COMMIT, lambda p: audio_repo(p))
        self.prepare(selection(self.root))
        fake.clear()
        with (
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
            no_hub(),
        ):
            # Installed, and in the cache.
            output, _ = self.prepare(selection(self.root))
            self.assertIn("is already installed", output)
            output, _ = self.prepare(selection(self.root, ASR_MODEL))
            self.assertIn("from the Hugging Face cache", output)
            for chosen, wanted in (
                (selection(self.root, "someone/other"), "someone/other"),
                (selection(self.root, revision="v2"), f"{MODEL}@v2"),
            ):
                with (
                    self.subTest(wanted=wanted),
                    self.assertRaises(models.ModelError) as refused,
                ):
                    self.prepare(chosen)
                self.assertEqual(
                    str(refused.exception),
                    f"{wanted} is neither installed nor in the Hugging Face "
                    "cache, and HF_HUB_OFFLINE is set (splash-omni serve --offline); "
                    "start without it to download the model",
                )
        self.assertEqual(fake.calls(), [])

    def test_a_revision_starts_from_the_cache_or_downloads_that_revision(self):
        fake = FakeHub(self, self.cache)
        fake.publish(MODEL, COMMIT, lambda p: audio_repo(p))
        fake.publish(MODEL, "b" * 40, lambda p: audio_repo(p), branch="v2")
        # A commit in the cache, and a branch the cache records.
        cached_snapshot(self.cache, MODEL, "c" * 40, lambda p: audio_repo(p))
        cached_model(self.cache, MODEL, "d" * 40, lambda p: audio_repo(p), branch="v3")
        for revision, commit in (("c" * 40, "c" * 40), ("v3", "d" * 40)):
            with self.subTest(revision=revision), no_hub():
                chosen = selection(self.root, revision=revision)
                self.prepare(chosen)
                self.assertEqual(
                    assembly.verify(chosen.link)["sources"]["target"]["revision"],
                    commit,
                )
        self.assertEqual(fake.calls(), [])
        # A revision the cache lacks is downloaded, that revision alone.
        chosen = selection(self.root, revision="v2")
        self.prepare(chosen)
        self.assertEqual(fake.requests, [(MODEL, "v2")])
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["target"]["revision"], "b" * 40
        )
        self.assertTrue(all(name.startswith(MODEL) for name in fake.downloads))
        fake.clear()
        with no_hub():
            self.prepare(chosen)
            self.prepare(selection(self.root, revision="b" * 40))
        self.assertEqual(fake.calls(), [])

    def test_a_moved_hub_cache_starts_the_installation_it_links(self):
        # HF_HUB_CACHE may name another folder than the one an installation
        # was built from; its links still name its files there.
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        pinned = selection(self.root, revision="a" * 40)
        self.prepare(chosen)
        self.prepare(pinned)
        installed = pins(self.cache)
        moved = self.root / "moved"
        moved.mkdir()
        fake.clear()
        fake.failure = httpx.ConnectError("[Errno 8] nodename nor servname provided")
        for start in (chosen, pinned):
            with self.subTest(start=start), hub_cache(moved), no_hub():
                output, _ = self.prepare(start)
                self.assertIn("is already installed", output)
        self.assertEqual(fake.calls(), [])
        self.assertEqual(pins(self.cache), installed)
        self.assertEqual(list(moved.iterdir()), [])

    def test_an_installation_of_a_family_this_release_does_not_name_is_reinstalled(
        self,
    ):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        # A verified assembly whose family this release no longer names is
        # assembled again from the commit it records, without the Hub.
        record = chosen.link / "model.json"
        changed = json.loads(record.read_text()) | {"family": "qwen3-omni"}
        record.unlink()
        record.write_text(json.dumps(changed))
        # It starts without the Hub all the same.
        self.assertTrue(upstream.present(chosen))
        fake.clear()
        with no_hub():
            output, _ = self.prepare(chosen)
        self.assertIn(
            f"Reinstalling {MODEL}: no supported family is named qwen3-omni", output
        )
        self.assertEqual(fake.calls(), [])
        self.assertEqual(assembly.verify(chosen.link)["family"], "qwen3-tts")

    def test_hub_snapshots_are_pinned_and_old_pins_retired(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        refs = sorted((self.cache / FOLDER / "refs/splash").glob("*/*"))
        self.assertEqual([ref.name for ref in refs], [COMMIT])
        # The cache loses the installed snapshot and the installation its
        # assembly: it is installed again from the Hub, at the commit main
        # names now, and the pin of the commit it no longer links goes.
        shutil.rmtree(hub.snapshot(MODEL, COMMIT))
        shutil.rmtree(chosen.link.resolve())
        fake.publish(MODEL, "b" * 40, lambda p: audio_repo(p))
        self.prepare(chosen)
        refs = sorted((self.cache / FOLDER / "refs/splash").glob("*/*"))
        self.assertEqual([ref.name for ref in refs], ["b" * 40])
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["target"]["revision"], "b" * 40
        )

    def test_pins_are_required_before_publishing(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        with (
            mock.patch.object(
                hub, "pin", side_effect=PermissionError(errno.EACCES, "no")
            ),
            self.assertRaises(PermissionError),
        ):
            self.prepare(chosen)
        self.assertFalse(chosen.link.exists())

    def test_pins_change_under_the_lock_and_are_repaired_on_start(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        pin = hub.pin

        def locked(*arguments):
            with (chosen.models_root / ".install.lock").open("a+b") as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return pin(*arguments)

        with mock.patch.object(hub, "pin", side_effect=locked) as pinned:
            self.prepare(chosen)
        self.assertEqual(pinned.call_count, 1)
        self.assertEqual(pins(self.cache), ["a" * 40])
        # A verified start restores lost pins.
        for ref in self.cache.glob("*/refs/splash/*/*"):
            ref.unlink()
        with mock.patch.object(hub, "pin", side_effect=locked):
            self.prepare(chosen)
        self.assertEqual(pins(self.cache), ["a" * 40])

    def test_a_read_only_cache_leaves_the_installation_usable(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        for ref in self.cache.glob("*/refs/splash/*/*"):
            ref.unlink()
        with mock.patch.object(
            hub.os, "replace", side_effect=OSError(errno.EROFS, "read only")
        ):
            output, warnings = self.prepare(chosen)
        self.assertIn("is already installed", output)
        self.assertIn("external cache pruning", warnings)
        self.assertEqual(pins(self.cache), [])

    def test_a_damaged_assembly_is_rebuilt_from_the_cache_without_the_hub(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        built = next((chosen.models_root / ".resolved").iterdir())
        (built / "tokenizer/tokenizer_config.json").unlink()
        fake.failure = httpx.ConnectTimeout("")
        fake.clear()
        with no_hub():
            output, _ = self.prepare(chosen)
        self.assertIn("Reinstalling " + MODEL, output)
        self.assertIn(f"Using {MODEL}@{'a' * 12} from the Hugging Face cache", output)
        self.assertEqual(fake.calls(), [])
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["target"],
            {"repo": MODEL, "revision": "a" * 40},
        )

    def test_damaged_assembly_is_rebuilt(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        built = next((chosen.models_root / ".resolved").iterdir())
        (built / "tokenizer/tokenizer_config.json").unlink()
        fake.downloads.clear()
        output, _ = self.prepare(chosen)
        self.assertIn("Reinstalling " + MODEL, output)
        self.assertIn(f"Rebuilding the damaged {built}", output)
        assembly.verify(built)
        # The cached snapshot is complete; nothing is downloaded again.
        self.assertNotIn(f"{MODEL}/model.safetensors", fake.downloads)

    def test_verify_checks_full_content_and_rejects_same_size_changes(self):
        source = self.root / "source"
        source.write_bytes(b"abcd")
        built = self.root / "assembly"
        built.mkdir()
        (built / "weight").symlink_to(source)
        record = {
            "version": 1,
            "model": MODEL,
            "family": "qwen3-tts",
            "target_format": "mlx-affine",
            "sources": {"target": {"repo": str(self.root), "revision": None}},
            "files": {"weight": assembly.file_record(source)},
        }
        (built / "model.json").write_text(json.dumps(record))
        assembly.verify(built, full=True)
        source.write_bytes(b"abce")
        with self.assertRaises(models.ModelError):
            assembly.verify(built)
        stat = source.stat()
        record["files"]["weight"].update(
            mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns
        )
        (built / "model.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(models.ModelError, "hash mismatch"):
            assembly.verify(built, full=True)

    def test_verify_rejects_a_record_of_another_shape(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        record = json.loads((chosen.link / "model.json").read_text())
        self.assertTrue(assembly._well_formed(record))
        for change in (
            lambda r: r.pop("family"),
            lambda r: r.update(metadata="0" * 64),
            lambda r: r.update(vision_format="none"),
            lambda r: r.update(target_format="gguf"),
            lambda r: r.update(target_format="safetensors"),
            lambda r: r.update(version=True),
            lambda r: r["sources"]["target"].update(revision="main"),
            # A draft, which the language families had.
            lambda r: r["sources"].update(draft={"repo": "x/y", "revision": None}),
            lambda r: r.update(sources={"draft": {"repo": "x/y", "revision": None}}),
            lambda r: r["files"]["config.json"].pop("mtime_ns"),
            lambda r: r["files"]["config.json"].update(digest="z" * 64),
            lambda r: r["files"].update({"../escape": r["files"]["config.json"]}),
            lambda r: r["files"].update({".": r["files"]["config.json"]}),
        ):
            changed = json.loads(json.dumps(record))
            change(changed)
            built = self.root / "copy"
            shutil.rmtree(built, ignore_errors=True)
            shutil.copytree(chosen.link, built, symlinks=True)
            (built / "model.json").unlink()
            (built / "model.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(models.ModelError, "invalid resolved model"):
                assembly.verify(built)

    def test_hub_resolution_pins_one_commit(self):
        fake = FakeHub(self, self.cache)
        fake.publish(MODEL, "a" * 40, lambda p: audio_repo(p), branch="v2")
        repo = hub.Repository.resolve(MODEL, "v2")
        self.assertEqual(fake.requests, [(MODEL, "v2")])
        self.assertEqual(repo.revision, "a" * 40)
        self.assertEqual(
            repo.file("config.json"), fake.snapshot(MODEL, "a" * 40) / "config.json"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                set(repo.download({"model.safetensors"})), {"model.safetensors"}
            )
        fake.publish(
            MODEL, "b" * 40, lambda p: audio_repo(p, tts_config(False)), branch="v2"
        )
        # The branch moved; the resolved repository still reads its commit.
        self.assertIn(
            "quantization_config", json.loads(repo.file("config.json").read_text())
        )
        self.assertEqual(repo.revision, "a" * 40)

    def test_only_an_absolute_path_is_a_local_repository(self):
        # A relative path in the working directory is still a Hub repository ID.
        (self.root / MODEL).mkdir(parents=True)
        fake = FakeHub(self, self.cache)
        fake.publish(MODEL, "a" * 40, lambda p: audio_repo(p), branch="branch")
        with contextlib.chdir(self.root):
            repo = hub.Repository.resolve(MODEL, "branch")
        self.assertEqual(fake.requests, [(MODEL, "branch")])
        self.assertEqual((repo.directory, repo.revision), (None, "a" * 40))
        local = hub.Repository.resolve(str(self.root / MODEL))
        self.assertEqual((local.directory, local.revision), (self.root / MODEL, None))
        with self.assertRaisesRegex(models.ModelError, "local directory not found"):
            hub.Repository.resolve(str(self.root / "deleted"))

    def test_the_cache_serves_the_recorded_pinned_or_named_commit(self):
        fake = FakeHub(self, self.cache)
        fake.failure = http_error(401)
        with self.assertRaisesRegex(
            models.ModelError,
            "cannot download owner/private, which is neither installed nor in "
            "the Hugging Face cache: 401 Client Error.*; set HF_TOKEN ",
        ):
            hub.Repository.resolve("owner/private")
        self.assertIsNone(hub.cached_commit("owner/private"))
        # A branch the cache recorded names its snapshot.
        cached_model(self.cache, "owner/private", "c" * 40, lambda p: audio_repo(p))
        self.assertEqual(hub.cached_commit("owner/private"), "c" * 40)
        self.assertIsNone(hub.cached_commit("owner/private", "v2"))
        self.assertEqual(hub.cached_commit("owner/private", "C" * 40), "c" * 40)
        self.assertIsNone(hub.cached_commit("owner/private", "d" * 40))
        refs = self.cache / "models--owner--private/refs"
        (refs / "main").write_text("d" * 40)
        self.assertIsNone(hub.cached_commit("owner/private"))
        # An installation's record or pins name it, however damaged.
        fake.failure = None
        fake.publish(MODEL, COMMIT, lambda p: audio_repo(p))
        chosen = selection(self.root)
        self.prepare(chosen)
        self.assertFalse((self.cache / FOLDER / "refs/main").exists())
        self.assertEqual(hub.cached_commit(MODEL, None, chosen.link), COMMIT)
        self.assertIsNone(hub.cached_commit(MODEL))
        shutil.rmtree(chosen.link.resolve())
        self.assertEqual(hub.cached_commit(MODEL, None, chosen.link), COMMIT)
        fake.clear()
        with no_hub():
            self.prepare(chosen)
        self.assertEqual(fake.calls(), [])
        assembly.verify(chosen.link)

    def test_a_failed_download_is_reported_without_a_traceback(self):
        fake = fake_hub(self, self.cache)
        fake.download_failure = httpx.ReadTimeout("timed out")
        errors = io.StringIO()
        with (
            contextlib.redirect_stderr(errors),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = models.main(
                ["--models", str(self.root / "fresh"), "--model", MODEL, "prepare"]
            )
        self.assertEqual(code, 1)
        self.assertEqual(
            errors.getvalue(), f"error: cannot install {MODEL}: timed out\n"
        )

    def test_verifying_a_missing_installation_says_so(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            code = models.main(["--models", str(self.root), "--model", MODEL, "verify"])
        self.assertEqual(code, 1)
        self.assertEqual(
            errors.getvalue(), f"error: {MODEL} is not installed in {self.root}\n"
        )

    def test_qwen3_asr_1_7b_installs_with_its_default_draft(self):
        draft = upstream.serve_options.DEFAULT_DRAFT_MODEL
        config = {
            **asr_config(),
            "thinker_config": {"text_config": {"hidden_size": 2048}},
        }
        fake = FakeHub(self, self.cache)
        fake.publish(ASR_MODEL, COMMIT, lambda p: audio_repo(p, config))
        fake.publish(TTS_MODEL, COMMIT, lambda p: audio_repo(p))
        chosen = selection(self.root, ASR_MODEL)
        self.enterContext(
            mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.cache)})
        )
        download = self.enterContext(
            mock.patch.object(upstream.serve_options, "download_model")
        )

        def prepare(model=ASR_MODEL, *options):
            with (
                engine_patch("qwen3-asr" if model == ASR_MODEL else "qwen3-tts"),
                contextlib.redirect_stdout(io.StringIO()) as output,
                contextlib.redirect_stderr(io.StringIO()) as warnings,
            ):
                code = models.main(
                    [
                        "--models",
                        str(chosen.models_root),
                        "--model",
                        model,
                        "prepare",
                        *options,
                    ]
                )
            self.assertEqual(code, 0)
            return output.getvalue(), warnings.getvalue()

        # --no-draft-model (serve's, or another --draft-model) skips it.
        prepare(ASR_MODEL, "--no-draft-model")
        download.assert_not_called()
        # By default the draft downloads beside the model.
        output, _ = prepare()
        download.assert_called_once_with(draft)
        self.assertIn(f"Downloading {draft}, the speculative draft", output)
        # A failed download warns; the model is installed and serves.
        download.reset_mock()
        download.side_effect = OSError("Network is unreachable")
        _, warnings = prepare()
        self.assertIn(f"cannot download the draft {draft}", warnings)
        self.assertIn("Network is unreachable", warnings)
        self.assertIn("without speculative decoding", warnings)
        assembly.verify(chosen.link)
        # Whatever the Hub client raises.
        download.side_effect = RuntimeError("Repository Not Found")
        _, warnings = prepare()
        self.assertIn("Repository Not Found", warnings)
        # A partial draft (an interrupted download) is fetched again.
        download.reset_mock()
        download.side_effect = None
        partial = cached_snapshot(
            self.cache,
            draft,
            "c" * 40,
            lambda p: audio_repo(p, asr_config()),
        )
        (partial / "model.safetensors").unlink()
        self.assertFalse(upstream.draft_present(draft))
        prepare()
        download.assert_called_once_with(draft)
        # A whole cached draft is not fetched again, and speech has none;
        # neither start asks the Hub.
        download.reset_mock()
        (partial / "model.safetensors").write_bytes(b"weights")
        self.assertTrue(upstream.draft_present(draft))
        prepare(TTS_MODEL)
        fake.clear()
        with no_hub():
            prepare()
            prepare(TTS_MODEL)
        download.assert_not_called()
        self.assertEqual(fake.calls(), [])

    def test_quick_and_full_verification_through_the_command_line(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        for full in (False, True):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = models.main(
                    [
                        "--models",
                        str(chosen.models_root),
                        "--model",
                        MODEL,
                        "verify",
                        *(["--full"] if full else []),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn(
                "preflight passed (" + ("full" if full else "quick"), output.getvalue()
            )


if __name__ == "__main__":
    unittest.main()
