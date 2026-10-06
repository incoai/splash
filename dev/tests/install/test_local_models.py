import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.install.test_gguf_metadata import fixture, loadable_tensors, write_gguf
from dev.tests.installer_fixtures import (
    DENSE,
    DRAFT_COMMIT,
    MODEL,
    MOE,
    FakeHub,
    cached_snapshot,
    draft_dir,
    mlx_target,
)
from install import assembly, models, upstream


class LocalModelsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.library = self.root / "local models"
        self.cache = self.root / "hub"
        self.fake = FakeHub(self, self.cache)

    def selection(self, **options):
        return models.Selection.of(
            self.root / "models",
            MODEL,
            language_only=True,
            **({"model_dir": self.library} | options),
        )

    def prepare(self, chosen):
        with contextlib.redirect_stdout(io.StringIO()):
            upstream.prepare(chosen)
        return assembly.verify(chosen.link, full=True)

    def test_existing_local_model_and_draft_are_reused_without_network(self):
        target = mlx_target(self.library / MODEL, DENSE)
        draft = draft_dir(self.library / DENSE.draft_repo, DENSE)
        chosen = self.selection()
        record = self.prepare(chosen)
        self.assertEqual(record["sources"]["target"]["repo"], str(target))
        self.assertEqual(
            (chosen.link / "target/model.safetensors").resolve(),
            target / "model.safetensors",
        )
        self.assertEqual(
            (chosen.link / "draft/model.safetensors").resolve(),
            draft / "model.safetensors",
        )
        self.prepare(chosen)
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.fake.downloads, [])

    def test_missing_draft_explains_download_without_fetching_weights(self):
        mlx_target(self.library / MODEL, DENSE)
        with self.assertRaisesRegex(
            models.ModelError, "Missing draft.*download-draft"
        ) as error:
            self.prepare(self.selection())
        self.assertIn(DENSE.draft_repo, str(error.exception))
        self.assertIn("--model-dir", str(error.exception))
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.fake.downloads, [])

    def test_draft_cached_without_a_branch_ref_is_reused_offline(self):
        mlx_target(self.library / MODEL, DENSE)
        cached = cached_snapshot(
            self.cache, DENSE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, DENSE)
        )
        chosen = self.selection()
        self.prepare(chosen)
        self.assertEqual(
            (chosen.link / "draft/model.safetensors").resolve(),
            cached / "model.safetensors",
        )
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.fake.downloads, [])

    def test_missing_target_downloads_into_library_after_draft_preflight(self):
        self.fake.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE))
        chosen = self.selection()
        with self.assertRaisesRegex(models.ModelError, "Missing draft"):
            self.prepare(chosen)
        self.assertFalse((self.library / MODEL / "model.safetensors").exists())
        self.assertNotIn(MODEL + "/model.safetensors", self.fake.downloads)
        draft_dir(self.library / DENSE.draft_repo, DENSE)
        self.prepare(chosen)
        weights = self.library / MODEL / "model.safetensors"
        self.assertTrue(weights.is_file())
        self.assertFalse(weights.is_symlink())
        self.assertEqual((chosen.link / "target/model.safetensors").resolve(), weights)
        self.assertFalse(
            (self.fake.snapshot(MODEL, "a" * 40) / "model.safetensors").exists()
        )

    def test_explicit_draft_download_goes_to_library_even_when_default_cached(self):
        mlx_target(self.library / MODEL, DENSE)
        cached_snapshot(
            self.cache, DENSE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, DENSE)
        )
        self.fake.publish(DENSE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, DENSE))
        chosen = self.selection(download_draft=True)
        self.prepare(chosen)
        draft = self.library / DENSE.draft_repo / "model.safetensors"
        self.assertTrue(draft.is_file())
        self.assertFalse(draft.is_symlink())
        self.assertEqual((chosen.link / "draft/model.safetensors").resolve(), draft)

    def test_standalone_download_fetches_no_target_weights_or_assembly(self):
        self.fake.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE))
        self.fake.publish(DENSE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, DENSE))
        for root in (None, self.library):
            with self.subTest(root=root), contextlib.redirect_stdout(io.StringIO()):
                args = ["--models", str(self.root / "models"), "--model", MODEL]
                if root:
                    args += ["--model-dir", str(root)]
                self.assertEqual(models.main([*args, "download-draft"]), 0)
            self.assertNotIn(MODEL + "/model.safetensors", self.fake.downloads)
        self.assertTrue(
            (self.library / DENSE.draft_repo / "model.safetensors").is_file()
        )
        self.assertFalse((self.root / "models/.resolved").exists())
        self.fake.requests.clear()
        self.fake.downloads.clear()
        mlx_target(self.library / MODEL, DENSE)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(models.main([*args, "download-draft"]), 0)
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.fake.downloads, [])

    def test_local_directory_identity_is_normalized_and_download_permission_is_not_identity(
        self,
    ):
        with mock.patch("pathlib.Path.home", return_value=self.root):
            # expanduser reads HOME through os.path, as an actual shell path does.
            with mock.patch.dict("os.environ", {"HOME": str(self.root)}):
                chosen = self.selection(model_dir="~/local models")
        self.assertEqual(chosen, self.selection())
        self.assertEqual(chosen.link, self.selection(download_draft=True).link)
        self.assertNotEqual(
            chosen.link, self.selection(model_dir=self.root / "elsewhere").link
        )
        self.assertNotEqual(chosen.link, self.selection(model_dir=None).link)

    def test_explicit_revision_refuses_unmanaged_local_weights(self):
        mlx_target(self.library / MODEL, DENSE)
        draft_dir(self.library / DENSE.draft_repo, DENSE)
        with self.assertRaisesRegex(models.ModelError, "revision.*unmanaged"):
            self.prepare(self.selection(revision="a" * 40))
        self.assertEqual(self.fake.requests, [])

    def test_cached_draft_from_existing_installation_survives_cache_location_change(
        self,
    ):
        self.fake.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE))
        self.fake.publish(DENSE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, DENSE))
        self.prepare(self.selection(model_dir=None))
        mlx_target(self.library / MODEL, DENSE)
        self.fake.requests.clear()
        self.fake.downloads.clear()
        with mock.patch(
            "huggingface_hub.constants.HF_HUB_CACHE", str(self.root / "empty")
        ):
            self.prepare(self.selection())
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.fake.downloads, [])

    def gguf_target(self, root, variant="Q4_K_M"):
        root.mkdir(parents=True, exist_ok=True)
        values = fixture(native=True)
        write_gguf(
            root / f"model-{variant}.gguf",
            values,
            loadable_tensors(values, self.root).items(),
        )

    def test_local_gguf_reuse_and_missing_variant_download(self):
        repo_id = "unsloth/Test-GGUF"
        self.gguf_target(self.library / repo_id)
        draft_dir(self.library / MOE.draft_repo, MOE)
        chosen = models.Selection.of(
            self.root / "models",
            repo_id + ":Q4_K_M",
            language_only=True,
            model_dir=self.library,
        )
        self.prepare(chosen)
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.fake.downloads, [])
        self.fake.publish(
            repo_id, "a" * 40, lambda p: self.gguf_target(p, "OTHER-Q4_K_M")
        )
        other = models.Selection.of(
            self.root / "models",
            repo_id + ":OTHER-Q4_K_M",
            language_only=True,
            model_dir=self.library,
        )
        self.prepare(other)
        self.assertTrue((self.library / repo_id / "model-OTHER-Q4_K_M.gguf").is_file())
        self.assertTrue((self.library / repo_id / "model-Q4_K_M.gguf").is_file())
        self.assertEqual(self.fake.downloads, [repo_id + "/model-OTHER-Q4_K_M.gguf"])

    def test_explicit_download_can_copy_cached_draft_without_network(self):
        mlx_target(self.library / MODEL, DENSE)
        cached_snapshot(
            self.cache, DENSE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, DENSE)
        )
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            self.prepare(self.selection(download_draft=True))
        self.assertTrue(
            (self.library / DENSE.draft_repo / "model.safetensors").is_file()
        )
        self.assertEqual(self.fake.requests, [])

    def test_incompatible_local_draft_is_not_silently_replaced(self):
        mlx_target(self.library / MODEL, DENSE)
        directory = draft_dir(self.library / DENSE.draft_repo, MOE)
        before = (directory / "config.json").read_bytes()
        with self.assertRaisesRegex(models.ModelError, "incompatible"):
            self.prepare(self.selection(download_draft=True))
        self.assertEqual((directory / "config.json").read_bytes(), before)
        self.assertEqual(self.fake.requests, [])

    def test_local_draft_override_is_used_and_missing_override_reports_download_command(
        self,
    ):
        mlx_target(self.library / MODEL, DENSE)
        directory = draft_dir(self.root / "custom draft", DENSE)
        chosen = self.selection(draft_model=str(directory))
        self.prepare(chosen)
        self.assertEqual(
            (chosen.link / "draft/model.safetensors").resolve(),
            directory / "model.safetensors",
        )
        with self.assertRaisesRegex(models.ModelError, "Missing draft") as error:
            self.prepare(self.selection(draft_model="company/custom-draft"))
        self.assertIn("--draft-model company/custom-draft", str(error.exception))

    def test_local_changes_rebuild_and_cleanup_does_not_delete_external_files(self):
        target = mlx_target(self.library / MODEL, DENSE)
        draft_dir(self.library / DENSE.draft_repo, DENSE)
        chosen = self.selection()
        self.prepare(chosen)
        original = chosen.link.resolve()
        (target / "tokenizer.json").write_text('{"changed": true}')
        self.prepare(chosen)
        self.assertNotEqual(chosen.link.resolve(), original)
        self.assertFalse(original.exists())
        self.assertEqual((target / "tokenizer.json").read_text(), '{"changed": true}')
        (target / "config.json").write_text("invalid")
        with self.assertRaisesRegex(models.ModelError, "could not read"):
            self.prepare(chosen)
        self.assertEqual((target / "config.json").read_text(), "invalid")
        self.assertEqual(self.fake.requests, [])

    def test_missing_local_vision_gives_language_only_hint(self):
        self.gguf_target(self.library / "unsloth/Test-GGUF")
        chosen = models.Selection.of(
            self.root / "models", "unsloth/Test-GGUF:Q4_K_M", model_dir=self.library
        )
        with self.assertRaisesRegex(models.ModelError, "--language-only"):
            self.prepare(chosen)
        self.assertEqual(self.fake.requests, [])

    def test_managed_revision_can_be_reused_but_other_revision_is_refused(self):
        self.fake.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE))
        draft_dir(self.library / DENSE.draft_repo, DENSE)
        self.prepare(self.selection())
        self.fake.requests.clear()
        self.prepare(self.selection(revision="a" * 40))
        self.assertEqual(self.fake.requests, [])
        self.prepare(self.selection(revision="main"))
        with self.assertRaisesRegex(models.ModelError, "different revision"):
            self.prepare(self.selection(revision="b" * 40))

    def test_sharded_target_and_draft_keep_indices_for_later_local_start(self):
        def sharded(root, builder):
            builder(root, DENSE)
            (root / "model.safetensors").rename(root / "part.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "part.safetensors"}})
            )

        self.fake.publish(MODEL, "a" * 40, lambda p: sharded(p, mlx_target))
        self.fake.publish(
            DENSE.draft_repo, DRAFT_COMMIT, lambda p: sharded(p, draft_dir)
        )
        chosen = self.selection(download_draft=True)
        self.prepare(chosen)
        self.fake.requests.clear()
        self.fake.downloads.clear()
        self.prepare(self.selection())
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.fake.downloads, [])
        for repo_id in (MODEL, DENSE.draft_repo):
            self.assertTrue(
                (self.library / repo_id / "model.safetensors.index.json").is_file()
            )

    def test_incomplete_or_incompatible_cached_draft_explains_failure(self):
        mlx_target(self.library / MODEL, DENSE)
        cached_snapshot(
            self.cache, DENSE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE)
        )
        with self.assertRaisesRegex(models.ModelError, "Missing draft.*incompatible"):
            self.prepare(self.selection())

    def test_non_directory_library_fails_before_network(self):
        self.library.write_text("keep me")
        with self.assertRaisesRegex(models.ModelError, "not a directory"):
            self.prepare(self.selection())
        self.assertEqual(self.fake.requests, [])

    def test_interrupted_draft_download_can_be_retried_from_empty_directory(self):
        mlx_target(self.library / MODEL, DENSE)
        self.fake.publish(DENSE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, DENSE))
        self.fake.download_failure = OSError("download interrupted")
        with self.assertRaisesRegex(models.ModelError, "interrupted"):
            self.prepare(self.selection(download_draft=True))
        # A filesystem or process interruption can leave the destination directory.
        (self.library / DENSE.draft_repo).mkdir(parents=True, exist_ok=True)
        self.fake.download_failure = None
        with self.assertRaisesRegex(models.ModelError, "Missing draft"):
            self.prepare(self.selection())
        self.prepare(self.selection(download_draft=True))

    def test_standalone_script_works_offline_with_local_sources(self):
        mlx_target(self.library / MODEL, DENSE)
        draft_dir(self.library / DENSE.draft_repo, DENSE)
        run = subprocess.run(
            [
                sys.executable,
                models.__file__,
                "--model",
                MODEL,
                "--model-dir",
                str(self.library),
                "download-draft",
            ],
            capture_output=True,
            text=True,
            env=dict(os.environ, HF_HUB_OFFLINE="1"),
            timeout=20,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("is ready in", run.stdout)

    def test_missing_target_weights_are_downloaded_without_replacing_metadata(self):
        self.fake.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE))
        target = mlx_target(self.library / MODEL, DENSE)
        (target / "model.safetensors").unlink()
        original = (target / "config.json").stat()
        draft_dir(self.library / DENSE.draft_repo, DENSE)
        self.prepare(self.selection())
        self.assertTrue((target / "model.safetensors").is_file())
        self.assertEqual((target / "config.json").stat().st_ino, original.st_ino)


if __name__ == "__main__":
    unittest.main()
