import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import (
    DENSE,
    DRAFT_COMMIT,
    GGML,
    MOE,
    cached_snapshot,
    draft_dir,
    fixture,
    loadable_tensors,
    mlx_target,
    vision_fixture,
    write_gguf,
)
from dev.tests.server_fixtures import keep_stop_signals, server_arguments
from install import assembly, hub, launcher, local
from server import server as api


class LocalServeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop("SPLASH_PORT", None)
        os.environ["HF_HUB_CACHE"] = str(self.root / "cache")
        keep_stop_signals(self)
        self.enterContext(mock.patch.object(launcher, "RUNTIME_DIR", self.root / "run"))
        self.enterContext(
            mock.patch.object(launcher.paths, "MODELS", self.root / "models")
        )
        self.enterContext(mock.patch.object(launcher, "_check_port"))
        self.enterContext(mock.patch.object(launcher, "_version", return_value="test"))

    def gguf(self):
        headers = self.root / "headers"
        headers.mkdir(exist_ok=True)
        values = fixture(native=True)
        return write_gguf(
            self.root / "my model.gguf",
            values,
            loadable_tensors(values, headers).items(),
        )

    def serve(self, target, *options):
        with (
            # Hosted macOS runners expose a virtual GPU. Runtime/device
            # checks have their own tests; keep local preparation real.
            mock.patch.object(launcher, "_ensure_runtime") as setup,
            mock.patch.object(
                launcher,
                "_ensure_installed",
                side_effect=AssertionError("Hub installation"),
            ),
            mock.patch.object(launcher.os, "execve") as execute,
        ):
            result = launcher.main(
                [
                    "serve",
                    "--model",
                    "local/my-model",
                    "--model-dir",
                    str(target),
                    *options,
                ]
            )
        setup.assert_called_once_with()
        return result, execute

    def test_local_gguf_is_prepared_by_splash_python_and_served_without_the_hub(self):
        target = self.gguf()
        draft = draft_dir(self.root / "my draft", MOE)
        result, execute = self.serve(
            target, "--draft-model", str(draft), "--max-context", "8K"
        )
        self.assertIsNone(result)
        execute.assert_called_once()
        binary, command, environment = execute.call_args.args
        self.assertEqual(binary, str(launcher.paths.PYTHON))
        with mock.patch.dict(os.environ, environment):
            served = api.parse_args(server_arguments(command))
        self.assertEqual(served.model, "local/my-model")
        self.assertEqual(served.max_context, 8192)
        prepared = Path(server_arguments(command)[0])
        record = assembly.verify(prepared, full=True)
        self.assertEqual(record["sources"]["target"]["repo"], str(target))
        self.assertEqual(record["sources"]["draft"]["repo"], str(draft))
        self.assertEqual(record["vision_format"], "none")

    def test_local_gguf_filename_symlink_and_projector_are_preserved(self):
        blob = self.gguf().rename(self.root / "blob-digest")
        target = self.root / "snapshot" / "named-model.gguf"
        target.parent.mkdir()
        target.symlink_to(blob)
        projector = write_gguf(
            self.root / "my projector.gguf",
            vision_fixture(),
            [("v.patch_embd.weight", GGML["F32"])],
        )
        draft = draft_dir(self.root / "draft", MOE)
        with mock.patch("os.getcwd", return_value=str(self.root)):
            result, execute = self.serve(
                "snapshot/named-model.gguf",
                "--draft-model",
                str(draft),
                "--mmproj",
                "my projector.gguf",
            )
        self.assertIsNone(result)
        prepared = Path(server_arguments(execute.call_args.args[1])[0])
        record = assembly.verify(prepared, full=True)
        self.assertEqual(record["sources"]["target"]["repo"], str(target))
        self.assertEqual(record["vision_format"], "gguf")
        self.assertEqual((prepared / assembly.GGUF_VISION).readlink(), projector)

    def test_local_mlx_directory_and_language_only_are_forwarded(self):
        target = mlx_target(self.root / "MLX model", DENSE)
        draft = draft_dir(self.root / "draft", DENSE)
        result, execute = self.serve(
            target, "--draft-model", str(draft), "--language-only"
        )
        self.assertIsNone(result)
        record = assembly.verify(
            Path(server_arguments(execute.call_args.args[1])[0]), full=True
        )
        self.assertEqual(record["target_format"], "mlx-affine")
        self.assertEqual(record["vision_format"], "none")

    def test_local_gguf_directory_uses_the_cached_default_draft(self):
        target = self.gguf()
        cache = self.root / "cache"
        cached_snapshot(
            cache, MOE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE)
        )
        refs = cache / hub.folder_name(MOE.draft_repo) / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(DRAFT_COMMIT)
        result, execute = self.serve(target.parent)
        self.assertIsNone(result)
        record = assembly.verify(
            Path(server_arguments(execute.call_args.args[1])[0]), full=True
        )
        self.assertEqual(
            record["sources"]["draft"],
            {"repo": MOE.draft_repo, "revision": DRAFT_COMMIT},
        )

    def test_conflicting_local_options_are_refused_before_setup(self):
        for options, reason in (
            (["--mmproj", "tower.gguf"], "--mmproj requires --model-dir"),
            (
                ["--model-dir", "model", "--revision", "main"],
                "--revision cannot be used with --model-dir",
            ),
            (
                ["--model-dir", "model", "--draft-model", "owner/draft"],
                "--draft-model must be a local DFlash2 draft directory",
            ),
            (
                ["--model-dir", "model", "--mmproj", "tower.gguf", "--language-only"],
                "--mmproj cannot be used with --language-only",
            ),
        ):
            with (
                self.subTest(options=options),
                mock.patch.object(launcher, "_ensure_runtime") as setup,
                mock.patch("sys.stderr", io.StringIO()) as error,
                self.assertRaises(SystemExit) as refused,
            ):
                launcher.main(["serve", "--model", "local/my-model", *options])
            self.assertEqual(refused.exception.code, 2)
            self.assertIn(reason, error.getvalue())
            setup.assert_not_called()

    def test_failed_local_preparation_does_not_start_the_server(self):
        result, execute = self.serve(self.root / "missing.gguf")
        self.assertEqual(result, 1)
        execute.assert_not_called()


class LocalPreparationCommandTests(unittest.TestCase):
    def setUp(self):
        keep_stop_signals(self)

    def test_helper_prints_only_the_public_interfaces_result(self):
        def prepare(*args, **options):
            print("Preparing the model...")
            return Path("/prepared/model with spaces")

        with (
            mock.patch.object(
                local, "prepare_local_model", side_effect=prepare
            ) as prepare_model,
            mock.patch("sys.stdout", io.StringIO()) as output,
            mock.patch("sys.stderr", io.StringIO()) as progress,
        ):
            result = local._main(
                [
                    "/target",
                    "--draft-model",
                    "/draft",
                    "--mmproj",
                    "/projector",
                    "--models",
                    "/models",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "/prepared/model with spaces\n")
        self.assertEqual(progress.getvalue(), "Preparing the model...\n")
        prepare_model.assert_called_once_with(
            "/target",
            "/draft",
            mmproj="/projector",
            language_only=False,
            models_dir=Path("/models"),
        )

    def test_helper_reports_refusals_and_interruptions_without_a_traceback(self):
        for failure, status in (
            (local.ModelError("bad model"), 1),
            (OSError("unreadable"), 1),
            (KeyboardInterrupt(), 130),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                mock.patch.object(local, "prepare_local_model", side_effect=failure),
                mock.patch("sys.stdout", io.StringIO()) as output,
                mock.patch("sys.stderr", io.StringIO()) as error,
            ):
                self.assertEqual(local._main(["/target"]), status)
            self.assertEqual(output.getvalue(), "")
            self.assertNotIn("Traceback", error.getvalue())
            self.assertEqual(
                error.getvalue(), "" if status == 130 else f"error: {failure}\n"
            )


if __name__ == "__main__":
    unittest.main()
