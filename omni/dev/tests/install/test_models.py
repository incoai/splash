import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import TTS_MODEL, audio_repo
from install import hub, upstream
from install import models as installer
from server import serve_options


class ModelArtifactTest(unittest.TestCase):
    MODEL_ID = "community/My-Splash.Model_1"
    REVISION = "a" * 40

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # Never use real user credentials or contact the network in these tests.
        for patch in (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("huggingface_hub.get_token", return_value=None),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def snapshot(self, model_id, revision):
        """A snapshot of model_id at revision in a Hub cache of its own, as a
        download leaves it."""
        cache = self.root / "cache" / revision
        snapshot = cache / hub.folder_name(model_id) / "snapshots" / revision
        audio_repo(snapshot)
        return snapshot

    def test_accepts_full_hub_ids_without_name_or_owner_allowlist(self):
        for model_id in (
            self.MODEL_ID,
            "a/b",
            "other-team/finetune",
            "incoai/anything",
            TTS_MODEL,
        ):
            with self.subTest(model=model_id):
                self.assertEqual(installer.validate_repo_id(model_id), model_id)
                self.assertEqual(serve_options.parse_model_id(model_id), model_id)
                self.assertEqual(
                    installer.selection_link(self.root, model_id), self.root / model_id
                )

    def test_missing_or_invalid_model_fails_before_creating_or_downloading(self):
        destination = self.root / "uncreated"
        base = ["--models", str(destination)]
        cases = [
            None,
            "short-name",
            "../outside",
            "owner/repo/extra",
            "owner/../repo",
            "https://huggingface.co/owner/repo",
            "owner/repo.git",
            "owner/repo--one",
            "owner/repo..one",
            "owner/-repo",
            "owner/repo.",
            "owner/" + "x" * 97,
            "owner/repo\n",
            "owner\\repo",
            " owner/repo",
            "",
            # A GGUF variant, which the language engine took; no audio model
            # has one.
            "owner/repo:UD-Q4_K_M",
            "owner/repo:",
        ]
        with mock.patch.object(upstream, "prepare") as prepare:
            for model in cases:
                args = (
                    [*base, "prepare"]
                    if model is None
                    else [*base, "--model", model, "prepare"]
                )
                with (
                    self.subTest(model=model),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        installer.main(args)
                    self.assertEqual(raised.exception.code, 2)
            prepare.assert_not_called()
        self.assertFalse(destination.exists())
        with self.assertRaises(installer.ModelError):
            installer.validate_repo_id(None)
        for bad in ("short-name", "owner/repo:UD-Q4_K_M", "owner/repo:"):
            with self.assertRaises(argparse.ArgumentTypeError):
                serve_options.parse_model_id(bad)
            with self.assertRaises(installer.ModelError):
                installer.selection_link(self.root, bad)

    def test_a_selection_validates_its_model_before_creating_paths(self):
        models = self.root / "uncreated"
        for bad in ("../outside", "owner/repo:Q4_K_M"):
            with self.subTest(model=bad), self.assertRaises(installer.ModelError):
                installer.Selection.of(models, bad)
        self.assertFalse(models.exists())

    def test_command_line_takes_required_model_before_command(self):
        args = installer.parse_args(["--model", self.MODEL_ID, "prepare"])
        self.assertEqual((args.command, args.model), ("prepare", self.MODEL_ID))
        self.assertIsNone(args.revision)

    def test_link_prints_the_selection_link_of_the_source_options(self):
        # make's MODEL_ROOT is this output.
        models = self.root / "models"
        for arguments, options in (
            (["--model", TTS_MODEL], {}),
            (
                ["--model", TTS_MODEL, "--revision", "b" * 40],
                {"revision": "b" * 40},
            ),
        ):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(
                    installer.main(["--models", str(models), *arguments, "link"]), 0
                )
            link = installer.selection_link(models.resolve(), arguments[1], **options)
            self.assertEqual(output.getvalue(), f"{link}\n")
        self.assertEqual(
            installer.selection_link(models, TTS_MODEL), models / TTS_MODEL
        )
        self.assertEqual(
            installer.selection_link(models, TTS_MODEL, revision="v2").parent,
            models / ".selections",
        )

    def test_selection_paths_do_not_conflict(self):
        root = Path("/models")
        paths = {
            installer.selection_link(root, TTS_MODEL),
            installer.selection_link(root, TTS_MODEL, revision="old"),
            installer.selection_link(root, TTS_MODEL, revision="new"),
        }
        self.assertEqual(len(paths), 3)
        # Each is where garbage collection finds the selection links.
        links = {self.root / path.relative_to("/") for path in paths}
        for link in links:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(self.root)
        self.assertEqual(set(installer.selection_links(self.root / "models")), links)

    def test_publish_refuses_to_replace_a_real_directory(self):
        snapshot = self.snapshot(self.MODEL_ID, self.REVISION)
        destination = self.root / "occupied"
        destination.mkdir()
        with self.assertRaisesRegex(installer.ModelError, "non-symlink"):
            installer.link_selection(destination, snapshot)
        # A link is replaced atomically, and nothing of the staging stays.
        link = self.root / "models" / self.MODEL_ID
        installer.link_selection(link, snapshot)
        other = self.snapshot(self.MODEL_ID, "b" * 40)
        installer.link_selection(link, other)
        self.assertEqual(link.resolve(), other.resolve())
        self.assertEqual([p.name for p in link.parent.iterdir()], [link.name])

    def test_installation_kind_is_an_assembly_or_nothing(self):
        link = self.root / "models" / self.MODEL_ID
        self.assertIsNone(installer.installation_kind(link))
        link.mkdir(parents=True)
        (link / "manifest.json").write_text("{}")
        # A legacy Splash package, which this release does not serve.
        self.assertIsNone(installer.installation_kind(link))
        (link / "model.json").write_text("{}")
        self.assertEqual(installer.installation_kind(link), installer.ASSEMBLY)

    def test_shared_hub_cache_keeps_each_installation_revision_pinned(self):
        from huggingface_hub import scan_cache_dir

        first = self.snapshot(self.MODEL_ID, self.REVISION)
        second = self.snapshot(self.MODEL_ID, "b" * 40)
        second = second.rename(first.parent / second.name)
        cache = first.parent.parent.parent
        refs = first.parent.parent / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text("b" * 40)
        install_a = self.root / "install-a" / self.MODEL_ID
        install_b = self.root / "install-b" / self.MODEL_ID
        pin_a = hub.pin(first, self.MODEL_ID, install_a)
        pin_b = hub.pin(second, self.MODEL_ID, install_b)
        self.assertNotEqual(pin_a.parent, pin_b.parent)
        self.assertEqual((refs / "main").read_text(), "b" * 40)
        scanned = scan_cache_dir(cache)
        self.assertFalse(scanned.warnings)
        revisions = next(iter(scanned.repos)).revisions
        self.assertEqual(len(revisions), 2)
        self.assertTrue(all(revision.refs for revision in revisions))

    def test_installation_lock_is_exclusive(self):
        with mock.patch.object(installer.fcntl, "flock") as flock:
            with installer.installation_lock(self.root):
                pass
        self.assertEqual(
            [call.args[1] for call in flock.call_args_list],
            [
                installer.fcntl.LOCK_EX | installer.fcntl.LOCK_NB,
                installer.fcntl.LOCK_UN,
            ],
        )


class ScriptEntryTests(unittest.TestCase):
    """The launcher runs install/models.py as a script: errors raised in the
    other installer modules must reach the user as one line, as they do
    through package imports."""

    def run_script(self, *arguments):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(installer.__file__).resolve()),
                    "--models",
                    str(root / "models"),
                    *arguments,
                ],
                env={
                    "HOME": str(root / "home"),
                    "HF_HUB_OFFLINE": "1",
                    "HF_HUB_CACHE": str(root / "hub"),
                },
                capture_output=True,
                text=True,
                timeout=60,
            )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertNotIn("Traceback", output)
        return output

    def test_an_uncached_model_offline_is_one_error_line(self):
        output = self.run_script("--model", "someone/not-cached", "prepare")
        self.assertIn(
            "error: someone/not-cached is neither installed nor in the Hugging "
            "Face cache, and HF_HUB_OFFLINE is set (splash-omni serve --offline); "
            "start without it to download the model\n",
            output,
        )

    def test_verifying_nothing_installed_is_one_error_line(self):
        output = self.run_script("--model", "someone/model", "verify")
        self.assertIn("error: someone/model is not installed in ", output)

    def test_install_entry_points_run_as_scripts(self):
        # As the launcher and make run them: by path, from any working
        # directory.
        install = Path(installer.__file__).resolve().parent
        for script in ("models.py", "launcher.py"):
            with (
                self.subTest(script=script),
                tempfile.TemporaryDirectory() as temporary,
            ):
                result = subprocess.run(
                    [sys.executable, str(install / script), "--help"],
                    cwd=temporary,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
