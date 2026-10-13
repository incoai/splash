import argparse
import contextlib
import errno
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dev.tests.installer_fixtures import http_error
from install import hub
from install import models as installer
from server import serve_options


class InstallerTest(unittest.TestCase):
    """Selections, their links and the Hub cache pins that keep an
    installation's snapshots."""

    MODEL_ID = "community/My-Splash.Model_1"
    REVISION = "a" * 40

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture_number = 0
        # Never use real user credentials or contact the network in these tests.
        for patch in (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("huggingface_hub.get_token", return_value=None),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.api = self.start_patch("huggingface_hub.HfApi")
        self.manifest_download = self.start_patch("huggingface_hub.hf_hub_download")
        self.download = self.start_patch("huggingface_hub.snapshot_download")

    def start_patch(self, name):
        patch = mock.patch(name)
        value = patch.start()
        self.addCleanup(patch.stop)
        return value

    def snapshot_fixture(self, *, revision=None, beside=None):
        """A snapshot of MODEL_ID at revision, in the repository of the
        snapshot beside or in a Hub cache of its own, its one file linked to
        its blob as the Hub cache links them."""
        self.fixture_number += 1
        repository = (
            beside.parent.parent
            if beside
            else self.root
            / f"cache-{self.fixture_number}"
            / ("models--" + self.MODEL_ID.replace("/", "--"))
        )
        snapshot = repository / "snapshots" / (revision or self.REVISION)
        snapshot.mkdir(parents=True)
        blob = repository / "blobs" / f"config-{self.fixture_number}"
        blob.parent.mkdir(exist_ok=True)
        blob.write_text("{}\n")
        (snapshot / "config.json").symlink_to(os.path.relpath(blob, snapshot))
        return snapshot

    def test_accepts_full_hub_ids_without_name_or_owner_allowlist(self):
        for model_id in (
            self.MODEL_ID,
            "a/b",
            "other-team/finetune",
            "incoai/anything",
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
        ]
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
        self.api.assert_not_called()
        self.download.assert_not_called()
        self.assertFalse(destination.exists())
        with self.assertRaises(installer.ModelError):
            installer.validate_repo_id(None)
        with self.assertRaises(argparse.ArgumentTypeError):
            serve_options.parse_model_id("short-name")

    def test_a_selection_validates_its_model_before_creating_paths(self):
        models = self.root / "uncreated"
        with self.assertRaises(installer.ModelError):
            installer.Selection.of(models, "../outside")
        self.assertFalse(models.exists())

    def test_command_line_takes_required_model_before_command(self):
        args = installer.parse_args(["--model", self.MODEL_ID, "prepare"])
        self.assertEqual((args.command, args.model), ("prepare", self.MODEL_ID))

    def test_variant_model_ids_parse_and_name_selection_links(self):
        self.assertEqual(
            installer.split_model_id("owner/repo:UD-Q4_K_M"),
            ("owner/repo", "UD-Q4_K_M"),
        )
        self.assertEqual(installer.split_model_id("owner/repo"), ("owner/repo", None))
        self.assertEqual(
            serve_options.parse_model_id("owner/repo:UD-Q4_K_M"),
            "owner/repo:UD-Q4_K_M",
        )
        for bad in ("owner/repo:", "owner/repo:a b", "owner/repo:..", "owner:v"):
            with self.assertRaises(installer.ModelError):
                installer.split_model_id(bad)
        with self.assertRaises(argparse.ArgumentTypeError):
            serve_options.parse_model_id("owner/repo:")
        models = self.root / "models"
        self.assertEqual(
            installer.selection_link(models, "owner/repo:UD-Q4_K_M"),
            models / "owner" / "repo:UD-Q4_K_M",
        )
        self.assertEqual(
            installer.selection_link(models, "owner/repo"), models / "owner/repo"
        )

    def test_link_prints_the_selection_link_of_the_source_options(self):
        # make's MODEL_ROOT is this output; a relative draft folder names the
        # installation splash serve --draft-model selects from the same folder.
        models = self.root / "models"
        draft = self.root / "draft"
        draft.mkdir()
        for arguments, options in (
            (["--model", "owner/repo:UD-Q4_K_M"], {}),
            (
                ["--model", "owner/repo", "--revision", "b" * 40, "--language-only"]
                + ["--draft-model", os.path.relpath(draft)],
                {
                    "revision": "b" * 40,
                    "language_only": True,
                    "draft_model": str(draft.resolve()),
                },
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

    def test_a_splash_package_is_refused_with_its_mlx_model(self):
        models = self.root / "models"
        for name, replacement in installer.PACKAGE_REPLACEMENTS.items():
            with self.subTest(format=name):
                # A link to a folder outside any Hub cache.
                package = self.root / "packages" / name
                package.mkdir(parents=True)
                (package / "manifest.json").write_text(
                    json.dumps({"format": {"name": name}})
                )
                link = installer.selection_link(models, self.MODEL_ID)
                link.parent.mkdir(parents=True, exist_ok=True)
                link.unlink(missing_ok=True)
                link.symlink_to(package, target_is_directory=True)
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    result = installer.main(
                        ["--models", str(models), "--model", self.MODEL_ID, "verify"]
                    )
                self.assertEqual(result, 1)
                self.assertIn(
                    f"{self.MODEL_ID} is a Splash package, which Splash no longer "
                    f"loads; serve the MLX model of its family instead: splash "
                    f"serve --model {replacement}",
                    errors.getvalue(),
                )
                # Only a Hub cache folder is named for deletion.
                self.assertNotIn("can be deleted", errors.getvalue())
        # Another tool's manifest.json names no package format.
        installer.refuse_package(self.MODEL_ID, {"format": "other"})
        installer.refuse_package(self.MODEL_ID, [])
        self.download.assert_not_called()

    def test_publish_refuses_to_replace_a_real_directory(self):
        snapshot = self.snapshot_fixture()
        destination = self.root / "occupied"
        destination.mkdir()
        with self.assertRaisesRegex(installer.ModelError, "non-symlink"):
            installer.link_selection(destination, snapshot)

    def test_shared_hub_cache_keeps_each_installation_revision_pinned(self):
        from huggingface_hub import scan_cache_dir

        first = self.snapshot_fixture()
        second = self.snapshot_fixture(revision="b" * 40, beside=first)
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

    def test_an_existing_pin_needs_no_cache_write(self):
        snapshot = self.snapshot_fixture()
        destination = self.root / "models" / self.MODEL_ID
        ref = hub.pin(snapshot, self.MODEL_ID, destination)
        with mock.patch.object(
            hub.os, "replace", side_effect=AssertionError("cache write")
        ):
            self.assertEqual(hub.pin(snapshot, self.MODEL_ID, destination), ref)
        self.assertEqual(ref.read_text(), self.REVISION)

    def test_pins_are_written_where_hard_links_are_unsupported(self):
        snapshot = self.snapshot_fixture()
        destination = self.root / "models" / self.MODEL_ID
        with mock.patch.object(
            hub.os,
            "link",
            side_effect=OSError(errno.ENOTSUP, "Operation not supported"),
        ):
            hub.pin(snapshot, self.MODEL_ID, destination)
        refs = list((snapshot.parent.parent / "refs/splash").glob("*/*"))
        self.assertEqual([ref.read_text() for ref in refs], [self.REVISION])

    def test_an_invalid_existing_pin_is_not_ignored(self):
        snapshot = self.snapshot_fixture()
        destination = self.root / "models" / self.MODEL_ID
        hub.pin(snapshot, self.MODEL_ID, destination).write_text("wrong")
        with self.assertRaisesRegex(
            installer.ModelError, "invalid installed snapshot reference"
        ):
            hub.pin(snapshot, self.MODEL_ID, destination)

    def test_a_read_only_cache_keeps_a_verified_installation_usable(self):
        snapshot = self.snapshot_fixture()
        destination = self.root / "models" / self.MODEL_ID
        for code in (errno.EACCES, errno.EPERM, errno.EROFS):
            with self.subTest(errno=code):
                errors = io.StringIO()
                with (
                    mock.patch.object(
                        hub.os, "replace", side_effect=OSError(code, "read only")
                    ),
                    contextlib.redirect_stderr(errors),
                ):
                    hub.repair_pins(destination, [(snapshot, self.MODEL_ID)])
                self.assertIn("external cache pruning", errors.getvalue())

    def test_a_failed_retirement_keeps_the_current_pin(self):
        snapshot = self.snapshot_fixture()
        destination = self.root / "models" / self.MODEL_ID
        ref = hub.pin(snapshot, self.MODEL_ID, destination)
        old = ref.parent / ("b" * 40)
        old.write_text("b" * 40)
        errors = io.StringIO()
        with (
            mock.patch.object(
                Path, "unlink", side_effect=PermissionError(errno.EACCES, "read only")
            ),
            contextlib.redirect_stderr(errors),
        ):
            hub.repair_pins(destination, [(snapshot, self.MODEL_ID)])
        self.assertEqual(ref.read_text(), self.REVISION)
        self.assertTrue(old.exists())
        self.assertIn("could not retire", errors.getvalue())

    def test_repairing_one_installation_retires_only_its_old_pins(self):
        first = self.snapshot_fixture()
        second = self.snapshot_fixture(revision="b" * 40, beside=first)
        destination = self.root / "install-a" / self.MODEL_ID
        other = self.root / "install-b" / self.MODEL_ID
        old_pin = hub.pin(first, self.MODEL_ID, destination)
        other_pin = hub.pin(first, self.MODEL_ID, other)
        hub.repair_pins(destination, [(second, self.MODEL_ID)])
        self.assertFalse(old_pin.exists())
        self.assertTrue(other_pin.exists())
        self.assertEqual([p.read_text() for p in old_pin.parent.iterdir()], ["b" * 40])

    def test_the_hub_must_resolve_a_commit(self):
        self.api.return_value.model_info.return_value = SimpleNamespace(
            sha="main", siblings=[]
        )
        with self.assertRaisesRegex(installer.ModelError, "did not resolve"):
            hub.Repository.resolve(self.MODEL_ID)

    def test_hub_errors_redact_the_token_and_say_how_to_authenticate(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_testdistributiontoken"}):
            message = hub.reason(
                RuntimeError("network unavailable hf_testdistributiontoken")
            )
        self.assertEqual(message, "network unavailable [redacted]")
        for status in (401, 403, 404, 500):
            with self.subTest(status=status):
                self.assertEqual(
                    "hf auth login" in hub.reason(http_error(status)),
                    status in (401, 403),
                )

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
        self.assertIn("error: cannot resolve someone/not-cached", output)

    def test_verifying_nothing_installed_is_one_error_line(self):
        output = self.run_script("--model", "someone/model:Q4_K_M", "verify")
        self.assertIn("error: someone/model:Q4_K_M is not installed in ", output)

    def test_installer_runs_as_a_script(self):
        # As the launcher and make run it: by path, from any working
        # directory.
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, str(Path(installer.__file__).resolve()), "--help"],
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
