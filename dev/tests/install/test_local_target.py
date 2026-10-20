"""Local target selection and real assembly regression coverage."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import DENSE, draft_dir, mlx_target
from install import assembly, launcher, models, upstream
from server import serve_options


class LocalTargetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.target = mlx_target(self.root / "target with spaces", DENSE)
        self.draft = draft_dir(self.root / "draft", DENSE)

    def selection(self, **options):
        return models.Selection.of(
            self.root / "models",
            str(self.target),
            language_only=True,
            draft_model=str(self.draft),
            **options,
        )

    def test_local_path_and_hub_ids_parse(self):
        for value in ("owner/repo", "owner/repo:Q4_K_M"):
            self.assertEqual(serve_options.parse_model_id(value), value)
        with self.assertRaises(argparse.ArgumentTypeError):
            serve_options.parse_model_directory(str(self.root / "missing"))
        with self.assertRaises(argparse.ArgumentTypeError):
            serve_options.parse_model_directory(str(self.target / "config.json"))

    def test_launcher_and_installer_accept_local_target(self):
        self.assertEqual(
            launcher.parse_args(["serve", "--modeldir", str(self.target)]).model,
            str(self.target),
        )
        self.assertEqual(
            models.parse_args(["--modeldir", str(self.target), "prepare"]).model,
            str(self.target),
        )

    def test_options_are_exclusive_and_model_remains_hub_only(self):
        for parse, prefix, suffix in (
            (launcher.parse_args, ["serve"], []),
            (models.parse_args, [], ["prepare"]),
        ):
            for flags in (
                ["--model", str(self.target)],
                ["--model", "owner/repo", "--modeldir", str(self.target)],
                ["--modeldir", str(self.target), "--revision", "main"],
            ):
                with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    parse(prefix + flags + suffix)
        self.assertEqual(
            serve_options.parse_model_directory(str(self.target)), str(self.target)
        )

    def test_launcher_passes_modeldir_to_installer(self):
        chosen = self.selection()
        with (
            mock.patch.object(launcher.paths, "PACKAGED", True),
            mock.patch.object(models, "run_engine"),
            mock.patch.object(launcher, "_run_held", return_value=0) as run,
        ):
            launcher._ensure_installed(chosen)
        command = run.call_args.args[0]
        self.assertIn("--modeldir", command)
        self.assertNotIn("--model", command)
        self.assertEqual(models.parse_args(command[2:]).model, str(self.target))

    def test_selection_is_contained_and_revision_refused(self):
        chosen = models.Selection.of(self.root / "models", str(self.target))
        self.assertEqual(chosen.link.parent, chosen.models_root / ".selections")
        with self.assertRaisesRegex(models.ModelError, "--revision"):
            self.selection(revision="main")
        alias = self.root / "alias"
        alias.symlink_to(self.target, target_is_directory=True)
        other = models.Selection.of(self.root / "models", str(alias))
        self.assertEqual(chosen.link, other.link)

    def test_prepare_and_restart_reinspect_without_target_download(self):
        chosen = self.selection()
        before = {p.name: p.read_bytes() for p in self.target.iterdir()}
        with (
            mock.patch("huggingface_hub.HfApi") as api,
            mock.patch("huggingface_hub.snapshot_download") as download,
            mock.patch.object(
                upstream, "inspect_target", wraps=upstream.inspect_target
            ) as inspect,
        ):
            upstream.prepare(chosen)
            upstream.prepare(chosen)
            self.assertEqual(inspect.call_count, 2)
            api.assert_not_called()
            download.assert_not_called()
        record = assembly.verify(chosen.link)
        self.assertEqual(
            record["sources"]["target"], {"repo": str(self.target), "revision": None}
        )
        self.assertEqual(
            (chosen.link / "target/model.safetensors").resolve(),
            self.target / "model.safetensors",
        )
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.target.iterdir()}
        )
        (self.target / "config.json").write_text("{}")
        with self.assertRaises(models.ModelError):
            upstream.prepare(chosen)
