import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import launcher

ROOT = Path(__file__).resolve().parents[3]


class TrainingLauncherTests(unittest.TestCase):
    def arguments(self):
        return [
            "train-draft",
            "--target",
            "/target",
            "--init-draft",
            "/draft",
            "--data",
            "/data",
            "--validation-data",
            "/validation",
            "--output",
            "/output",
            "--dry-run",
        ]

    def test_training_does_not_read_server_port_or_build_native_engine(self):
        with (
            mock.patch.dict(os.environ, {"SPLASH_PORT": "invalid"}),
            mock.patch.object(launcher, "_ensure_installed") as install,
        ):
            args = launcher.parse_args(self.arguments())
            self.assertEqual(args.command, "train-draft")
            self.assertFalse(hasattr(args, "port"))
            with mock.patch("draft_training.__main__.main", return_value=0) as train:
                self.assertEqual(launcher.main(self.arguments()), 0)
            train.assert_called_once_with(["train", *self.arguments()[1:]])
            install.assert_not_called()

    def test_packaged_release_help_does_not_advertise_optional_source_command(self):
        with (
            mock.patch.object(launcher.paths, "PACKAGED", True),
            mock.patch.object(launcher, "_version", return_value="test"),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            with self.assertRaises(SystemExit):
                launcher.parse_args(["--help"])
            self.assertNotIn("train-draft", output.getvalue())

    def test_source_help_does_not_import_tensor_dependencies(self):
        script = 'import sys; from install.launcher import main; assert "torch" not in sys.modules; main(["train-draft", "--help"])'
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--validation-data", result.stdout)
        self.assertIn("--preflight", result.stdout)

    def test_shell_wrapper_selects_explicit_training_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            python = Path(temporary) / "training python"
            python.write_text(
                '#!/bin/sh\nif [ "$1" = -c ]; then exit 0; fi\nprintf "%s\\n" "$@"\n'
            )
            python.chmod(0o755)
            result = subprocess.run(
                [str(ROOT / "splash"), "train-draft", "--help"],
                env={**os.environ, "SPLASH_TRAINING_PYTHON": str(python)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("train-draft", result.stdout)
            self.assertIn("install/launcher.py", result.stdout)
