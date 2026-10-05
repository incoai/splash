import hashlib
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.benchmarks import backend_regression, weights
from dev.tests import agent_real, smoke_real

ROOT = Path(__file__).resolve().parents[3]

# A base interpreter that passes the Makefile's version checks and creates
# environments whose python logs each pip install, and each program it is
# asked to run, beside the environment, so the rules run without a real
# interpreter, index or program.
BOOTSTRAP = """#!/bin/sh
here=$(cd "$(dirname "$0")" && pwd)
case "$1" in
  -c) case "$2" in *_base_executable*) echo "$here/python3";; esac; exit 0;;
  -m) test "$3" = --clear && { rm -rf "$4"; set -- "$1" "$2" "$4"; }
      mkdir -p "$3/bin" && echo "home = $here" > "$3/pyvenv.cfg"
      cp "$here/environment-python" "$3/bin/python"; exit 0;;
esac
exit 1
"""
ENVIRONMENT_PYTHON = """#!/bin/sh
environment=$(cd "$(dirname "$0")/.." && pwd)
case "$1 $2 $3" in
  -c*) test ! -e "$environment/broken";;
  "-m pip install") shift 3; echo "$*" >> "$environment/../pip.log";;
  "-m pip --version"|"-m pip check") ;;
  *) echo "$*" >> "$environment/../python.log";;
esac
"""

# make install builds the engine first, as the installer checks a model's
# configuration with it; no installer runs under the fake interpreter, so
# these tests take the engine as built.
ENGINE_BUILT = ("-o", "build/splash")


# What a calling make or shell exports that would configure the make tested.
INHERITED = (
    "MAKEFLAGS",
    "MFLAGS",
    "MAKELEVEL",
    "MODEL",
    "REVISION",
    "DRAFT_MODEL",
    "LANGUAGE_ONLY",
    "PYTHON_CANDIDATES",
)


class MakefileTests(unittest.TestCase):
    def make(self, *arguments, **environment):
        inherited = {k: v for k, v in os.environ.items() if k not in INHERITED}
        return subprocess.run(
            ("make", "--no-print-directory", *arguments),
            cwd=ROOT,
            env={**inherited, **environment},
            capture_output=True,
            text=True,
            timeout=120,
        )

    def interpreter(self, directory: Path) -> Path:
        for name, script in (
            ("python3", BOOTSTRAP),
            ("environment-python", ENVIRONMENT_PYTHON),
        ):
            (directory / name).write_text(script)
            (directory / name).chmod(0o755)
        return directory / "python3"

    def test_a_rebuilt_environment_gets_the_development_requirements_again(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            environment = directory / "venv"
            arguments = (
                "-j4",
                "install-development",
                f"VENV={environment}",
                f"PYTHON_CANDIDATES={self.interpreter(directory)}",
            )
            installs = [
                "--only-binary=:all: -r install/requirements.txt",
                "-r dev/requirements.txt",
            ]
            for rebuild in (False, True):
                with self.subTest(rebuild=rebuild):
                    if rebuild:
                        (environment / "broken").touch()
                    result = self.make(*arguments)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    log = (directory / "pip.log").read_text().splitlines()
                    self.assertEqual(log, installs * (1 + rebuild))
                    self.assertEqual(
                        (environment / ".dev-requirements-installed").read_text(),
                        hashlib.sha256(
                            (ROOT / "dev/requirements.txt").read_bytes()
                        ).hexdigest()
                        + "\n",
                    )
            # A current environment installs nothing.
            self.assertEqual(self.make(*arguments).returncode, 0)
            self.assertEqual(len((directory / "pip.log").read_text().splitlines()), 4)

    def test_python_candidates_may_come_from_the_environment(self):
        # As the release job sets them.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            environment = directory / "venv"
            result = self.make(
                "install-environment",
                f"VENV={environment}",
                PYTHON_CANDIDATES=str(self.interpreter(directory)),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (environment / "pyvenv.cfg").read_text(), f"home = {directory}\n"
            )

    def test_language_only_is_one_or_zero(self):
        for value, text_only in (("", False), ("0", False), ("1", True)):
            with (
                self.subTest(value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                directory = Path(directory)
                result = self.make(
                    *ENGINE_BUILT,
                    "install",
                    "MODEL=owner/repo",
                    f"LANGUAGE_ONLY={value}",
                    f"VENV={directory / 'venv'}",
                    f"PYTHON_CANDIDATES={self.interpreter(directory)}",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                language_only = "--language-only " if text_only else ""
                self.assertEqual(
                    (directory / "python.log").read_text(),
                    f"install/models.py --model owner/repo {language_only}prepare\n",
                )
        result = self.make("model-selection", "MODEL=owner/repo", "LANGUAGE_ONLY=yes")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LANGUAGE_ONLY is 1 (text only) or 0", result.stderr)

    def test_architecture_check_runs_on_the_environments_python(self):
        # It parses the server's sources, which python3, possibly Xcode's
        # 3.9, cannot.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            result = self.make(
                "architecture-check",
                f"VENV={directory / 'venv'}",
                f"PYTHON_CANDIDATES={self.interpreter(directory)}",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (directory / "python.log").read_text(),
                "dev/tools/check_architecture.py\n",
            )

    def test_the_real_model_scripts_take_the_arguments_make_passes(self):
        # Only a release runs these scripts, on a model. Each parses the
        # command lines make -n prints for its target, for a selection with
        # every source option and its installation in MODEL_ROOT.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            # backend_regression requires the installation, and builds in
            # BASELINE and in this checkout, its candidate.
            model_root = directory / "model"
            model_root.mkdir()
            (model_root / "model.json").touch()
            baseline, candidate = directory / "baseline", directory / "candidate"
            for checkout in (baseline, candidate):
                (checkout / "build/engine-tests").mkdir(parents=True)
                (checkout / backend_regression.BENCHMARK).touch()
                (checkout / backend_regression.METALLIB).touch()
            (candidate / "build" / weights.WEIGHT_DIGESTS).touch()
            for target, script, parse_args in (
                ("test-http-real", "dev/tests/smoke_real.py", smoke_real.parse_args),
                ("release-check", "dev/tests/smoke_real.py", smoke_real.parse_args),
                ("test-agent-real", "dev/tests/agent_real.py", agent_real.parse_args),
                ("test-release-real", "dev/tests/agent_real.py", agent_real.parse_args),
                (
                    "test-performance-real",
                    "dev.benchmarks.backend_regression",
                    backend_regression.parse_args,
                ),
            ):
                with (
                    self.subTest(target=target),
                    mock.patch.object(backend_regression, "ROOT", candidate),
                ):
                    result = self.make(
                        "-n",
                        target,
                        "MODEL=owner/repo",
                        "REVISION=main",
                        "DRAFT_MODEL=owner/draft",
                        "LANGUAGE_ONLY=1",
                        f"MODEL_ROOT={model_root}",
                        f"BASELINE={baseline}",
                        f"VENV={directory / 'venv'}",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    commands = [
                        shlex.split(line)
                        for line in result.stdout.replace("\\\n", "").splitlines()
                        if script in line
                    ]
                    self.assertTrue(commands, f"{target} does not run {script}")
                    for command in commands:
                        arguments = parse_args(command[command.index(script) + 1 :])
                        self.assertEqual(arguments.package, model_root)


if __name__ == "__main__":
    unittest.main()
