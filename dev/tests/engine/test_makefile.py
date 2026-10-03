import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# A base interpreter that passes the Makefile's version checks and creates
# environments whose python logs each pip install beside the environment,
# so the environment rules run without a real interpreter or index.
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
  *) exit 1;;
esac
"""


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
        result = self.make("-n", "_install-environment", PYTHON_CANDIDATES="python3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("for candidate in python3; do", result.stdout)

    def test_language_only_is_one_or_zero(self):
        for value, text_only in (("", False), ("0", False), ("1", True)):
            with self.subTest(value=value):
                result = self.make(
                    "-n", "_install", "MODEL=owner/repo", f"LANGUAGE_ONLY={value}"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                prepare = next(
                    line
                    for line in result.stdout.splitlines()
                    if "install/models.py" in line
                )
                self.assertEqual("--language-only" in prepare, text_only)
        result = self.make("model-selection", "MODEL=owner/repo", "LANGUAGE_ONLY=yes")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LANGUAGE_ONLY is 1 (text only) or 0", result.stderr)

    def test_architecture_check_parses_with_the_environments_python(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory) / "venv"
            result = self.make("-n", "architecture-check", f"VENV={environment}")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                f"{environment}/bin/python dev/tools/check_architecture.py\n",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
