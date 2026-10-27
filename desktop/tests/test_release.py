import hashlib
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/release.sh"


class ReleaseTests(unittest.TestCase):
    def test_missing_archive_fails_validation(self):
        result = subprocess.run(["/bin/sh", str(SCRIPT), "--validate"],
                                capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SPLASH_RUNTIME_ARCHIVE is missing", result.stderr)

    def test_mismatched_checksum_fails_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "runtime.tar.gz"
            archive.write_bytes(b"fixture")
            environment = dict(os.environ, SPLASH_RUNTIME_ARCHIVE=str(archive),
                               SPLASH_RUNTIME_VERSION="1.4.0", SPLASH_RUNTIME_SHA256="0" * 64)
            result = subprocess.run(["/bin/sh", str(SCRIPT), "--validate"],
                                    capture_output=True, text=True, env=environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("runtime checksum mismatch", result.stderr)

    def test_valid_runtime_archive_passes_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "splash-1.4.0-arm64-macos26"
            (folder / "python/bin").mkdir(parents=True)
            (folder / "install").mkdir()
            (folder / "release.json").write_text('{"version":"1.4.0"}')
            python = folder / "python/bin/python3"
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o755)
            (folder / "install/launcher.py").write_text("# fixture\n")
            archive = root / "runtime.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(folder, arcname=folder.name)
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            environment = dict(os.environ, SPLASH_RUNTIME_ARCHIVE=str(archive),
                               SPLASH_RUNTIME_VERSION="1.4.0", SPLASH_RUNTIME_SHA256=checksum)
            result = subprocess.run(["/bin/sh", str(SCRIPT), "--validate"],
                                    capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_archive_without_executable_runtime_fails_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "splash-1.4.0-arm64-macos26"
            folder.mkdir()
            (folder / "release.json").write_text('{"version":"1.4.0"}')
            archive = root / "runtime.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(folder, arcname=folder.name)
            environment = dict(os.environ, SPLASH_RUNTIME_ARCHIVE=str(archive),
                               SPLASH_RUNTIME_VERSION="1.4.0",
                               SPLASH_RUNTIME_SHA256=hashlib.sha256(archive.read_bytes()).hexdigest())
            result = subprocess.run(["/bin/sh", str(SCRIPT), "--validate"],
                                    capture_output=True, text=True, env=environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("runtime archive is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
