"""Exercise the real bootstrap process against deterministic fake Apple tools."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "dev/tools/ensure_metal_toolchain.py"
TOOL = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root=Path(os.environ["METAL_FIXTURE"])
name=Path(sys.argv[0]).name
args=sys.argv[1:]
with (root/"calls").open("a") as log: log.write(json.dumps([name,*args])+"\\n")
if name=="sleep": sys.exit(0)
if name=="xcode-select": print("/Fixture/Xcode.app/Contents/Developer"); sys.exit(0)
if name=="xcodebuild":
 if args==["-version"]: print("Xcode fixture"); sys.exit(0)
 mode=(root/"mode").read_text()
 attempt=len((root/"downloads").read_text())+1 if (root/"downloads").exists() else 1
 with (root/"downloads").open("a") as f:f.write("x")
 if mode in ["install", "error_but_installed"] or (mode=="retry" and attempt==2): (root/"installed").touch()
 sys.exit(0 if mode in ["install", "success_but_missing"] else 70)
if name=="xcrun":
 if "--show-sdk-path" in args: print("/Fixture/SDK"); sys.exit(0)
 if "--find" in args: print("/Fixture/shim"); sys.exit(0)
 if not (root/"installed").exists(): sys.exit(1)
 if "metallib" in args and (root/"mode").read_text()=="broken_link": sys.exit(1)
 if "-o" in args:
  output=Path(args[args.index("-o")+1])
  if (root/"mode").read_text()!="empty_output":output.write_bytes(b"compiled")
 sys.exit(0)
sys.exit(1)
"""


class MetalToolchainTests(unittest.TestCase):
    def run_bootstrap(self, mode, installed=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mode").write_text(mode)
            if installed:
                (root / "installed").touch()
            for tool in ["xcrun", "xcodebuild", "xcode-select", "sleep"]:
                path = root / tool
                path.write_text(TOOL)
                path.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(root) + os.pathsep + os.environ["PATH"],
                METAL_FIXTURE=tmp,
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            downloads = (
                len((root / "downloads").read_text())
                if (root / "downloads").exists()
                else 0
            )
            calls = (root / "calls").read_text()
            return result, downloads, calls

    def test_usable_compiler_does_not_need_a_working_asset_catalog(self):
        result, downloads, calls = self.run_bootstrap("catalog_failure", installed=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(downloads, 0)
        self.assertIn('"metallib"', calls)

    def test_missing_toolchain_is_installed_then_compiled_and_linked(self):
        result, downloads, calls = self.run_bootstrap("install")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(downloads, 1)
        self.assertIn('"metallib"', calls)

    def test_transient_catalog_failure_is_retried(self):
        result, downloads, _ = self.run_bootstrap("retry")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(downloads, 2)

    def test_installer_error_does_not_hide_a_now_usable_compiler(self):
        result, downloads, _ = self.run_bootstrap("error_but_installed")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(downloads, 1)

    def test_installer_success_alone_cannot_pass_setup(self):
        result, downloads, _ = self.run_bootstrap("success_but_missing")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(downloads, 3)

    def test_a_present_shim_or_broken_linker_cannot_pass_setup(self):
        for mode, installed in [
            ("catalog_failure", False),
            ("broken_link", True),
            ("empty_output", True),
        ]:
            with self.subTest(mode=mode):
                result, downloads, _ = self.run_bootstrap(mode, installed=installed)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(downloads, 3)


if __name__ == "__main__":
    unittest.main()
