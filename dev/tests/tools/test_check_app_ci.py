"""Exercise the CI preference guard without touching real system preferences."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FAKE_TOOL = """
import json, os, sys
from pathlib import Path
root=Path(os.environ["APP_CI_FIXTURE"])
state=root/"state"
value=json.loads(state.read_text())
if Path(sys.argv[0]).name=="make":
 (root/"during").write_text(json.dumps(value))
 sys.exit(int(os.environ.get("APP_STATUS","0")))
args=sys.argv[1:]
if args[0]=="read":
 if value is None:sys.exit(1)
 print(value);sys.exit(0)
if args[0]=="delete":state.write_text("null");sys.exit(0)
if args[0]=="write":
 new=0 if args[-1] in ["false","0"] else 1
 if new==0 and os.environ.get("DISABLE_FAIL"):sys.exit(9)
 if new==1 and os.environ.get("RESTORE_FAIL"):sys.exit(9)
 state.write_text(json.dumps(new));sys.exit(0)
sys.exit(1)
"""


class AppCiTests(unittest.TestCase):
    def run_guard(self, original, **settings):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").write_text(json.dumps(original))
            for name in ["defaults", "make"]:
                path = root / name
                path.write_text(f"#!{sys.executable}\n" + FAKE_TOOL)
                path.chmod(0o755)
            result = subprocess.run(
                ["/bin/sh", str(ROOT / "dev/tools/check_app_ci.sh")],
                env=dict(
                    os.environ,
                    PATH=str(root) + os.pathsep + os.environ["PATH"],
                    APP_CI_FIXTURE=tmp,
                    **settings,
                ),
                capture_output=True,
                text=True,
                timeout=10,
            )
            during = (
                json.loads((root / "during").read_text())
                if (root / "during").exists()
                else None
            )
            return result, json.loads((root / "state").read_text()), during

    def test_enabled_preference_is_off_during_tests_then_restored(self):
        result, after, during = self.run_guard(1)
        self.assertEqual((result.returncode, after, during), (0, 1, 0))

    def test_disabled_preference_stays_disabled(self):
        result, after, during = self.run_guard(0)
        self.assertEqual((result.returncode, after, during), (0, 0, 0))

    def test_absent_preference_is_removed_after_tests(self):
        result, after, during = self.run_guard(None)
        self.assertEqual((result.returncode, after, during), (0, None, 0))

    def test_app_failure_is_preserved_and_preference_restored(self):
        result, after, during = self.run_guard(1, APP_STATUS="17")
        self.assertEqual((result.returncode, after, during), (17, 1, 0))

    def test_failed_disable_does_not_run_tests(self):
        result, after, during = self.run_guard(1, DISABLE_FAIL="1")
        self.assertEqual((result.returncode, after, during), (9, 1, None))

    def test_failed_restoration_fails_the_gate(self):
        result, _, during = self.run_guard(1, RESTORE_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(during, 0)

    def test_unexpected_original_value_is_unchanged(self):
        result, after, during = self.run_guard(7)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((after, during), (7, None))


if __name__ == "__main__":
    unittest.main()
