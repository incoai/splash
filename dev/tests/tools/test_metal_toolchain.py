import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/scripts/install-metal-toolchain.sh"


class MetalToolchainTests(unittest.TestCase):
    def run_installer(self, mode):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            commands = {
                "xcrun": """#!/bin/sh
printf 'probe\\n' >> "$TEST_ROOT/events"
test "$TEST_MODE" = installed && exit 0
test "$TEST_MODE" = transient && test -f "$TEST_ROOT/ready"
""",
                "xcodebuild": """#!/bin/sh
printf 'download\\n' >> "$TEST_ROOT/events"
count=0
if test -f "$TEST_ROOT/count"; then count=$(cat "$TEST_ROOT/count"); fi
count=$((count + 1))
printf '%s' "$count" > "$TEST_ROOT/count"
if test "$TEST_MODE" = transient && test "$count" -eq 3; then
  touch "$TEST_ROOT/ready"
  exit 0
fi
test "$TEST_MODE" = false_success
""",
                "sleep": '#!/bin/sh\nprintf "wait %s\\n" "$1" >> "$TEST_ROOT/events"\n',
            }
            for name, text in commands.items():
                path = root / name
                path.write_text(text)
                path.chmod(0o755)
            result = subprocess.run(
                ["/bin/bash", str(SCRIPT)],
                env={
                    **os.environ,
                    "PATH": f"{root}:/usr/bin:/bin",
                    "TEST_ROOT": str(root),
                    "TEST_MODE": mode,
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result, (root / "events").read_text().splitlines()

    def test_available_compiler_needs_no_catalog_or_wait(self):
        result, events = self.run_installer("installed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(events, ["probe"])

    def test_transient_catalog_failures_recover_with_bounded_waits(self):
        result, events = self.run_installer("transient")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(events.count("download"), 3)
        self.assertEqual(
            [x for x in events if x.startswith("wait")], ["wait 15", "wait 30"]
        )
        self.assertEqual(events[-1], "probe")

    def test_successful_download_without_compiler_still_fails(self):
        result, events = self.run_installer("false_success")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events.count("download"), 3)
        self.assertEqual(events.count("probe"), 4)

    def test_persistent_catalog_failure_still_fails(self):
        result, events = self.run_installer("unavailable")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events.count("download"), 3)
        self.assertEqual(events.count("probe"), 1)
