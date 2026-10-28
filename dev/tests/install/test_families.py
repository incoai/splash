import io
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from install import families, launcher, models
from server import serve_options

ROOT = Path(__file__).resolve().parents[3]

# The options of `splash serve` the models of each family cannot use, as the
# engine's facts of its layout and the family's release give them.
UNUSABLE = {
    "Qwen3.8-27B": {
        "--expert-memory",
        "--ple-memory",
        "--sparse-kv-memory",
        "--verify-rows",
    },
    "Qwen3.6-35B-A3B": {"--ple-memory", "--sparse-kv-memory", "--disable-ane"},
    "Qwen3.5-122B-A10B": {"--ple-memory", "--sparse-kv-memory", "--disable-ane"},
    "Qwen3.8-Flash-Next": {"--disable-ane"},
    "Flint": {
        "--language-only",
        "--state-format",
        "--ple-memory",
        "--sparse-kv-memory",
        "--disable-ane",
        "--max-image-pixels",
        "--reasoning",
        "--default-reasoning-effort",
        "--thinking-token-budget",
        "--thinking-end-text",
    },
}


class OptionReportTests(unittest.TestCase):
    def test_each_family_is_offered_the_options_its_models_can_use(self):
        # The engine's own facts (its families command), as the app reads
        # them through the script.
        engine_facts = json.loads(models.run_engine(["families"], "family report"))
        report = families.option_report(engine_facts)
        self.assertEqual(
            {name: set(family["unusable"]) for name, family in report.items()},
            UNUSABLE,
        )
        # Flint serves its native window alone; YaRN stretches the others'.
        self.assertEqual(
            {
                name: (family["context_tokens"], family["max_context_tokens"])
                for name, family in report.items()
            },
            {
                name: (262144, 262144 if name == "Flint" else 1010000)
                for name in UNUSABLE
            },
        )
        # --max-context takes at most the most context any family serves.
        self.assertEqual(
            serve_options.MAX_YARN_CONTEXT_TOKENS,
            max(facts["max_context_tokens"] for facts in engine_facts.values()),
        )

    def test_every_option_that_needs_a_fact_is_an_option_of_splash_serve(self):
        needs = {
            option.flag: option.needs
            for option in serve_options.SERVE_OPTIONS
            if option.needs is not None
        } | families.LAUNCHER_NEEDS
        with (
            mock.patch("sys.stdout", io.StringIO()) as output,
            self.assertRaises(SystemExit),
        ):
            launcher.parse_args(["serve", "--help"])
        flags = set(re.findall(r"(?m)^  (--[a-z-]+)", output.getvalue()))
        self.assertLessEqual(set(needs), flags)

    def test_the_script_prints_the_report(self):
        # As the app runs it: isolated, from its own folder.
        output = subprocess.run(
            [sys.executable, "-I", "-B", str(ROOT / "install/families.py")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        report = json.loads(output.splitlines()[-1])
        self.assertEqual(
            {name: set(family["unusable"]) for name, family in report.items()},
            UNUSABLE,
        )


if __name__ == "__main__":
    unittest.main()
