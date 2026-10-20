from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from dev.benchmarks import qualification_matrix as matrix


class QualificationMatrixTests(unittest.TestCase):
    def arguments(self, **changes):
        return argparse.Namespace(
            **{
                "candidate": Path("build/candidate"),
                "baseline": Path("build/baseline"),
                "output": Path("build/qualification"),
                "suite": "all",
                "threshold": 0.1,
                "live_gdn": True,
                "tokens": 512,
                "samples": 3,
                "max_memory": "20G",
                "residency": "sequential",
                **changes,
            }
        )

    def test_matrix_covers_formats_widths_and_mixed_without_tracing_speed(self):
        jobs = matrix.jobs(
            self.arguments(),
            [{"label": "moe-gguf", "root": "install/model", "model": "owner/model"}],
        )
        self.assertEqual(len(jobs), 9)
        correctness = [job for job in jobs if "oracle" in job["id"]]
        self.assertEqual(len(correctness), 4)
        formats = set()
        for job in correctness:
            command = job["command"]
            formats.add(
                (
                    command[command.index("--kv-format") + 1],
                    command[command.index("--state-format") + 1],
                )
            )
            self.assertEqual(job["environment"]["MTL_SHADER_VALIDATION"], "1")
            self.assertIn("--live-gdn", command)
        self.assertEqual(
            formats,
            {("int8", "fp32"), ("int8", "bf16"), ("bf16", "fp32"), ("bf16", "bf16")},
        )
        serving = [job for job in jobs if "oracle" not in job["id"]]
        for job in serving:
            self.assertNotIn("--trace", job["command"])
            self.assertNotIn("MTL_SHADER_VALIDATION", job["environment"])
            self.assertEqual(
                job["command"][job["command"].index("--residency") + 1], "sequential"
            )
            self.assertEqual(
                job["command"][job["command"].index("--warmup-seconds") + 1], "0"
            )
            self.assertEqual(
                job["command"][job["command"].index("--warmup-rounds") + 1], "4"
            )
            self.assertEqual(
                job["command"][job["command"].index("--warmup-tokens") + 1], "512"
            )
        self.assertEqual(sum("mixed" in job["id"] for job in serving), 1)
        self.assertEqual(len({job["id"] for job in jobs}), len(jobs))

    def test_memory_override_and_ordinary_binary_capability(self):
        jobs = matrix.jobs(
            self.arguments(suite="serving", live_gdn=False, residency="paired"),
            [
                {
                    "label": "moe-mlx",
                    "root": "install/model",
                    "model": "owner/model",
                    "max_memory": "26G",
                }
            ],
        )
        for job in jobs:
            command = job["command"]
            self.assertEqual(command[command.index("--max-memory") + 1], "26G")
            self.assertNotIn("SPLASH_LIVE_GDN=1", command)
            self.assertEqual(command[command.index("--residency") + 1], "paired")


if __name__ == "__main__":
    unittest.main()
