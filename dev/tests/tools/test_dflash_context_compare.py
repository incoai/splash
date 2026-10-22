import copy
import unittest

from dev.benchmarks.dflash_context_compare import bf16, compare
from dev.tests.tools.test_draft_replay import fixture


def capture():
    meta, row = fixture()
    meta.update(
        model_fingerprint="fixture",
        prompt_sha256="a" * 64,
        prompt_tokens=10,
        kv_format="int8",
        state_format="fp32",
        presence_penalty=0.0,
        restore_prefix=0,
        materialize_prefix=0,
        verify_rows="all",
        budget_learning="live",
    )
    row["context_sha256"] = "b" * 64
    row["draft_hidden_bf16"] = [0x3F80] * 8
    return meta, [row]


class ContextComparisonTests(unittest.TestCase):
    def test_identical_context_and_zero_drift(self):
        data = capture()
        report = compare(data, copy.deepcopy(data))
        self.assertEqual(report["matched_contexts"], 1)
        self.assertEqual(report["comparisons"][0]["hidden_max_abs"], 0)

    def test_known_bf16_drift(self):
        left = capture()
        right = copy.deepcopy(left)
        right[1][0]["draft_hidden_bf16"][0] = 0x4000
        report = compare(left, right)["comparisons"][0]
        self.assertEqual(report["hidden_max_abs"], 1)
        self.assertAlmostEqual(report["hidden_rms"], 1 / (8**0.5))

    def test_mismatched_prompt_policy_and_divergent_context_are_rejected(self):
        for field, value in (
            ("prompt_sha256", "c" * 64),
            ("kv_format", "bf16"),
            ("presence_penalty", 1),
        ):
            left, right = capture(), capture()
            right[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare(left, right)
        left, right = capture(), capture()
        right[1][0]["context_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "no identical contexts"):
            compare(left, right)

    def test_invalid_hidden_and_missing_partner(self):
        with self.assertRaises(ValueError):
            bf16([0x7F80])
        with self.assertRaises(ValueError):
            bf16([True])
        left, right = capture(), capture()
        del right[1][0]["draft_hidden_bf16"]
        with self.assertRaises(ValueError):
            compare(left, right)

    def test_adaptive_scratch_matches_beyond_retention_are_censored(self):
        from dev.benchmarks.draft_replay import summarize

        meta, row = fixture((10, 20, 30))
        row.update(verify_rows=2, retained=2)
        row["target_tokens"] = row["target_tokens"][:2]
        row["output_tokens"] = row["output_tokens"][:2]
        report = summarize(meta, [row])
        self.assertEqual(report["accepted_proposals"], 1)
        self.assertEqual(report["first_rejections"]["censored"], 1)
        row["retained"] = 3
        with self.assertRaises(ValueError):
            summarize(meta, [row])

    def test_archived_capture_integrity_and_reports(self):
        import hashlib
        import json
        from pathlib import Path

        from dev.benchmarks.draft_replay import read_capture

        root = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/results/dflash-restored-context-m5-pro"
        )
        manifest = json.loads((root / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            self.assertEqual(
                hashlib.sha256((root / name).read_bytes()).hexdigest(), digest, name
            )
        for case in manifest["cases"]:
            left, right = [read_capture(root / name) for name in case["captures"]]
            report = compare(left, right)
            self.assertEqual(report, json.loads((root / case["report"]).read_text()))
            self.assertEqual(report["matched_contexts"], 4)
