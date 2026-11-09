import copy
import json
import tempfile
import unittest
from pathlib import Path

from dev.benchmarks.draft_comparison import NAMES, summarize, target_sources


class DraftComparisonTests(unittest.TestCase):
    def rounds(self):
        baseline = {
            "solo": {
                name: {
                    "usage": {"completion_tokens": 11 if i % 2 == 0 else 101},
                    "latency": {"stream_tokens_per_second": 10 if i % 2 == 0 else 20},
                    "output_sha256": name,
                }
                for i, name in enumerate(NAMES)
            },
            "batch_wall_seconds": 4,
        }
        baseline["batch"] = copy.deepcopy(baseline["solo"])
        candidate = copy.deepcopy(baseline)
        for request in candidate["solo"].values():
            request["latency"]["stream_tokens_per_second"] *= 2
        candidate["batch_wall_seconds"] = 2
        return [baseline, candidate, copy.deepcopy(candidate), copy.deepcopy(baseline)]

    def test_weights_decode_time_and_excludes_prefill_token(self):
        result = summarize(self.rounds())
        combined = result["milliseconds_per_token"]["combined_solo"]
        # 220 decode tokens over 12 seconds, then over 6 seconds. An average
        # of the four request rates would give a different answer.
        self.assertAlmostEqual(combined["baseline"], 1000 * 12 / 220)
        self.assertAlmostEqual(combined["candidate"], 1000 * 6 / 220)
        self.assertAlmostEqual(combined["regression"], -0.5)
        # Concurrent end-to-end throughput includes all 224 completion tokens.
        batch = result["milliseconds_per_token"]["four_concurrent"]
        self.assertAlmostEqual(batch["baseline"], 1000 * 4 / 224)
        self.assertTrue(result["solo_output_hashes_match"])

    def test_marks_noisy_comparison_and_changed_output(self):
        rounds = self.rounds()
        rounds[3]["solo"][NAMES[0]]["latency"]["stream_tokens_per_second"] = 5
        rounds[2]["solo"][NAMES[1]]["output_sha256"] = "different"
        result = summarize(rounds)
        self.assertEqual(
            result["milliseconds_per_token"][NAMES[0]]["verdict"], "inconclusive"
        )
        self.assertFalse(result["solo_output_hashes_match"])

    def test_compares_target_sources_independently_of_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"hidden_size": 5120}')
            manifest = {
                "files": {
                    "target/model.safetensors": {"bytes": 100, "digest": "same-target"},
                    "draft/model.safetensors": {"bytes": 10, "digest": "first-draft"},
                }
            }
            (root / "model.json").write_text(json.dumps(manifest))
            baseline = target_sources(root)
            manifest["files"]["draft/model.safetensors"]["digest"] = "other-draft"
            (root / "model.json").write_text(json.dumps(manifest))
            self.assertEqual(target_sources(root), baseline)
            manifest["files"]["target/model.safetensors"]["digest"] = "other-target"
            (root / "model.json").write_text(json.dumps(manifest))
            self.assertNotEqual(target_sources(root), baseline)


if __name__ == "__main__":
    unittest.main()
