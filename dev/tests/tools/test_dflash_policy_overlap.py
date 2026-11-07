import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dev.benchmarks.dflash_policy_overlap import (
    analyze,
    analyze_row,
    context_digest,
    main,
    penalize,
    proposal,
    rejection_reference,
)


def fixture():
    prompt, output, prefix = [0, 1], [1], [2]
    digest = context_digest(prompt, output, prefix)
    return {
        "schema": "dflash-policy-overlap-v1",
        "distribution_source": "test_fixture",
        "target_identity": "fixture-target",
        "draft_identity": "fixture-draft",
        "tokenizer_identity": "fixture-tokenizer",
        "records": [
            {
                "prompt_tokens": prompt,
                "output_tokens": output,
                "within_block_prefix": prefix,
                "target_context_sha256": digest,
                "proposal_context_sha256": digest,
                "target_probabilities": [0.1, 0.2, 0.3, 0.4],
                "candidate_ids": [0, 1, 2],
                "candidate_scores": [1.0, 2.0, 3.0],
                "proposal_temperature": 1.0,
                "penalties": {"repetition": 1.2, "presence": 0.4, "frequency": 0.2},
            }
        ],
    }


class PolicyOverlapTests(unittest.TestCase):
    def test_output_counts_differ_from_prompt_repetition(self):
        self.assertEqual(
            penalize(4.0, 1, {1}, Counter(), repetition=2, presence=3, frequency=2), 2.0
        )
        self.assertEqual(
            penalize(
                -4.0, 1, set(), Counter({1: 2}), repetition=2, presence=3, frequency=2
            ),
            -15.0,
        )
        self.assertEqual(
            penalize(
                4.0, 1, set(), Counter({1: 2}), repetition=1, presence=-3, frequency=-2
            ),
            11.0,
        )

    def test_tail_mass_and_default_fast_control(self):
        doc = fixture()
        row = doc["records"][0]
        row["penalties"] = dict(repetition=1.0, presence=0.0, frequency=0.0)
        result = analyze(doc)["rows"][0]
        self.assertAlmostEqual(result["target_mass_outside_support"], 0.4)
        variants = list(result["variants"].values())
        self.assertEqual(variants[0], variants[1])
        self.assertEqual(variants[1], variants[2])
        self.assertLessEqual(variants[0]["overlap"], 0.6 + 1e-12)

    def test_prefix_penalty_is_a_separate_factor_and_can_hurt(self):
        row = fixture()["records"][0]
        result = analyze_row(row)
        self.assertNotEqual(
            result["variants"]["history"], result["variants"]["history_and_prefix"]
        )
        row["target_probabilities"] = [0.0, 0.0, 1.0, 0.0]
        result = analyze_row(row)["variants"]
        self.assertLess(
            result["history_and_prefix"]["overlap"], result["history"]["overlap"]
        )

    def test_exact_rejection_correction_on_many_full_distributions(self):
        rng = random.Random(19)
        for _ in range(100):
            p = [rng.random() for _ in range(5)]
            p = [x / sum(p) for x in p]
            q = proposal([0, 2, 4], [rng.random() for _ in range(3)], 1.0, 5)
            law = rejection_reference(p, q)
            for actual, expected in zip(law["result"], p):
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            rejection_reference([1.0, 0.0], [1.0, 0.0])["result"], [1.0, 0.0]
        )
        self.assertEqual(
            rejection_reference([0.0, 1.0], [1.0, 0.0])["result"], [0.0, 1.0]
        )

    def test_sampled_frequencies_and_greedy_token_ties(self):
        q = proposal([2, 0, 1], [1.0, 1.0, 0.0], 1.0, 3)
        draws = Counter(random.Random(3).choices(range(3), weights=q, k=40000))
        for i in range(3):
            self.assertAlmostEqual(draws[i] / 40000, q[i], delta=0.01)
        self.assertEqual(proposal([2, 0, 1], [1.0, 1.0, 0.0], 0.0, 3), [1.0, 0.0, 0.0])

    def test_bad_prefix_or_missing_tail_is_rejected(self):
        for field, value in (
            ("target_context_sha256", "wrong"),
            ("target_probabilities", [0.1] * 4),
            ("proposal_temperature", -1),
            ("candidate_ids", [0, 0, 2]),
        ):
            doc = fixture()
            doc["records"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze(doc)

    def test_cli_round_trip_preserves_fixture_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = (
                Path(directory) / "capture.json",
                Path(directory) / "report.json",
            )
            source.write_text(json.dumps(fixture()))
            main([str(source), "--output", str(output)])
            self.assertEqual(
                json.loads(output.read_text())["distribution_source"], "test_fixture"
            )
