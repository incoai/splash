import unittest

from dev.benchmarks.draft_acceptance import prefix_profile


class DraftAcceptanceTests(unittest.TestCase):
    def test_reached_denominator_and_censored_cycles(self):
        cycles = [
            {"drafted_tokens": 3, "accepted_draft_tokens": n} for n in [0, 1, 3]
        ] + [
            {"drafted_tokens": 1, "accepted_draft_tokens": 1},
            {"drafted_tokens": 3, "accepted_draft_tokens": 3},
        ]
        profile = prefix_profile(cycles, 3)
        self.assertEqual(profile["eligible_cycles"], 3)
        self.assertEqual(profile["excluded_partial_cycles"], 1)
        self.assertEqual(profile["excluded_final_cycles"], 1)
        self.assertEqual(profile["accepted_prefix_histogram"], [1, 1, 0, 1])
        self.assertEqual(profile["positions"][1]["reached"], 2)
        self.assertEqual(profile["positions"][1]["conditional_acceptance"], 0.5)
        self.assertAlmostEqual(profile["positions"][1]["survival"], 1 / 3)

    def test_unreached_positions_have_no_conditional_estimate(self):
        cycles = [{"drafted_tokens": 2, "accepted_draft_tokens": 0}] * 2
        profile = prefix_profile(cycles, 2)
        self.assertIsNone(profile["positions"][1]["conditional_acceptance"])
        self.assertEqual(profile["positions"][1]["survival"], 0)
        self.assertIsNone(prefix_profile([], 2)["full_acceptance_fraction"])

    def test_invalid_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            prefix_profile([{"drafted_tokens": 3, "accepted_draft_tokens": 4}], 3)


if __name__ == "__main__":
    unittest.main()
