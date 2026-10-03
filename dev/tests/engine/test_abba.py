import unittest

from dev.benchmarks import abba


class AbbaTests(unittest.TestCase):
    def test_regression_within_the_floor_passes(self):
        # b = 100, c = 101.9: 1.9% slower with no spread.
        result = abba.compare(100, 101.9, 101.9, 100)
        self.assertEqual(result["verdict"], abba.PASS)
        self.assertTrue(result["pass"])
        self.assertAlmostEqual(result["regression"], 0.019)
        self.assertEqual(result["bound"], abba.REGRESSION_FLOOR)
        self.assertEqual(result["noise"], 0)

    def test_regression_above_the_floor_fails_on_a_quiet_run(self):
        result = abba.compare(100, 103, 103, 100)
        self.assertEqual(result["verdict"], abba.FAIL)
        self.assertFalse(result["pass"])

    def test_bound_scales_with_the_measured_spread(self):
        # Baseline rounds 98 and 102: noise 4%, bound 8%, regression 3%.
        result = abba.compare(98, 103, 103, 102)
        self.assertAlmostEqual(result["noise"], 0.04)
        self.assertAlmostEqual(result["bound"], 0.08)
        self.assertEqual(result["verdict"], abba.PASS)
        # The same spread measured on the candidate's rounds counts too.
        result = abba.compare(100, 110, 100, 100)
        self.assertAlmostEqual(result["noise"], 10 / 105)
        self.assertEqual(result["verdict"], abba.INCONCLUSIVE)
        # A regression beyond twice the spread still fails.
        result = abba.compare(99, 110, 110, 101)
        self.assertAlmostEqual(result["bound"], 0.04)
        self.assertEqual(result["verdict"], abba.FAIL)

    def test_a_noisy_run_is_never_a_pass(self):
        # 6% baseline spread; the candidate is faster, yet nothing is measured.
        result = abba.compare(97, 80, 80, 103)
        self.assertGreater(result["noise"], abba.NOISE_LIMIT)
        self.assertEqual(result["verdict"], abba.INCONCLUSIVE)
        self.assertFalse(result["pass"])
        self.assertIn("rerun idle", abba.describe(result))
        # Exactly the limit is still a measurement.
        result = abba.compare(97.5, 100, 100, 102.5)
        self.assertAlmostEqual(result["noise"], abba.NOISE_LIMIT)
        self.assertEqual(result["verdict"], abba.PASS)

    def test_a_faster_candidate_passes(self):
        result = abba.compare(100, 80, 81, 101)
        self.assertLess(result["regression"], 0)
        self.assertEqual(result["verdict"], abba.PASS)

    def test_round_order_matters(self):
        # Baseline slow and candidate fast, or the reverse: ABBA, not AABB.
        self.assertEqual(abba.compare(100, 110, 110, 100)["verdict"], abba.FAIL)
        self.assertEqual(abba.compare(110, 100, 100, 110)["verdict"], abba.PASS)

    def test_samples_reduce_to_round_medians(self):
        result = abba.compare_samples([[100, 1, 101], [100], [100, 99, 500], [100]])
        self.assertEqual(result["rounds"], [100, 100, 100, 100])
        self.assertEqual(result["verdict"], abba.PASS)
        for rounds in ([[1], [1], [1]], [[1], [], [1], [1]], [[1]] * 5):
            with self.subTest(rounds=rounds), self.assertRaises(ValueError):
                abba.compare_samples(rounds)

    def test_invalid_samples_are_refused_wherever_they_fall(self):
        # A median over a NaN is NaN or any sample, by the NaN's position.
        for bad in (float("nan"), float("inf"), 0, -1.0, None, True):
            for position in range(3):
                samples = [100.0, 101.0]
                samples.insert(position, bad)
                with (
                    self.subTest(samples=samples),
                    self.assertRaisesRegex(ValueError, "samples"),
                ):
                    abba.compare_samples([samples, [100.0], [100.0], [100.0]])

    def test_invalid_medians_are_refused(self):
        for medians in (
            (0, 1, 1, 1),
            (1, -1, 1, 1),
            (1, 1, float("nan"), 1),
            (1, 1, 1, float("inf")),
            (1, 1, 1, None),
            (1, 1, 1, True),
        ):
            with self.subTest(medians=medians), self.assertRaises(ValueError):
                abba.compare(*medians)


if __name__ == "__main__":
    unittest.main()
