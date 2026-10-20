import unittest

from dev.benchmarks.prefill_prediction import evaluate


def observation(rows=2048, history=0, wall_ms=200, **extra):
    return {
        "rows": rows,
        "maximum_history": history,
        "wall_ms": wall_ms,
        "representative": True,
        "width": 1,
        **extra,
    }


class PrefillPredictionTests(unittest.TestCase):
    def test_startup_seed_predicts_before_first_observation(self):
        report = evaluate(
            [observation(wall_ms=800)], seed={"rows": 2048, "wall_ms": 200}
        )
        self.assertEqual(report["predictions"][0]["ema_ms"], 200)
        self.assertEqual(report["predictions"][0]["bucketed_ms"], 200)

    def test_observation_cannot_train_its_own_prediction(self):
        report = evaluate([observation(wall_ms=200), observation(wall_ms=800)])
        prediction = report["predictions"][0]
        self.assertEqual(prediction["ema_ms"], 200)
        self.assertEqual(prediction["bucketed_ms"], 200)
        self.assertFalse(prediction["uses_bucket"])

    def test_insufficient_history_falls_back_and_auxiliary_ragged_samples_are_excluded(
        self,
    ):
        report = evaluate(
            [
                observation(),
                observation(),
                observation(),
                observation(wall_ms=10000, representative=False),
                observation(wall_ms=10000, width=2),
                observation(rows=1, wall_ms=10000),
                observation(history=32000, wall_ms=800),
            ]
        )
        self.assertFalse(report["predictions"][-1]["uses_bucket"])
        self.assertEqual(report["predictions"][-1]["ema_ms"], 200)
        self.assertEqual(
            report["excluded"], {"small": 1, "nonrepresentative": 1, "ragged": 1}
        )

    def test_history_buckets_reduce_error_on_repeated_cost_regimes(self):
        rows = []
        for _ in range(12):
            rows += [
                observation(history=0, wall_ms=200),
                observation(history=32000, wall_ms=800),
            ]
        report = evaluate(rows)
        self.assertLess(
            report["errors"]["bucketed"]["mean_absolute_relative_error"],
            report["errors"]["ema"]["mean_absolute_relative_error"],
        )
        self.assertTrue(report["predictions"][-1]["uses_bucket"])


if __name__ == "__main__":
    unittest.main()
