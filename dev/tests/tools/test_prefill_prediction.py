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

    def test_ragged_history_includes_nonleading_lane_and_width_has_own_cell(self):
        lanes = [
            {"lane": 0, "rows": 128, "history": 0},
            {"lane": 1, "rows": 896, "history": 32000},
        ]
        rows = [
            observation(rows=1024, history=32000, wall_ms=800, width=2, lanes=lanes)
        ] * 4
        rows += [observation(rows=1024, history=28000, wall_ms=800)]
        report = evaluate(rows)
        self.assertTrue(report["predictions"][-2]["uses_bucket"])
        self.assertEqual(report["predictions"][-2]["history"], 28000)
        self.assertFalse(report["predictions"][-1]["uses_bucket"])
        self.assertEqual(report["excluded"]["ragged"], 0)
        rows[0] = {**rows[0], "lanes": [lanes[0]]}
        with self.assertRaisesRegex(ValueError, "ragged"):
            evaluate(rows)

    def test_gpu_and_ane_costs_use_distinct_prior_samples(self):
        rows = [observation(predicted_ane=True)] * 3
        rows += [
            observation(predicted_ane=False, wall_ms=800),
            observation(predicted_ane=True),
        ]
        report = evaluate(rows)
        self.assertFalse(report["predictions"][-2]["uses_bucket"])
        self.assertTrue(report["predictions"][-1]["uses_bucket"])


if __name__ == "__main__":
    unittest.main()
