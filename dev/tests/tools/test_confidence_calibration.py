import unittest

from dev.benchmarks import confidence_calibration as calibration


class ConfidenceCalibrationTests(unittest.TestCase):
    def test_censored_suffix_is_excluded_from_calibration(self):
        common = {"context": 2048, "workload": "code", "position": 3}
        rows = [
            {**common, "observed": True, "accepted_prefix": True, "confidence": 0.4},
            {**common, "observed": False, "accepted_prefix": False, "confidence": 0.4},
        ]
        result = calibration.calibrate(rows)[0]
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["censored"], 1)
        self.assertEqual(result["bins"][0]["count"], 1)
        self.assertEqual(result["bins"][0]["prefix_survival"], 1)
        self.assertAlmostEqual(result["bins"][0]["brier_score"], 0.36)

    def test_workloads_and_contexts_do_not_pool(self):
        rows = [
            {
                "context": context,
                "workload": workload,
                "position": 1,
                "observed": True,
                "accepted_prefix": True,
                "confidence": 1.0,
            }
            for context, workload in [(2048, "code"), (8192, "code"), (2048, "prose")]
        ]
        result = calibration.calibrate(rows)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(r["bins"][0]["lower_confidence"] == 0.9 for r in result))

    def test_request_mapping_excludes_warmup_and_rejects_wrong_geometry(self):
        report = {
            "complete": True,
            "scenario": "decode",
            "settings": {"batch": 1},
            "residency": "paired",
            "warmup": {"rounds_per_arm": 1},
            "cases": [{"context": 2048, "workload": "code", "rounds": [{"server": 0}]}],
        }

        def event(name, ident, rows, context=0):
            return {
                "name": name,
                "ph": "X",
                "ts": ident,
                "dur": 0,
                "args": {
                    "id": ident,
                    "rows": rows,
                    "context": context,
                    "flags": 1,
                    "value": 0.5,
                    "width": 1,
                },
            }

        document = {
            "traceEvents": [
                event("request.active", 1, 512),
                event("request.active", 2, 2048),
                event("draft.confidence", 1, 1, 8),
                event("draft.confidence", 2, 1, 8),
            ]
        }
        rows = calibration.observations(report, document, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["workload"], "code")
        document["traceEvents"][1]["args"]["rows"] = 8192
        with self.assertRaisesRegex(ValueError, "geometry"):
            calibration.observations(report, document, 0)
        report["complete"] = False
        with self.assertRaisesRegex(ValueError, "complete"):
            calibration.observations(report, document, 0)


if __name__ == "__main__":
    unittest.main()
