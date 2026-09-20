import concurrent.futures
import math
import unittest
from unittest import mock

from server.latency import BUCKETS, LatencyMetrics, RequestLatency, prometheus_latency
from server.metrics import prometheus_metrics


class LatencyTests(unittest.TestCase):
    def test_buckets_boundaries_and_invalid_samples(self):
        metrics = LatencyMetrics()
        for duration in (0, 0.001, 0.002, 1801, -1, math.inf, math.nan):
            metrics.observe("ttft", duration)
        sample = metrics.snapshot()["ttft"]
        self.assertEqual(sample["count"], 4)
        self.assertEqual(list(sample["buckets"].values())[:2], [2, 3])
        self.assertEqual(sample["buckets"]["+Inf"], 4)
        text = prometheus_metrics({"latency": metrics.snapshot()})
        self.assertIn('splash_ttft_seconds_bucket{le="0.001"} 2', text)
        self.assertIn('splash_ttft_seconds_bucket{le="+Inf"} 4', text)
        self.assertIn("splash_ttft_seconds_count 4", text)
        self.assertEqual(len(sample["buckets"]), len(BUCKETS) + 1)

    def test_timer_records_failures_and_native_batches_are_not_individual_tokens(self):
        metrics = LatencyMetrics()
        with mock.patch(
            "server.latency.time.monotonic", side_effect=[10, 11, 15, 15.5, 16.5]
        ):
            with self.assertRaises(ValueError):
                with metrics.measure("template"):
                    raise ValueError("bad template")
            request = RequestLatency(metrics, received_at=9)
            request.tokens()
            request.tokens()
            request.tokens()
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["template"]["sum"], 1)
        self.assertEqual(snapshot["ttft"]["sum"], 6)
        self.assertEqual(snapshot["ttft"]["count"], 1)
        self.assertEqual(snapshot["output_interval"]["sum"], 1.5)
        self.assertEqual(snapshot["output_interval"]["count"], 2)

    def test_parallel_recording_and_snapshot_ownership(self):
        metrics = LatencyMetrics()

        def worker(_):
            for _ in range(1000):
                metrics.observe("tokenization", 0.01)

        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            list(pool.map(worker, range(4)))
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["tokenization"]["count"], 4000)
        snapshot["tokenization"]["buckets"].clear()
        self.assertEqual(metrics.snapshot()["tokenization"]["count"], 4000)
        self.assertTrue(prometheus_latency(metrics.snapshot()))


if __name__ == "__main__":
    unittest.main()
