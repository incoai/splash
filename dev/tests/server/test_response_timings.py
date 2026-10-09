import json
import math
import unittest

from dev.tests.server_fixtures import (
    FakeRuntime,
    Harness,
    HarnessTestCase,
    Plan,
    chat_body,
)
from server.backend import CacheInfo, NativeResult
from server.metrics import metrics_dict, timings_dict


class ResponseTimingsTests(HarnessTestCase):
    def test_http_counts_and_rates_with_and_without_usage(self):
        for stream, usage in ((False, False), (True, False), (True, True)):
            for batches in ([[14, 15]], [[14, 15], [14, 15, 4]]):
                for cached in (0, 1):
                    with self.subTest(
                        stream=stream, usage=usage, batches=batches, cached=cached
                    ):
                        harness = Harness(
                            FakeRuntime(Plan(batches, matched_tokens=cached))
                        )
                        try:
                            status, _, payload = harness.request(
                                "POST",
                                "/v1/chat/completions",
                                chat_body(
                                    stream=stream,
                                    stream_options={"include_usage": usage},
                                    reasoning_effort="none",
                                ),
                            )
                        finally:
                            harness.close()
                        self.assertEqual(status, 200, payload)
                        if stream:
                            rows = [
                                line[6:]
                                for line in payload.decode().splitlines()
                                if line.startswith("data: ")
                            ]
                            self.assertEqual(rows.pop(), "[DONE]")
                            chunks = [json.loads(row) for row in rows]
                            timed = [chunk for chunk in chunks if "timings" in chunk]
                            self.assertEqual(len(timed), 1)
                            self.assertEqual(
                                timed[0]["choices"][0]["finish_reason"], "stop"
                            )
                            self.assertEqual(
                                any("usage" in chunk for chunk in chunks), usage
                            )
                            timings = timed[0]["timings"]
                        else:
                            response = json.loads(payload)
                            timings = response["timings"]
                            self.assertEqual(
                                timings["predicted_n"],
                                response["usage"]["completion_tokens"],
                            )
                            self.assertEqual(
                                timings["prompt_n"], response["usage"]["prompt_tokens"]
                            )
                        count = sum(map(len, batches))
                        self.assertEqual(
                            timings,
                            {
                                "prompt_n": 2,
                                "prompt_ms": 1.0,
                                "prompt_per_second": (2 - cached) * 1000.0,
                                "predicted_n": count,
                                "predicted_ms": 2.0,
                                "predicted_per_second": (count - len(batches[0]))
                                * 500.0,
                                "cache_n": cached,
                            },
                        )

    def test_no_invalid_json_numbers_or_invented_rates(self):
        for interval in (0.0, -1.0, math.nan, math.inf, 5e-324):
            with self.subTest(interval=interval):
                result = NativeResult(
                    "stop",
                    20,
                    5,
                    interval,
                    interval,
                    10.0,
                    prefill_tokens=12,
                    cache=CacheInfo(matched_tokens=8),
                    first_token_batch_tokens=2,
                )
                timings = timings_dict(result)
                json.dumps(timings, allow_nan=False)
                self.assertEqual(timings["prompt_per_second"], 0.0)
                self.assertEqual(timings["predicted_per_second"], 0.0)
                self.assertGreaterEqual(timings["prompt_ms"], 0.0)
                self.assertGreaterEqual(timings["predicted_ms"], 0.0)
                self.assertEqual(timings["predicted_n"], 5)

    def test_unknown_first_emission_does_not_invent_throughput(self):
        result = NativeResult("stop", 20, 5, 1.0, 2.0, 10.0, prefill_tokens=20)
        self.assertEqual(timings_dict(result)["predicted_per_second"], 0.0)

    def test_tps_omits_prefill_token_and_invalid_timings(self):
        result = NativeResult(
            "stop", 1, 1, 1.0, 2.0, 3.0, 1, first_token_batch_tokens=1
        )
        metrics = metrics_dict(result)
        self.assertNotIn("stream_tokens_per_second", metrics["request_latency"])
        self.assertEqual(metrics["prefill"]["tokens"], 1)
        result.completion_tokens = 3
        for interval in (0.0, math.inf, 5e-324):
            result.first_token_to_done_ms = interval
            self.assertNotIn(
                "stream_tokens_per_second", metrics_dict(result)["request_latency"]
            )

    def test_stream_rate_excludes_the_whole_first_speculative_batch(self):
        for batches, remaining in (([[14, 15]], 0), ([[14, 15], [14, 15, 4]], 3)):
            with self.subTest(batches=batches):
                harness = self.harness(FakeRuntime(Plan(batches)))
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(reasoning_effort="none")
                )
                self.assertEqual(status, 200)
                metrics = json.loads(payload)["metrics"]
                self.assertEqual(metrics["decode"]["tokens"], remaining)
                latency = metrics["request_latency"]
                if remaining:
                    self.assertEqual(latency["stream_tokens_per_second"], 1500.0)
                else:
                    self.assertNotIn("stream_tokens_per_second", latency)


if __name__ == "__main__":
    unittest.main()
