"""Validity boundaries of the serving trace/paired benchmark reports."""

import math
import os
import unittest
from unittest.mock import patch

from dev.benchmarks import serving_regression, trace_report


def event(name, start, duration, **values):
    return {"name": name, "ph": "X", "ts": start, "dur": duration, "args": values}


class TraceReportTests(unittest.TestCase):
    def test_idle_time_is_not_a_busy_gpu_gap(self):
        report = trace_report.summarize(
            {
                "traceEvents": [
                    event("request.active", 0, 10),
                    event("metal.command_span", 1, 1),
                    event("metal.command_span", 3, 1),
                    event("request.active", 100, 10),
                    event("metal.command_span", 101, 1),
                ]
            }
        )
        self.assertEqual(report["gpu_inter_command_gaps"]["count"], 2)
        self.assertEqual(report["gpu_gaps_with_a_request_active"]["count"], 1)
        self.assertEqual(report["gpu_gaps_with_a_request_active"]["median_ms"], 0.001)

    def test_overlapping_requests_keep_the_whole_gap_active(self):
        report = trace_report.summarize(
            {
                "traceEvents": [
                    event("request.active", 0, 3),
                    event("request.active", 2, 3),
                    event("metal.command_span", 1, 0.5),
                    event("metal.command_span", 4, 0.5),
                ]
            }
        )
        self.assertEqual(report["gpu_gaps_with_a_request_active"]["count"], 1)
        self.assertEqual(report["gpu_gaps_with_a_request_active"]["median_ms"], 0.0025)

    def test_partial_overflowing_and_invalid_traces_are_rejected(self):
        for values in (
            {"dropped_host_events": 1},
            {"dropped_commands": 1},
            {"command_in_flight_at_stop": True},
            {"exit_status": 70},
            {"traceEvents": [event("host", -1, 1)]},
            {"traceEvents": [event("host", 0, -1)]},
            {"traceEvents": [event("host", math.nan, 1)]},
            {
                "traceEvents": [
                    event("metal.command_span", 0, 5),
                    event("metal.command_span", 2, 5),
                ]
            },
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                trace_report.summarize({"traceEvents": [], **values})

    def test_prefill_observations_keep_the_representative_flag(self):
        report = trace_report.summarize(
            {
                "traceEvents": [
                    event(
                        "batch.observation",
                        1,
                        0,
                        id=1,
                        flags=3,
                        rows=2048,
                        width=1,
                        context=8192,
                        value=40,
                    ),
                    event(
                        "batch.observation",
                        2,
                        0,
                        id=2,
                        flags=1,
                        rows=128,
                        width=2,
                        context=32,
                        value=5,
                    ),
                    event(
                        "batch.observation",
                        3,
                        0,
                        id=3,
                        flags=2,
                        rows=8,
                        width=1,
                        context=8192,
                        value=2,
                    ),
                ]
            }
        )
        rows = report["prefill_observations"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["representative"])
        self.assertFalse(rows[1]["representative"])
        self.assertEqual(rows[0]["maximum_history"], 8192)
        self.assertEqual(rows[0]["wall_ms"], 40)


class PairedServingTests(unittest.TestCase):
    def test_warmup_uses_equal_matched_rounds_until_both_arms_are_ready(self):
        import argparse

        args = argparse.Namespace(
            warmup_context=2048, warmup_tokens=512, warmup_seconds=3
        )
        with (
            patch.object(
                serving_regression, "prepared_prompt", side_effect=["a", "b", "c"]
            ),
            patch.object(serving_regression, "completion") as complete,
            patch.object(
                serving_regression.time,
                "monotonic",
                side_effect=[0, 3, 3, 4, 4, 7, 7, 8, 8, 11, 11, 12],
            ),
        ):
            result = serving_regression.warm_servers(["base", "candidate"], args, None)
        self.assertEqual(
            result, {"rounds_per_arm": 3, "request_wall_seconds_by_arm": [9, 3]}
        )
        self.assertEqual(
            [(call.args[0], call.args[2]) for call in complete.call_args_list],
            [
                ("base", "a"),
                ("candidate", "a"),
                ("base", "b"),
                ("candidate", "b"),
                ("base", "c"),
                ("candidate", "c"),
            ],
        )

    def test_cached_and_prefix_resume_primers_share_only_the_intended_prefix(self):
        import argparse

        for mode, expected in (
            ("cold", None),
            ("cached", "abcdefgh"),
            ("prefix-resume", "abcd"),
        ):
            with (
                self.subTest(mode=mode),
                patch.object(serving_regression, "completion") as complete,
            ):
                complete.return_value = {
                    "usage": {"prompt_tokens_details": {"cached_tokens": 0}}
                }
                salt = serving_regression.prepare_mixed_cache(
                    "server", argparse.Namespace(cache_mode=mode), "abcdefgh"
                )
                if expected is None:
                    self.assertIsNone(salt)
                    complete.assert_not_called()
                else:
                    self.assertEqual(complete.call_args.args[2], expected)
                    self.assertEqual(complete.call_args.kwargs["cache_salt"], salt)
                    self.assertEqual(complete.call_args.kwargs["tokens"], 1)

    def test_noisy_speedup_and_changed_outputs_do_not_disappear(self):
        rows = [
            {"seconds_per_token": value, "responses": [{"message_sha256": output}]}
            for value, output in [(1, "a"), (0.8, "a"), (0.8, "b"), (1.2, "a")]
        ]
        report = serving_regression.summarize(rows)
        self.assertEqual(report["verdict"], "inconclusive")
        self.assertFalse(report["output_equal"])

    def test_child_environment_is_restored_after_failure(self):
        with patch.dict(os.environ, {"TRACE_TEST_EXISTING": "original"}):
            os.environ.pop("TRACE_TEST_NEW", None)
            with self.assertRaises(RuntimeError):
                with serving_regression.child_environment(
                    {"TRACE_TEST_EXISTING": "temporary", "TRACE_TEST_NEW": "yes"}
                ):
                    self.assertEqual(os.environ["TRACE_TEST_NEW"], "yes")
                    raise RuntimeError("startup failed")
            self.assertEqual(os.environ["TRACE_TEST_EXISTING"], "original")
            self.assertNotIn("TRACE_TEST_NEW", os.environ)


if __name__ == "__main__":
    unittest.main()
