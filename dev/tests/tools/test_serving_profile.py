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

    def test_ragged_geometry_and_startup_seed_are_retained(self):
        report = trace_report.summarize(
            {
                "traceEvents": [
                    event("prefill.seed", 0, 0, rows=2048, value=256),
                    event("prefill.lane", 1, 0, id=9, flags=1, rows=64, context=8192),
                    event("prefill.lane", 1, 0, id=9, flags=0, rows=128, context=32),
                    event(
                        "batch.observation",
                        2,
                        0,
                        id=9,
                        flags=3,
                        rows=192,
                        width=2,
                        context=8192,
                        value=50,
                    ),
                ]
            }
        )
        self.assertEqual(report["prefill_seeds"], [{"rows": 2048, "wall_ms": 256}])
        self.assertEqual(
            report["prefill_observations"][0]["lanes"],
            [
                {"lane": 0, "rows": 128, "history": 32},
                {"lane": 1, "rows": 64, "history": 8192},
            ],
        )

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
    def test_trace_option_refuses_a_binary_that_would_ignore_the_trace_environment(
        self,
    ):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "splash"
            binary.write_bytes(b"production")
            with self.assertRaisesRegex(
                serving_regression.smoke.SmokeFailure, "does not support"
            ):
                serving_regression.require_trace_support([binary])
            binary.write_bytes(b"SPLASH_SERVING_TRACE")
            serving_regression.require_trace_support([binary])

    def test_verification_controls_are_sent_per_arm_and_confirmed_by_status(self):
        import argparse
        from pathlib import Path

        args = argparse.Namespace(
            verify_rows=None, baseline_verify_rows="all", candidate_verify_rows="auto"
        )
        self.assertEqual(
            serving_regression.server_settings(args, Path("base"), 0).verify_rows, "all"
        )
        self.assertEqual(
            serving_regression.server_settings(args, Path("candidate"), 1).verify_rows,
            "auto",
        )
        serving_regression.require_verification_policy({"verify_rows": "all"}, "all")
        serving_regression.require_verification_policy({"verify_rows": "auto"}, "auto")
        serving_regression.require_verification_policy(
            {"verify_rows": "all"}, "auto"
        )  # Dense fallback.
        for value in ({}, {"verify_rows": "auto"}):
            with self.assertRaisesRegex(
                serving_regression.smoke.SmokeFailure, "verification policy"
            ):
                serving_regression.require_verification_policy(value, "all")

    def test_suspend_is_rejected_even_when_abba_costs_could_be_stable(self):
        report = {
            "environment": {"unix_seconds": 100, "awake_seconds": 20},
            "final_environment": {"unix_seconds": 110, "awake_seconds": 30},
        }
        serving_regression.validate_awake_period(report)
        report["final_environment"]["unix_seconds"] += 600
        with self.assertRaisesRegex(serving_regression.smoke.SmokeFailure, "suspend"):
            serving_regression.validate_awake_period(report)
        self.assertEqual(report["unaccounted_wall_seconds"], 600)

    def test_sleep_assertion_is_released_on_failure(self):
        from unittest.mock import Mock

        keeper = Mock()
        with (
            patch.object(serving_regression.sys, "platform", "darwin"),
            patch.object(serving_regression.subprocess, "Popen", return_value=keeper),
        ):
            with self.assertRaisesRegex(RuntimeError, "workload failed"):
                with serving_regression.keep_awake():
                    raise RuntimeError("workload failed")
        keeper.terminate.assert_called_once()
        keeper.wait.assert_called_once_with(timeout=5)

    def test_mixed_stream_refuses_a_short_output_budget(self):
        import argparse
        import json
        from unittest.mock import Mock

        args = argparse.Namespace(model="test/model", request_timeout=1)
        for count in (31, 32):
            usage = {"prompt_tokens": 2048, "completion_tokens": count}
            frames = [
                {"choices": [{"delta": {"content": "answer"}}]},
                {"choices": [], "usage": usage},
            ]
            connection = Mock()
            connection.getresponse.return_value.status = 200
            connection.getresponse.return_value.readline.side_effect = [
                *(b"data: " + json.dumps(frame).encode() + b"\n" for frame in frames),
                b"data: [DONE]\n",
            ]
            with patch.object(
                serving_regression.http.client,
                "HTTPConnection",
                return_value=connection,
            ):
                if count == 31:
                    with self.assertRaisesRegex(
                        serving_regression.smoke.SmokeFailure, "output budget"
                    ):
                        serving_regression.streamed(Mock(port=1234), args, "prompt", 32)
                else:
                    result = serving_regression.streamed(
                        Mock(port=1234), args, "prompt", 32
                    )
                    self.assertEqual(result["usage"], usage)
            connection.close.assert_called_once()

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

    def test_native_overlap_rejects_serial_admission_and_cache_restore_only(self):
        active = {"ended": 10}
        long = {"ended": 20, "usage": {"prompt_tokens": 8192}}
        decoder = {
            "id": 1,
            "phase": "decode",
            "max_new_tokens": 512,
            "prompt_tokens": 2048,
        }
        peer = {
            "id": 2,
            "phase": "prefill",
            "max_new_tokens": 32,
            "prompt_tokens": 8192,
            "prompt_processed": 0,
        }
        sample = {
            "at": 4,
            "before_short": False,
            "prefill_rows": 2048,
            "requests": [decoder, peer],
        }
        self.assertIsNone(
            serving_regression.native_overlap([sample], 1, 8192, 0, long, active, 2048)
        )
        peer["prompt_processed"] = 4096
        sample["prefill_rows"] = 6144
        self.assertEqual(
            serving_regression.native_overlap([sample], 1, 8192, 0, long, active, 2048),
            "long-prefill",
        )
        self.assertIsNone(
            serving_regression.native_overlap(
                [sample], 1, 8192, 4096, long, active, 2048
            )
        )
        sample["at"] = 12
        self.assertIsNone(
            serving_regression.native_overlap([sample], 1, 8192, 0, long, active, 2048)
        )

    def test_stale_poll_cannot_supply_overlap_evidence(self):
        active = {"ended": 10}
        long = {"ended": 20, "usage": {"prompt_tokens": 8192}}
        sample = {
            "at": 4,
            "before_short": True,
            "status_stale": True,
            "prefill_rows": 6144,
            "requests": [
                {"id": 1, "phase": "decode"},
                {
                    "id": 2,
                    "phase": "prefill",
                    "prompt_tokens": 8192,
                    "max_new_tokens": 32,
                    "prompt_processed": 4096,
                },
            ],
        }
        self.assertIsNone(
            serving_regression.native_overlap([sample], 1, 8192, 0, long, active, 2048)
        )
        fresh = {**sample, "status_stale": False}
        self.assertEqual(
            serving_regression.native_overlap(
                [sample, fresh], 1, 8192, 0, long, active, 2048
            ),
            "long-prefill",
        )
        fresh["requests"] = []
        self.assertIsNone(
            serving_regression.native_overlap(
                [sample, fresh], 1, 8192, 0, long, active, 2048
            )
        )

    def test_transient_stale_poll_is_recorded_but_initial_snapshot_must_be_fresh(self):
        snapshot = {
            "transport": {"status_stale": True},
            "scheduler": {"prefill_rows": 12},
            "memory_pressure": "normal",
            "active_requests": [],
        }
        with patch.object(serving_regression, "status", return_value=snapshot):
            with self.assertRaises(serving_regression.smoke.SmokeFailure):
                serving_regression.native_sample(None, None)
            value = serving_regression.native_sample(None, None, allow_stale=True)
        self.assertTrue(value["status_stale"])
        self.assertEqual(value["prefill_rows"], 12)

    def test_sequential_residency_closes_each_server_before_starting_the_next(self):
        import argparse
        import io
        import tempfile
        from pathlib import Path

        live, order = [], []

        class Server:
            def __init__(self, settings):
                self.side = str(settings.binary)
                if live:
                    raise AssertionError("two servers were resident")
                live.append(self)
                order.append(self.side)
                self.log = io.StringIO("server log")

            def wait_ready(self, timeout):
                return {}

            def stop(self):
                if self in live:
                    live.remove(self)

            def close(self):
                self.stop()
                self.log.close()

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                output=Path(directory),
                trace=False,
                probes=False,
                environments=[{}, {}],
                startup_timeout=1,
                batch=1,
            )
            report = {
                "initial_status": [],
                "complete": False,
                "environment": {"unix_seconds": 100, "awake_seconds": 20},
            }
            case = {
                "context": 2048,
                "workload": "code",
                "sample": 0,
                "prompts": ["p"],
                "rounds": [],
            }
            with (
                patch.object(
                    serving_regression, "case_specs", return_value=iter([case])
                ),
                patch.object(serving_regression.smoke, "RealServer", Server),
                patch.object(serving_regression, "warm_servers", return_value={}),
                patch.object(serving_regression, "status", return_value={}),
                patch.object(
                    serving_regression,
                    "environment",
                    return_value={"unix_seconds": 100, "awake_seconds": 20},
                ),
                patch.object(
                    serving_regression,
                    "measure_case",
                    side_effect=lambda *a: {"seconds_per_token": 1, "responses": []},
                ),
            ):
                serving_regression.run_sequential(
                    args, None, report, [Path("base"), Path("candidate")]
                )
            self.assertTrue(report["complete"])
            self.assertEqual(
                [row["arm"] for row in report["cases"][0]["rounds"]], [1, 2, 3, 4]
            )
        self.assertEqual(order, ["base", "candidate", "candidate", "base"])
        self.assertFalse(live)

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
