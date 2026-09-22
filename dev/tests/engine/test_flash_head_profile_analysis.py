"""Tiny CPU-only oracles for trained-head trace aggregation."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from dev.tools import analyze_flash_head_profile as analysis


def trace(
    case="proposal-ctx128",
    *,
    phase="proposal",
    context=128,
    rows=1,
    mode="stage",
    status="complete",
):
    return {
        "case": case,
        "phase": phase,
        "context_per_lane": context,
        "true_rows_per_lane": rows,
        "physical_rows": rows,
        "lanes": 1,
        "command": {
            "mode": mode,
            "status": status,
            "reason": "",
            "gpu_seconds": 0.010,
            "wall_seconds": 0.020,
            "dispatch_count": 3,
            "dispatches_total": 3,
            "dispatches_emitted": 3,
            "encoder_boundaries_altered": mode == "stage",
            "sampling_barriers": mode == "dispatch",
            "dropped_profiles_before": 0,
            "dispatches": [
                {"pipeline": "zero", "timestamps_valid": True, "gpu_seconds": 0},
                {"pipeline": "qsa", "timestamps_valid": True, "gpu_seconds": 0.004},
                {"pipeline": "qsa", "timestamps_valid": True, "gpu_seconds": 0.002},
            ],
        },
        "family_attribution": {
            "classification_complete": True,
            "full_command_gpu_seconds": 0.010,
            "sum_timed_dispatch_gpu_seconds": 0.006,
            "command_minus_timed_dispatch_seconds": 0.004,
            "families": {
                "normalization": {
                    "dispatches": 1,
                    "timed_dispatches": 1,
                    "gpu_seconds": 0,
                },
                "attention": {
                    "dispatches": 2,
                    "timed_dispatches": 2,
                    "gpu_seconds": 0.006,
                },
            },
        },
    }


class HeadProfileAnalysisTests(unittest.TestCase):
    def test_sums_fractions_and_signed_remainder_are_independent_of_command_time(self):
        first, second = trace(), trace()
        second["command"]["gpu_seconds"] = 0.005
        second["family_attribution"]["full_command_gpu_seconds"] = 0.005
        second["family_attribution"]["command_minus_timed_dispatch_seconds"] = -0.001
        report = analysis.summarize_records([first, second])
        overall = report["overall"]
        self.assertTrue(report["valid"])
        self.assertTrue(overall["attribution_complete"])
        self.assertAlmostEqual(overall["command_gpu"]["sum_ms"], 15)
        self.assertAlmostEqual(overall["sum_emitted_timed_dispatch_gpu"]["sum_ms"], 12)
        self.assertAlmostEqual(
            overall["command_minus_emitted_timed_dispatch_gpu"]["sum_ms"], 3
        )
        self.assertAlmostEqual(
            overall["command_minus_emitted_timed_dispatch_gpu"]["minimum_ms"], -1
        )
        self.assertEqual(overall["pipelines"][0]["pipeline"], "qsa")
        self.assertAlmostEqual(overall["pipelines"][0]["share_of_timed_gpu"], 1)
        zero = next(
            entry for entry in overall["pipelines"] if entry["pipeline"] == "zero"
        )
        self.assertEqual((zero["timed_dispatches"], zero["gpu_ms"]), (2, 0))

    def test_untimed_dispatch_is_unknown_not_zero_and_changes_coverage(self):
        row = trace()
        row["command"]["dispatches"][0].update(timestamps_valid=False, gpu_seconds=None)
        row["family_attribution"]["families"]["normalization"]["timed_dispatches"] = 0
        report = analysis.summarize_records([row])
        self.assertTrue(report["valid"])
        overall = report["overall"]
        self.assertFalse(overall["attribution_complete"])
        self.assertEqual(overall["untimed_emitted_dispatches"], 1)
        self.assertAlmostEqual(overall["timed_dispatch_coverage"], 2 / 3)
        self.assertIsNone(
            next(
                entry for entry in overall["pipelines"] if entry["pipeline"] == "zero"
            )["gpu_ms"]
        )
        self.assertIsNone(
            next(
                entry
                for entry in overall["families"]
                if entry["family"] == "normalization"
            )["gpu_ms"]
        )

    def test_unsupported_and_command_only_capture_keep_command_time_but_no_kernel_time(
        self,
    ):
        unsupported, command_only = trace(status="unsupported"), trace(mode="command")
        unsupported["command"]["reason"] = "timestamp counters unavailable"
        for row in (unsupported, command_only):
            for dispatch in row["command"]["dispatches"]:
                dispatch.update(timestamps_valid=False, gpu_seconds=None)
            for family in row["family_attribution"]["families"].values():
                family.update(timed_dispatches=0, gpu_seconds=0)
            row["family_attribution"].update(
                sum_timed_dispatch_gpu_seconds=0,
                command_minus_timed_dispatch_seconds=0.010,
            )
        report = analysis.summarize_records([unsupported, command_only])
        self.assertTrue(report["valid"])
        overall = report["overall"]
        self.assertFalse(overall["attribution_complete"])
        self.assertEqual(
            overall["command_status_counts"], {"complete": 1, "unsupported": 1}
        )
        self.assertAlmostEqual(overall["command_gpu"]["sum_ms"], 20)
        self.assertIsNone(overall["sum_emitted_timed_dispatch_gpu"]["sum_ms"])
        self.assertIsNone(overall["command_minus_emitted_timed_dispatch_gpu"]["sum_ms"])
        self.assertEqual(overall["timed_dispatches"], 0)

    def test_groups_keep_cases_modes_phases_and_geometry_separate(self):
        rows = [
            trace("sample-0"),
            trace("sample-1"),
            trace("sample-1", context=2048),
            trace("sample-0", mode="dispatch"),
            trace("fold", phase="committed_fold", rows=8),
            trace("joint", phase="joint_proposal", rows=4),
        ]
        report = analysis.summarize_records(rows)
        self.assertEqual(len(report["by_case"]), 6)
        self.assertEqual(len(report["by_context_rows"]), 5)
        short = next(
            group
            for group in report["by_context_rows"]
            if group["context_per_lane"] == 128
            and group["phase"] == "proposal"
            and group["mode"] == "stage"
        )
        self.assertEqual(short["commands"], 2)
        self.assertEqual(short["cases"], ["sample-0", "sample-1"])
        self.assertEqual(
            report["overall"]["perturbation"]["encoder_boundaries_altered_commands"], 5
        )
        self.assertEqual(
            report["overall"]["perturbation"]["sampling_barrier_commands"], 1
        )

    def test_truncated_serialization_does_not_compare_full_family_sum_to_partial_pipelines(
        self,
    ):
        row = trace()
        row["command"]["dispatches"] = row["command"]["dispatches"][:2]
        row["command"].update(dispatches_emitted=2, dispatches_truncated=True)
        report = analysis.summarize_records([row])
        self.assertTrue(report["valid"])
        self.assertFalse(report["overall"]["attribution_complete"])
        self.assertEqual(report["overall"]["dispatches_not_emitted"], 1)
        self.assertAlmostEqual(report["overall"]["pipelines"][0]["gpu_ms"], 4)
        self.assertAlmostEqual(report["overall"]["families"][0]["gpu_ms"], 6)

    def test_truncated_capture_still_validates_family_totals_and_internal_sums(self):
        row = trace()
        row["command"]["dispatches"] = row["command"]["dispatches"][:2]
        row["command"].update(dispatches_emitted=2, dispatches_truncated=True)
        row["family_attribution"]["families"]["attention"].update(
            dispatches=999, timed_dispatches=999, gpu_seconds=999
        )
        row["family_attribution"]["timed_dispatches"] = 3
        report = analysis.summarize_records([row])
        self.assertFalse(report["valid"])
        messages = " ".join(report["validation_errors"])
        self.assertIn("family dispatch counts exceed", messages)
        self.assertIn("attribution timed_dispatches", messages)
        self.assertIn("sum_timed_dispatch_gpu_seconds", messages)

    def test_unknown_enums_and_unrepresentable_numbers_are_invalid_input(self):
        for kind in ("mode", "status"):
            row = trace()
            row["command"][kind] = "bogus"
            report = analysis.summarize_records([row])
            self.assertFalse(report["valid"])
        row = trace()
        row["command"]["gpu_seconds"] = 10**1000
        report = analysis.summarize_records([row])
        self.assertFalse(report["valid"])
        json.dumps(report, allow_nan=False)

    def test_inconsistent_attribution_and_nonfinite_times_are_validation_errors(self):
        row = trace()
        row["family_attribution"]["families"]["attention"]["gpu_seconds"] = 0.009
        row["family_attribution"]["command_minus_timed_dispatch_seconds"] = 0.007
        row["command"]["dispatches"][0]["gpu_seconds"] = float("nan")
        report = analysis.summarize_records([row])
        self.assertFalse(report["valid"])
        self.assertFalse(report["overall"]["attribution_complete"])
        self.assertTrue(
            any("family GPU sum" in message for message in report["validation_errors"])
        )
        self.assertTrue(
            any(
                "command_minus_timed_dispatch_seconds" in message
                for message in report["validation_errors"]
            )
        )
        json.dumps(report, allow_nan=False)

    def test_jsonl_reader_and_case_object_preserve_missing_geometry(self):
        row = trace()
        row["case"] = {"name": "nested", "context_rows": 2048, "rows": 8}
        for key in ("context_per_lane", "true_rows_per_lane", "physical_rows", "lanes"):
            del row[key]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.trace.jsonl"
            path.write_text("\n" + json.dumps(row) + "\n")
            report = analysis.analyze(path)
        group = report["by_case"][0]
        self.assertEqual(
            (group["case"], group["context_per_lane"], group["true_rows_per_lane"]),
            ("nested", 2048, 8),
        )
        self.assertIsNone(group["physical_rows"])
        self.assertIsNone(group["lanes"])

    def test_cli_writes_complete_json_and_discloses_perturbed_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = (
                Path(directory) / "report.trace.jsonl",
                Path(directory) / "summary.json",
            )
            source.write_text(json.dumps(trace()) + "\n")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = analysis.main(
                    [str(source), "--output", str(output), "--top", "1"]
                )
            summary = json.loads(output.read_text())
        self.assertEqual(status, 0)
        self.assertTrue(summary["valid"])
        self.assertEqual(len(summary["reports"][0]["overall"]["pipelines"]), 2)
        self.assertIn("signed command-minus-counter sum=4.000 ms", stdout.getvalue())
        self.assertIn("Stage mode inserts an encoder boundary", stdout.getvalue())
        self.assertIn("timestamp sampling barriers", stdout.getvalue())

    def test_invalid_row_does_not_leave_overall_attribution_complete(self):
        report = analysis.summarize_records([trace(), {"case": "bad", "phase": []}])
        self.assertFalse(report["valid"])
        self.assertFalse(report["overall"]["attribution_complete"])
        self.assertEqual(report["trace_rows"], 2)
        self.assertEqual(report["overall"]["commands"], 1)

    def test_jsonl_semantic_errors_use_physical_line_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.trace.jsonl"
            path.write_text("\n" + json.dumps({"case": "bad", "phase": "bogus"}) + "\n")
            report = analysis.analyze(path)
        self.assertIn(f"{path.resolve()}:2:", report["validation_errors"][0])


if __name__ == "__main__":
    unittest.main()
