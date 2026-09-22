import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from dev.tools import flash_expert_cache_plan as planner


class FlashExpertCachePlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def record(self, ids=None, *, rows=1, phase="decode"):
        ids = [3] * (rows * 10) if ids is None else ids
        return {
            "source_identity": planner.SOURCE_IDENTITY,
            "phase": phase,
            "rows": rows,
            "layer_ids": [list(ids) for _ in range(48)],
        }

    def capture(self, name, *records):
        path = self.root / name
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        return path

    def test_merges_captures_and_never_invents_unobserved_experts(self):
        first = self.capture("first.jsonl", self.record([3] * 6 + [9] * 4))
        second = self.capture("second.jsonl", self.record([9] * 7 + [11] * 3))
        originals = {path: path.read_bytes() for path in (first, second)}
        plan = planner.create_plan([first, second])
        self.assertEqual(plan["schema"], "splash-flash-hot-expert-plan-v1")
        self.assertEqual(plan["source_identity"], planner.SOURCE_IDENTITY)
        self.assertEqual(plan["selected_experts"], [[3, 9, 11] for _ in range(48)])
        self.assertEqual(len(plan["layer_stats"]), 48)
        for stats in plan["layer_stats"]:
            self.assertEqual(stats["observed_assignments"], 20)
            self.assertEqual(stats["selected_assignments"], 20)
            self.assertEqual(stats["observed_experts"], 3)
            self.assertEqual(stats["selected_count"], 3)
            self.assertEqual(stats["hit_rate"], 1)
        self.assertEqual(plan["aggregate"]["observed_assignments"], 20 * 48)
        self.assertEqual(plan["aggregate"]["selected_assignments"], 20 * 48)
        self.assertEqual(plan["aggregate"]["hit_rate"], 1)
        self.assertEqual(plan["scope"]["validated_records"], 2)
        self.assertEqual(plan["scope"]["included_records"], 2)
        self.assertEqual(plan["scope"]["excluded_records"], 0)
        self.assertEqual(plan["scope"]["included_rows"], 2)
        self.assertEqual(plan["scope"]["excluded_rows"], 0)
        self.assertIn("memory_estimate", plan)
        self.assertEqual(len(plan["captures"]), 2)
        metadata = {Path(item["path"]).resolve(): item for item in plan["captures"]}
        for path, raw in originals.items():
            self.assertEqual(path.read_bytes(), raw)
            capture = metadata[path.resolve()]
            self.assertEqual(capture["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(capture["bytes"], len(raw))
            self.assertEqual(capture["records"], 1)
            self.assertEqual(capture["included_records"], 1)
            self.assertEqual(capture["excluded_records"], 0)

    def test_frequency_wins_before_stable_id_ties_and_output_is_sorted(self):
        ids = [500] * 6 + list(reversed(range(34)))
        path = self.capture("frequencies.jsonl", self.record(ids, rows=4))
        plan = planner.create_plan([path], limit=32)
        self.assertEqual(
            plan["selected_experts"], [list(range(31)) + [500] for _ in range(48)]
        )
        for stats in plan["layer_stats"]:
            self.assertEqual(stats["observed_assignments"], 40)
            self.assertEqual(stats["selected_assignments"], 37)
            self.assertEqual(stats["observed_experts"], 35)
            self.assertEqual(stats["selected_count"], 32)
            self.assertAlmostEqual(stats["hit_rate"], 37 / 40)
        self.assertEqual(plan["aggregate"]["observed_assignments"], 40 * 48)
        self.assertEqual(plan["aggregate"]["selected_assignments"], 37 * 48)
        self.assertAlmostEqual(plan["aggregate"]["hit_rate"], 37 / 40)

    def test_supported_limits_cut_ties_by_id_at_the_boundary(self):
        for limit in (32, 64, 128, 256):
            with self.subTest(limit=limit):
                rows = math.ceil((limit + 1) / 10)
                ids = list(reversed(range(rows * 10)))
                path = self.capture(f"limit-{limit}.jsonl", self.record(ids, rows=rows))
                plan = planner.create_plan([path], limit=limit)
                self.assertEqual(
                    plan["selected_experts"], [list(range(limit)) for _ in range(48)]
                )
                for stats in plan["layer_stats"]:
                    self.assertEqual(stats["selected_count"], limit)
                    self.assertEqual(stats["selected_assignments"], limit)
                    self.assertAlmostEqual(stats["hit_rate"], limit / (rows * 10))

    def test_layer_counts_are_independent(self):
        record = self.record()
        record["layer_ids"] = [[layer] * 9 + [layer + 100] for layer in range(48)]
        path = self.capture("layers.jsonl", record)
        plan = planner.create_plan([path], limit=32)
        self.assertEqual(
            plan["selected_experts"], [[layer, layer + 100] for layer in range(48)]
        )
        for stats in plan["layer_stats"]:
            self.assertEqual(stats["observed_assignments"], 10)
            self.assertEqual(stats["observed_experts"], 2)

    def test_phase_filter_validates_all_records_and_reports_excluded_work(self):
        path = self.capture(
            "phases.jsonl",
            self.record([400] * 20, rows=2, phase="prefill"),
            self.record([5] * 10, phase="decode"),
        )
        plan = planner.create_plan([path], phase="decode")
        self.assertEqual(plan["selected_experts"], [[5] for _ in range(48)])
        expected_scope = {
            "phase_filter": "decode",
            "validated_records": 2,
            "included_records": 1,
            "excluded_records": 1,
            "included_rows": 1,
            "excluded_rows": 2,
        }
        for field, value in expected_scope.items():
            self.assertEqual(plan["scope"][field], value)
        self.assertEqual(plan["captures"][0]["records"], 2)
        self.assertEqual(plan["captures"][0]["included_records"], 1)
        self.assertEqual(plan["captures"][0]["excluded_records"], 1)
        self.assertEqual(plan["aggregate"]["observed_assignments"], 10 * 48)
        self.assertEqual(plan["aggregate"]["selected_assignments"], 10 * 48)
        self.assertEqual(plan["aggregate"]["hit_rate"], 1)

    def test_invalid_records_cannot_hide_in_an_excluded_phase(self):
        for failure in ("source", "rows", "ids"):
            with self.subTest(failure=failure):
                excluded = self.record(phase="prefill")
                if failure == "source":
                    excluded["source_identity"] = "b" * 64
                elif failure == "rows":
                    excluded["rows"] = True
                else:
                    excluded["layer_ids"][0][0] = 512
                path = self.capture(
                    f"excluded-{failure}.jsonl", excluded, self.record()
                )
                with self.assertRaises(planner.PlanError):
                    planner.create_plan([path], phase="decode")

    def test_wrong_source_is_rejected_and_explicit_source_must_match_records(self):
        wrong = self.record()
        wrong["source_identity"] = "b" * 64
        path = self.capture("wrong-source.jsonl", wrong)
        with self.assertRaises(planner.PlanError):
            planner.create_plan([path])
        plan = planner.create_plan([path], source_identity="b" * 64)
        self.assertEqual(plan["source_identity"], "b" * 64)

    def test_invalid_expert_ids_are_rejected_without_coercion(self):
        for value in (-1, 512, True, False, 3.0, "3", None, float("nan"), float("inf")):
            with self.subTest(value=value):
                record = self.record()
                record["layer_ids"][17][4] = value
                path = self.capture("invalid-id.jsonl", record)
                with self.assertRaises(planner.PlanError):
                    planner.create_plan([path])

    def test_incomplete_layer_or_assignment_captures_are_rejected(self):
        malformed = []
        missing_layer = self.record()
        missing_layer["layer_ids"].pop()
        malformed.append(missing_layer)
        extra_layer = self.record()
        extra_layer["layer_ids"].append([3] * 10)
        malformed.append(extra_layer)
        for size in (0, 9, 11):
            record = self.record()
            record["layer_ids"][20] = [3] * size
            malformed.append(record)
        all_empty = self.record()
        all_empty["layer_ids"] = [[] for _ in range(48)]
        malformed.append(all_empty)
        nested = self.record()
        nested["layer_ids"][0] = [[3] * 10]
        malformed.append(nested)
        for index, record in enumerate(malformed):
            with self.subTest(index=index):
                path = self.capture("incomplete.jsonl", record)
                with self.assertRaises(planner.PlanError):
                    planner.create_plan([path])

    def test_missing_fields_invalid_phases_and_rows_are_rejected(self):
        records = [{**self.record(), "unexpected": "field"}]
        for field in ("source_identity", "phase", "rows", "layer_ids"):
            record = self.record()
            del record[field]
            records.append(record)
        for value in (0, 2049, True, 1.0, "1", None, float("nan")):
            records.append({**self.record(), "rows": value})
        for value in ("all", "Prefill", "", None, True):
            records.append({**self.record(), "phase": value})
        for index, record in enumerate(records):
            with self.subTest(index=index):
                path = self.capture("invalid-record.jsonl", record)
                with self.assertRaises(planner.PlanError):
                    planner.create_plan([path])

    def test_empty_inputs_and_empty_phase_selections_are_rejected(self):
        empty = self.root / "empty.jsonl"
        empty.write_text("")
        prefill = self.capture("prefill-only.jsonl", self.record(phase="prefill"))
        for paths, options in (
            ([], {}),
            ([empty], {}),
            ([prefill], {"phase": "decode"}),
        ):
            with (
                self.subTest(paths=paths, options=options),
                self.assertRaises(planner.PlanError),
            ):
                planner.create_plan(paths, **options)

    def test_invalid_options_and_malformed_json_raise_plan_error(self):
        self.assertTrue(issubclass(planner.PlanError, ValueError))
        valid = self.capture("valid.jsonl", self.record())
        for limit in (0, 16, 33, 512, True, 32.0, "32"):
            with self.subTest(limit=limit), self.assertRaises(planner.PlanError):
                planner.create_plan([valid], limit=limit)
        with self.assertRaises(planner.PlanError):
            planner.create_plan([valid], phase="invalid")
        malformed = self.root / "malformed.jsonl"
        for raw in ("{\n", "[]\n", "null\n", "42\n"):
            with self.subTest(raw=raw):
                malformed.write_text(raw)
                with self.assertRaises(planner.PlanError):
                    planner.create_plan([malformed])


if __name__ == "__main__":
    unittest.main()
