"""Mutation tests for publication gates; no model, GPU, or recorded timings."""

import json
import unittest

from dev.benchmarks import experiment_common as common
from dev.benchmarks import lookup_regression as lookup
from dev.benchmarks import lookup_task_score as score


def cohort_document(cohort="complete"):
    config = lookup.COHORTS[cohort]
    names, edits = config["names"], config["names"][: config["edits"]]
    inputs = {
        r["name"]: r["input_tokens"]
        for r in lookup.fixture_document(cohort)["fixtures"]
    }
    pins = json.loads((lookup.FIXTURES / "model-pins.json").read_text())
    doc = {
        "schema_version": 2,
        "experimental_prompt_lookup": False,
        "lookup_output_tokens": 256,
        "lookup_edit_output_tokens": 1024,
        "lookup_control_output_tokens": 256,
        "lookup_fixture_version": config["version"],
        "lookup_fixture_cohort": config["native"],
        "lookup_fixture_sha256": config["sha256"],
        "prompt_lookup_match_tokens": 16,
        "prompt_lookup_search_tokens": 65536,
        "prompt_lookup_proposal_tokens": 7,
        "lookup_tokenizer_sha256": lookup.TOKENIZER_SHA256,
        "lookup_template_sha256": lookup.TEMPLATE_SHA256,
        "lookup_runtime_instrumented": False,
        "lookup_logits_capture_enabled": False,
        "lookup_compilation_profile": "production_O3_flags",
        "lookup_abba_enabled": True,
        "lookup_abba_groups": 3,
        "lookup_abba_order": list(lookup.ORDER),
        "lookup_thermal_warmup_requests": 8 * len(names),
        "lookup_thermal_warmup_in_primary": False,
        "ane_ffn_share": 0,
        "ane_ffn_bytes": 0,
        "max_context_tokens": 16384,
        "geometry": lookup.GEOMETRY,
        "skipped_context_lengths": [],
        "performance_pass": True,
        "performance_failures": [],
        "no_ane_context_tokens": 16384,
        "build_id": "synthetic-test",
        "identity": {
            "device": "synthetic-test",
            "loaded_model_layout_sha256": pins["loaded_model_layout_sha256"],
        },
    }

    def row(name, sample, enabled):
        result = dict.fromkeys(lookup.COUNTERS, 0)
        result.update(dict.fromkeys(lookup.TIMINGS, 1.0))
        result.update(
            scenario=name,
            sample=sample,
            input_tokens=list(inputs[name]),
            input_token_hash=common.token_hash(inputs[name]),
            prompt_tokens=len(inputs[name]),
            output_tokens=[100] * 11 + [248046],
            requested_output_tokens=1024 if name in edits else 256,
            actual_output_tokens=12,
            cache_status="miss",
            target_prefill_rows=len(inputs[name]),
            wall_ms=90.0 if enabled and name in edits else 100.0,
            ttft_ms=10.0,
            prefill_gpu_ms=5.0,
            decode_gpu_ms=60.0,
            decode_wall_ms=70.0,
            decode_batches=2,
            decode_output_tokens=11,
            drafted_tokens=14,
            accepted_draft_tokens=10,
            prompt_lookup_search_wall_ms=0.1 if enabled else 0.0,
        )
        if enabled:
            result.update(
                prompt_lookup_searches=2,
                prompt_lookup_hits=1,
                prompt_lookup_bypassed_cycles=1,
                prompt_lookup_proposed_tokens=7,
                prompt_lookup_accepted_tokens=6,
            )
        return result

    doc["lookup_thermal_warmup_measurements"] = [
        row(name, sample, False) for sample in range(8) for name in names
    ]
    doc["measurements"] = []
    for group in range(1, 4):
        for name in names:
            for slot, variant in enumerate(lookup.ORDER, 1):
                item = row(name, group - 1, variant == "on")
                item.update(
                    lookup_abba_group=group,
                    lookup_abba_slot=slot,
                    lookup_abba_variant=variant,
                )
                doc["measurements"].append(item)
    return doc


class LookupProtocolTests(unittest.TestCase):
    def test_both_fixed_cohorts_and_actual_decode_counts(self):
        for cohort, count in (("complete", 6), ("fresh", 9)):
            report = lookup.analyze(cohort_document(cohort), cohort)
            self.assertTrue(report["primary_decision"]["qualified"])
            self.assertEqual(report["retained_warmup_requests"], 8 * count)
            self.assertEqual(report["retained_measured_requests"], 12 * count)
            self.assertEqual(
                report["edit_decode_tokens_per_second_off_on"], [1000 * 11 / 70] * 2
            )

    def test_raw_semantic_corruption_fails_closed(self):
        mutations = {
            "missing measured row": lambda d: d["measurements"].pop(),
            "missing warmup": lambda d: d["lookup_thermal_warmup_measurements"].pop(),
            "wrong slot": lambda d: d["measurements"][1].update(lookup_abba_slot=3),
            "bool slot": lambda d: d["measurements"][0].update(lookup_abba_slot=True),
            "wrong input": lambda d: d["measurements"][0]["input_tokens"].__setitem__(
                0, 999
            ),
            "different output": lambda d: d["measurements"][1][
                "output_tokens"
            ].__setitem__(0, 101),
            "invalid EOS": lambda d: d["measurements"][1]["output_tokens"].__setitem__(
                -1, 100
            ),
            "wrong count": lambda d: d["measurements"][1].update(
                actual_output_tokens=13
            ),
            "cache hit": lambda d: d["measurements"][0].update(
                cache_status="prefix_hit"
            ),
            "unexpected lookup": lambda d: d["measurements"][0].update(
                prompt_lookup_searches=1
            ),
            "bad acceptance": lambda d: d["measurements"][1].update(
                prompt_lookup_accepted_tokens=8
            ),
            "bad decode": lambda d: d["measurements"][1].update(
                decode_output_tokens=100
            ),
            "bad layout": lambda d: d["identity"].update(
                loaded_model_layout_sha256="wrong"
            ),
            "bad tokenizer": lambda d: d.update(lookup_tokenizer_sha256="wrong"),
            "instrumentation": lambda d: d.update(lookup_runtime_instrumented=True),
            "adaptive": lambda d: d.update(experimental_adaptive_prompt_lookup=True),
            "native failure": lambda d: d.update(performance_failures=["failure"]),
            "NaN timing": lambda d: d["measurements"][1].update(wall_ms=float("nan")),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                document = cohort_document()
                mutate(document)
                with self.assertRaises(ValueError):
                    lookup.analyze(document, "complete")

    def test_case_drift_blocks_aggregate_win(self):
        document = cohort_document()
        document["measurements"][3]["wall_ms"] = 120
        result = lookup.analyze(document, "complete")
        self.assertFalse(result["primary_decision"]["qualified"])
        self.assertIn("lookup_edit_python", result["unstable_cases"])

    def test_one_regressing_edit_blocks_aggregate_win(self):
        document = cohort_document()
        for row in document["measurements"]:
            if (
                row["scenario"] == "lookup_edit_python"
                and row["lookup_abba_variant"] == "on"
            ):
                row["wall_ms"] = 101
        result = lookup.analyze(document, "complete")
        self.assertFalse(result["primary_decision"]["qualified"])
        self.assertEqual(result["nonpositive_edits"], ["lookup_edit_python"])

    def test_material_control_regression_blocks_win(self):
        document = cohort_document()
        for row in document["measurements"]:
            if (
                row["scenario"] == "lookup_control_http"
                and row["lookup_abba_variant"] == "on"
            ):
                row["wall_ms"] = 104
        result = lookup.analyze(document, "complete")
        self.assertFalse(result["primary_decision"]["qualified"])
        self.assertEqual(
            result["material_control_regressions"], ["lookup_control_http"]
        )

    def test_task_diagnostic_requires_entire_response(self):
        self.assertTrue(
            score.assess_response(
                "fresh_edit_clause", "heading\ntext", "heading\ntext"
            )["exact_complete_edit"]
        )
        self.assertFalse(
            score.assess_response("fresh_edit_clause", "text", "heading\ntext")[
                "exact_complete_edit"
            ]
        )
        self.assertFalse(
            score.assess_response(
                "lookup_edit_python", "Here it is:\n```python\nx\n```", "x"
            )["exact_complete_edit"]
        )


if __name__ == "__main__":
    unittest.main()
