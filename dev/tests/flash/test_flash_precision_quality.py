"""CPU-only checks of quality fixture detection, not model-generated answers."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.benchmarks import flash_precision_quality as quality


def record(text, prompt=512, *, calls=None, finish="stop"):
    return {
        "text": text,
        "tool_calls": calls or [],
        "finish_reason": finish,
        "errors": [],
        "http_status": 200,
        "content_type": "text/event-stream",
        "stream": True,
        "done": True,
        "first_content_ms": 1,
        "request_ids": ["fixture"],
        "model_ids": [quality.MODEL],
        "expected_model": quality.MODEL,
        "reasoning_text": "",
        "error_frames": [],
        "cache_disabled": True,
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": 20,
            "total_tokens": prompt + 20,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "metrics": {"cache": {"matched_tokens": 0}, "prefill": {"tokens": prompt}},
    }


def spec(identifier):
    row = next(row for row in quality.supplement_specs() if row["id"] == identifier)
    row["prompt_token_count"] = row.get("target_prompt_tokens", 512)
    return row


class PrecisionQualityTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch(
            "http.client.HTTPConnection",
            side_effect=AssertionError("CPU tests must not contact services"),
        )
        patch.start()
        self.addCleanup(patch.stop)

    def test_correct_arithmetic_passes_and_wrong_math_fails(self):
        case = spec("long_arithmetic_multi_step")
        self.assertEqual(quality.check_record(record("9352"), case), [])
        self.assertTrue(quality.check_record(record("9342"), case))

    def test_json_types_duplicate_keys_and_extra_keys_are_checked(self):
        case = spec("long_unconstrained_json")
        correct = '{"capital":"Canberra","product":391,"prime":true}'
        self.assertEqual(quality.check_record(record(correct), case), [])
        for wrong in (
            '{"capital":"Canberra","product":"391","prime":true}',
            '{"capital":"Canberra","capital":"Sydney","product":391,"prime":true}',
            '{"capital":"Canberra","product":391,"prime":true,"extra":0}',
        ):
            self.assertTrue(quality.check_record(record(wrong), case))

    def test_factual_concepts_and_empty_output_are_checked(self):
        case = spec("long_factual_prose")
        text = "Evaporation moves water into the atmosphere. Condensation turns water vapor into droplets that form clouds. Precipitation brings water back to the surface as rain or snow."
        self.assertEqual(quality.check_record(record(text), case), [])
        self.assertTrue(
            quality.check_record(
                record("Evaporation is part of the water cycle."), case
            )
        )

    def test_cache_prompt_rows_and_protocol_leaks_fail(self):
        case = spec("long_arithmetic_multiply")
        reused = record("391")
        reused["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
        self.assertTrue(quality.check_record(reused, case))
        self.assertTrue(quality.check_record(record("391", prompt=128), case))
        self.assertTrue(quality.check_record(record("<think>391</think>"), case))

    def test_heldout_code_including_negative_even_numbers(self):
        case = spec("long_python_code")
        code = (
            "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n"
        )
        self.assertEqual(quality.check_record(record(code), case), [])
        wrong = "def sum_even(numbers):\n    return sum(n for n in numbers if n > 0 and n % 2 == 0)\n"
        self.assertTrue(quality.check_record(record(wrong), case))

    def test_code_checker_accepts_loop_and_fences(self):
        case = spec("long_python_code")
        code = "```python\ndef sum_even(numbers):\n    total = 0\n    for n in numbers:\n        if n % 2 == 0:\n            total += n\n    return total\n```"
        self.assertEqual(quality.check_record(record(code), case), [])

    def test_code_checker_rejects_import_io_and_top_level_execution(self):
        expected = spec("long_python_code")["expected"]
        for code in (
            "import os\ndef sum_even(numbers):\n    return 0",
            "def sum_even(numbers):\n    return open('/tmp/quality').read()",
            "print('hello')\ndef sum_even(numbers):\n    return 0",
        ):
            self.assertTrue(quality.code_errors(code, expected))

    def test_infinite_generated_code_is_bounded(self):
        expected = spec("long_python_code")["expected"]
        self.assertTrue(
            quality.code_errors(
                "def sum_even(numbers):\n    while True:\n        pass", expected
            )
        )

    def test_retrieval_answers_remain_lane_specific(self):
        group = next(row for row in quality.supplement_specs() if "lanes" in row)
        self.assertEqual(len(group["lanes"]), 4)
        self.assertEqual(len({lane["expected"]["value"] for lane in group["lanes"]}), 4)
        for lane in group["lanes"]:
            lane["prompt_token_count"] = 512
            self.assertEqual(
                quality.check_record(record(lane["expected"]["value"]), lane), []
            )
            self.assertTrue(quality.check_record(record("wrong-lane"), lane))

    def test_deterministic_specs_are_frozen_and_all_use_large_prefill(self):
        left, right = quality.supplement_specs(), quality.supplement_specs()
        self.assertEqual(left, right)
        for row in left:
            for lane in row.get("lanes", [row]):
                self.assertGreaterEqual(lane["target_prompt_tokens"], 512)
                self.assertEqual(lane["body"]["reasoning_effort"], "none")
                self.assertEqual(lane["body"]["temperature"], 0)

    def test_plan_hash_detects_mutation(self):
        data = {
            "schema": quality.SCHEMA,
            "source_identity": quality.SOURCE_ID,
            "supplements": [],
        }
        data["content_sha256"] = quality.canonical_hash(data)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            quality.write_fresh(path, data)
            self.assertEqual(quality.read_plan(path), data)
            altered = copy.deepcopy(data)
            altered["supplements"] = ["changed"]
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                quality.read_plan(path)

    def test_fresh_artifact_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            quality.write_fresh(path, {"original": True})
            with self.assertRaises(FileExistsError):
                quality.write_fresh(path, {"changed": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})


if __name__ == "__main__":
    unittest.main()
