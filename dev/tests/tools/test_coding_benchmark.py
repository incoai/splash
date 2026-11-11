import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.benchmarks.serving import coding
from dev.tests.tools.test_serving_benchmark import StreamingServer, content_event


class CodingBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def workload(self):
        (self.root / "source.cpp").write_text("first\nsecond\nthird\n")
        return {
            "sources": [{"path": "source.cpp", "start": 2, "end": 3}],
            "cases": [{"id": "coding", "turns": ["Write a test", "Improve it"]}],
        }

    def test_ranges_are_exact_and_never_silently_truncated(self):
        workload = self.workload()
        source, provenance = coding.prepare(self.root, workload)
        self.assertEqual(source, "--- source.cpp:2-3 ---\nsecond\nthird\n")
        self.assertEqual(
            provenance[0]["excerpt_sha256"], coding.digest("second\nthird\n")
        )
        workload["sources"][0]["end"] = 4
        with self.assertRaises(ValueError):
            coding.prepare(self.root, workload)

    def test_sources_cannot_escape_the_selected_repository(self):
        workload = self.workload()
        workload["sources"][0]["path"] = "../outside.cpp"
        with self.assertRaises(ValueError):
            coding.prepare(self.root, workload)

    def conversation(self, responder, case=None):
        if case is None:
            case = {"id": "coding", "turns": ["write", "improve"]}
        workload = self.workload()
        workload["cases"] = [case]
        path = self.root / "workload.json"
        path.write_text(json.dumps(workload))
        with StreamingServer(responder) as server:
            args = argparse.Namespace(
                out=self.root / "out",
                workload=path,
                repo=self.root,
                build="A",
                model="model",
                users=1,
                rounds=1,
                tokens=4096,
                port=server.port,
                timeout=5,
                validator_timeout=5,
            )
            with mock.patch.object(coding.clients, "get_json", return_value={}):
                coding.run(args)
        return json.loads((args.out / "results.jsonl").read_text())["requests"]

    @staticmethod
    def answer(text, reason="stop"):
        return [
            content_event(text),
            {
                "choices": [{"delta": {}, "finish_reason": reason}],
                "usage": {"completion_tokens": 3, "prompt_tokens": 20},
            },
        ]

    def test_followup_uses_the_actual_answer_and_keeps_the_prefix(self):
        bodies = []

        def responder(body):
            bodies.append(json.loads(body))
            return self.answer("actual first answer" if len(bodies) == 1 else "second")

        rows = self.conversation(responder)
        self.assertEqual(len(rows), 2)
        first, second = [body["messages"] for body in bodies]
        self.assertEqual(first, second[: len(first)])
        self.assertEqual(
            second[-3], {"role": "assistant", "content": "actual first answer"}
        )
        self.assertTrue(all(row["complete"] for row in rows))
        self.assertFalse(any(row["pass"] for row in rows))

    def test_length_stop_fails_and_does_not_continue_or_validate(self):
        case = {
            "id": "coding",
            "turns": ["write", "improve"],
            "validators": {"1": ["ignored"]},
        }
        with mock.patch.object(coding, "validate") as validate:
            rows = self.conversation(
                lambda body: self.answer("cut off", "length"), case
            )
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["complete"])
        self.assertFalse(rows[0]["pass"])
        validate.assert_not_called()

    def test_stream_missing_done_is_a_failure(self):
        with mock.patch.object(
            coding.clients, "Stream", side_effect=ConnectionError("missing DONE")
        ):
            rows = [coding.generate(1, {}, {}, self.root / "answer.txt", 5)]
        self.assertFalse(rows[0]["pass"])
        self.assertIn("missing DONE", rows[0]["error"])

    def test_validator_failure_and_timeout_are_preserved(self):
        answer = self.root / "answer.txt"
        answer.write_text("code")
        failed = coding.validate(
            [sys.executable, "-c", "raise SystemExit(7)"], answer, self.root, 5
        )
        self.assertFalse(failed["pass"])
        self.assertEqual(failed["returncode"], 7)
        timed = coding.validate(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            answer,
            self.root,
            0.01,
        )
        self.assertFalse(timed["pass"])
        self.assertIn("error", timed)

    def test_health_counts_restarts_and_reclaim_errors_but_allows_covered_counts(self):
        before = {"transport": {"restarts": 2}}
        after = {
            "transport": {"restarts": 3},
            "memory_governor": {
                "count_checks": {"covered": 10, "overcounted": 1, "short_releases": 0}
            },
        }
        self.assertEqual(
            coding.health_errors(before, after),
            ["transport.restarts", "memory_governor.count_checks.overcounted"],
        )
        after["transport"]["restarts"] = 2
        after["memory_governor"]["count_checks"]["overcounted"] = 0
        self.assertEqual(coding.health_errors(before, after), [])

    def test_run_records_missing_turns_as_failure(self):
        workload = self.workload()
        path = self.root / "workload.json"
        path.write_text(json.dumps(workload))
        args = argparse.Namespace(
            out=self.root / "out",
            workload=path,
            repo=self.root,
            build="A",
            model="m",
            users=1,
            rounds=1,
            tokens=4096,
            port=1,
            timeout=5,
            validator_timeout=5,
        )
        with (
            mock.patch.object(coding.clients, "get_json", return_value={}),
            mock.patch.object(
                coding,
                "generate",
                return_value={
                    "case": "coding",
                    "turn": 1,
                    "pass": False,
                    "complete": False,
                },
            ),
        ):
            result = coding.run(args)
        self.assertFalse(result["pass"])
        with self.assertRaises(FileExistsError):
            coding.run(args)

    def cohort(self, name, build, wall=10):
        path = self.root / name
        path.mkdir()
        meta = {
            "build": build,
            "model": "m",
            "users": 1,
            "rounds": 1,
            "max_completion_tokens": 4096,
            "workload_sha256": "w",
            "source_sha256": "s",
        }
        rows = [
            {
                "user": 1,
                "case": "c",
                "turn": turn,
                "request_sha256": str(turn),
                "pass": True,
                "complete": True,
                "finish_reason": "stop",
                "validation": {"pass": True},
                "usage": {"completion_tokens": 100},
            }
            for turn in (1, 2)
        ]
        record = {
            "round": 1,
            "wall_s": wall,
            "generation_wall_s": wall,
            "expected": 2,
            "requests": rows,
        }
        (path / "meta.json").write_text(json.dumps(meta))
        (path / "results.jsonl").write_text(json.dumps(record) + "\n")
        return path

    def test_compare_distinguishes_nonregression_from_gain(self):
        paths = [self.cohort(str(i), build) for i, build in enumerate("ABBA")]
        result = coding.compare(paths)
        self.assertTrue(result["pass"])
        self.assertFalse(result["gain"])

    def test_compare_rejects_a_truncated_answer_despite_a_saved_pass(self):
        paths = [self.cohort(str(i), build) for i, build in enumerate("ABBA")]
        target = paths[1] / "results.jsonl"
        record = json.loads(target.read_text())
        record["requests"][0]["finish_reason"] = "length"
        target.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "unvalidated"):
            coding.compare(paths)

    def test_compare_allows_real_followups_but_rejects_changed_initial_requests(self):
        paths = [self.cohort(str(i), build) for i, build in enumerate("ABBA")]
        target = paths[1] / "results.jsonl"
        record = json.loads(target.read_text())
        record["requests"][1]["request_sha256"] = "actual answer differs"
        target.write_text(json.dumps(record) + "\n")
        self.assertTrue(coding.compare(paths)["pass"])
        record["requests"][0]["request_sha256"] = "different task"
        target.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "initial requests"):
            coding.compare(paths)

    def test_validation_runs_between_waves_and_is_supplied_to_next_turn(self):
        workload = self.workload()
        workload["cases"][0]["validators"] = {"1": ["trusted"], "2": ["trusted"]}
        path = self.root / "workload.json"
        path.write_text(json.dumps(workload))
        generated = []

        def respond(body):
            generated.append(body)
            return self.answer("code")

        checked = []

        def validate(*args):
            checked.append(len(generated))
            return {"pass": True, "returncode": 0}

        with StreamingServer(respond) as server:
            args = argparse.Namespace(
                out=self.root / "out",
                workload=path,
                repo=self.root,
                build="A",
                model="m",
                users=1,
                rounds=1,
                tokens=4096,
                port=server.port,
                timeout=5,
                validator_timeout=5,
            )
            with (
                mock.patch.object(coding.clients, "get_json", return_value={}),
                mock.patch.object(coding, "validate", side_effect=validate),
            ):
                result = coding.run(args)
        self.assertTrue(result["pass"])
        self.assertEqual(result["validated"], 2)
        self.assertEqual(checked, [1, 2])
        self.assertIn(
            "Local validator result",
            json.loads(generated[1])["messages"][-2]["content"],
        )

    def test_unequal_conversations_validate_only_after_the_concurrent_wave(self):
        workload = self.workload()
        workload["cases"] = [
            {"id": "short", "turns": ["one"], "validators": {"1": ["trusted"]}},
            {
                "id": "long",
                "turns": ["one", "two"],
                "validators": {"1": ["trusted"], "2": ["trusted"]},
            },
        ]
        path = self.root / "workload.json"
        path.write_text(json.dumps(workload))
        generated = []
        checked = []

        def respond(body):
            generated.append(body)
            return self.answer("code")

        def validate(*args):
            checked.append(len(generated))
            return {"pass": True}

        with StreamingServer(respond) as server:
            args = argparse.Namespace(
                out=self.root / "out",
                workload=path,
                repo=self.root,
                build="A",
                model="m",
                users=2,
                rounds=1,
                tokens=4096,
                port=server.port,
                timeout=5,
                validator_timeout=5,
            )
            with (
                mock.patch.object(coding.clients, "get_json", return_value={}),
                mock.patch.object(coding, "validate", side_effect=validate),
            ):
                result = coding.run(args)
        self.assertTrue(result["pass"])
        self.assertEqual(result["validated"], 3)
        self.assertEqual(checked, [2, 2, 3])

    def test_compare_rejects_missing_records_and_changed_initial_requests(self):
        paths = [self.cohort(str(i), build) for i, build in enumerate("ABBA")]
        target = paths[1] / "results.jsonl"
        record = json.loads(target.read_text())
        record["requests"].pop()
        target.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            coding.compare(paths)
        record["expected"] = 1
        target.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "cohorts"):
            coding.compare(paths)


if __name__ == "__main__":
    unittest.main()
