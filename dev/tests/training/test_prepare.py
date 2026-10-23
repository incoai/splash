import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from draft_training.config import TrainingError, read_json
from draft_training.data import conversations, disjoint, split_manifest
from draft_training.prepare import prepare_data

ROOT = Path(__file__).resolve().parents[3]


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.jsonl"

    def write_records(self, records):
        self.source.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_normalizes_pairs_and_groups_transitive_source_and_prompt_overlap(self):
        self.write_records(
            [
                {"prompt": "shared  prompt", "response": "one", "source_id": "a"},
                {"prompt": "different", "completion": "two", "source_id": "a"},
                {"prompt": "shared\nprompt", "response": "three", "source_id": "b"},
                {"prompt": "fourth", "source_id": "b"},
                *({"prompt": f"independent {i}"} for i in range(12)),
            ]
        )
        output = self.root / "prepared"
        report = prepare_data(self.source, output, 0.2, 0.2, 17)
        identities = {
            name: split_manifest(output / f"{name}.jsonl") for name in report["splits"]
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            disjoint(identities[left], identities[right])
        shared_locations = []
        for name in identities:
            records = conversations(output / f"{name}.jsonl")
            if any(r["source_id"] in ("a", "b") for r in records):
                shared_locations.append(name)
                self.assertEqual(sum(r["source_id"] in ("a", "b") for r in records), 4)
                self.assertTrue(
                    any(
                        m["role"] == "assistant" for r in records for m in r["messages"]
                    )
                )
        self.assertEqual(len(shared_locations), 1)
        self.assertEqual(sum(s["conversations"] for s in report["splits"].values()), 16)
        self.assertEqual(read_json(output / "preparation.json"), report)

    def test_same_seed_reproduces_files_and_refuses_overwrite(self):
        self.write_records([{"prompt": f"example {i}"} for i in range(20)])
        a, b = self.root / "a", self.root / "b"
        prepare_data(self.source, a)
        prepare_data(self.source, b)
        for name in (
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "preparation.json",
        ):
            self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
        original = (a / "train.jsonl").read_bytes()
        with self.assertRaisesRegex(TrainingError, "already exists"):
            prepare_data(self.source, a)
        self.assertEqual((a / "train.jsonl").read_bytes(), original)

    def test_invalid_data_or_insufficient_groups_publishes_no_output(self):
        for records, message in (
            ([{"prompt": "same"}] * 4, "three independent"),
            ([{"prompt": 17}], "nonempty text"),
            ([{"prompt": "q", "response": "a", "completion": "b"}], "not both"),
        ):
            with self.subTest(records=records):
                self.write_records(records)
                output = self.root / "invalid"
                with self.assertRaisesRegex(TrainingError, message):
                    prepare_data(self.source, output)
                self.assertFalse(output.exists())

    def test_cli_prepares_data_without_importing_tensor_dependencies(self):
        self.write_records([{"prompt": f"question {i}"} for i in range(10)])
        script = (
            "import sys; from draft_training.__main__ import main; "
            "result=main(sys.argv[1:]); assert 'torch' not in sys.modules; "
            "raise SystemExit(result)"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                "prepare-data",
                "--data",
                str(self.source),
                "--output",
                str(self.root / "cli"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["groups"],
            {"train": 8, "validation": 1, "test": 1},
        )

    def test_invalid_fractions_fail_before_creating_output(self):
        from draft_training.__main__ import main

        self.write_records([{"prompt": f"question {i}"} for i in range(10)])
        for fraction in ("nan", "0", "1"):
            with (
                self.subTest(fraction=fraction),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                output = self.root / "invalid-fraction"
                self.assertEqual(
                    main(
                        [
                            "prepare-data",
                            "--data",
                            str(self.source),
                            "--output",
                            str(output),
                            "--validation-fraction",
                            fraction,
                        ]
                    ),
                    1,
                )
                self.assertFalse(output.exists())
