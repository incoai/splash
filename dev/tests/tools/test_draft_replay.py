"""Acceptance labels must stop at a rejection or a retained-prefix boundary."""

import copy
import gzip
import hashlib
import itertools
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from dev.benchmarks.draft_replay import probabilities, read_capture, select, summarize


def fixture(reference=(11, 20, 30)):
    n, k = 3, 2
    candidates = [[10, 11], [20, 21], [30, 31]]
    unary = [[1.0, 0.0] for _ in range(n)]
    edges = [[[0.0] * k for _ in range(1 if p == 0 else k)] for p in range(n)]
    proposed = [10, 20, 30]
    accepted = next((i for i in range(n) if proposed[i] != reference[i]), n)
    target = list(proposed[:accepted]) + [reference[accepted] if accepted < n else 40]
    record = {
        "kind": "cycle",
        "request_id": 1,
        "cycle": 0,
        "lane": 0,
        "position": 10,
        "accepted": accepted,
        "retained": accepted + 1,
        "verify_rows": n + 1,
        "proposed": proposed,
        "candidates": sum(candidates, []),
        "unary": sum(unary, []),
        "edges": [sum(row, []) for row in edges],
        "confidence": [probabilities(row)[0] for row in unary],
        "target_tokens": target,
        "output_tokens": [1] + target[:-1],
    }
    metadata = {
        "kind": "metadata",
        "schema": 1,
        "sampling": False,
        "positions": n,
        "candidates": k,
        "width": 1,
    }
    return metadata, record


class DraftReplayTests(unittest.TestCase):
    def test_first_rejection_selection_and_recall_are_distinct(self):
        for reference, category in (
            ((11, 20, 30), "selection"),
            ((99, 20, 30), "candidate_recall"),
        ):
            meta, record = fixture(reference)
            report = summarize(meta, [record])
            self.assertEqual(report["first_rejections"][category], 1)
            self.assertEqual(
                [r["reached"] for r in report["conditional_acceptance"]], [1, 0, 0]
            )
            self.assertIsNone(report["conditional_acceptance"][1]["rate"])

    def test_later_rejection_labels_only_the_prefix(self):
        meta, record = fixture((10, 21, 30))
        report = summarize(meta, [record])
        self.assertEqual(
            [r["accepted"] for r in report["conditional_acceptance"]], [1, 0, 0]
        )
        self.assertEqual(
            [r["reached"] for r in report["conditional_acceptance"]], [1, 1, 0]
        )
        self.assertEqual(report["mean_retained_tokens"], 2)

    def test_full_acceptance_includes_the_bonus(self):
        meta, record = fixture((10, 20, 30))
        report = summarize(meta, [record])
        self.assertEqual(report["mean_retained_tokens"], 4)
        self.assertEqual(
            report["counterfactual"]["mean_tokens_at_original_starts"]["greedy"], 4
        )

    def test_short_retention_censors_unconsumed_labels(self):
        meta, record = fixture((10, 20, 30))
        record["retained"] = 2
        record["target_tokens"] = record["target_tokens"][:2]
        record["output_tokens"] = record["output_tokens"][:2]
        report = summarize(meta, [record])
        self.assertEqual(report["accepted_proposals"], 1)
        self.assertEqual(report["first_rejections"]["censored"], 1)
        self.assertEqual(report["counterfactual"]["censored_reference_cycles"], 1)

    def test_terminal_correction_does_not_add_a_failure_label(self):
        meta, record = fixture()
        # The pending correction is emitted as a terminal token without KV.
        # Guessing that token would not retain another row, so censor it just
        # as an accepted terminal proposal is censored by retained - 1.
        record["output_tokens"].append(record["target_tokens"][-1])
        report = summarize(meta, [record])
        self.assertEqual(
            report["first_rejections"],
            {"selection": 0, "candidate_recall": 0, "censored": 1},
        )
        self.assertEqual(report["conditional_acceptance"][0]["reached"], 0)

    def test_following_cycle_supplies_true_continuation_after_rejection(self):
        meta, first = fixture((11, 20, 30))
        _, second = fixture((10, 20, 30))
        second.update(cycle=1, position=11)
        second["output_tokens"][0] = 11
        report = summarize(meta, [first, second])
        self.assertEqual(report["counterfactual"]["complete_reference_cycles"], 2)
        means = report["counterfactual"]["mean_tokens_at_original_starts"]
        self.assertEqual(means["greedy"], 2.5)
        self.assertEqual(means["candidate_oracle"], 3.0)

    def test_requests_do_not_share_continuations(self):
        meta, first = fixture((11, 20, 30))
        _, second = fixture((10, 20, 30))
        second["request_id"] = 2
        second["lane"] = 1
        meta["width"] = 2
        report = summarize(meta, [first, second])
        self.assertEqual(report["counterfactual"]["complete_reference_cycles"], 1)

    def test_invalid_capture_and_layout_are_rejected(self):
        meta, record = fixture()
        mutations = (
            {"unary": [math.nan] * 6},
            {"confidence": [0.9] * 3},
            {"proposed": [11, 20, 30]},
            {"edges": [[0] * 4] * 3},
            {"candidates": [10] * 6},
            {"accepted": 4},
            {"verify_rows": 0},
            {"target_tokens": [10]},
            {"cycle": 1},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize(meta, [dict(record, **mutation)])
        with self.assertRaises(ValueError):
            summarize(dict(meta, sampling=True), [record])

    def test_conflicting_or_noncontiguous_references_are_rejected(self):
        meta, first = fixture((11, 20, 30))
        second = copy.deepcopy(first)
        second.update(cycle=1, position=11)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            summarize(meta, [first, second])
        second["position"] = 20
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            summarize(meta, [first, second])

    def test_truncated_files_and_wrong_record_count_are_rejected(self):
        meta, record = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            for end in ([], [{"kind": "end", "records": 2}]):
                path.write_text("\n".join(map(json.dumps, [meta, record, *end])))
                with self.assertRaises(ValueError):
                    read_capture(path)
            path.write_text(
                "\n".join(
                    map(json.dumps, [meta, record, {"kind": "end", "records": 1}])
                )
            )
            self.assertEqual(read_capture(path), (meta, [record]))

    def test_compressed_capture(self):
        meta, record = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl.gz"
            with gzip.open(path, "wt") as file:
                file.write(
                    "\n".join(
                        map(json.dumps, [meta, record, {"kind": "end", "records": 1}])
                    )
                )
            self.assertEqual(read_capture(path), (meta, [record]))

    def test_boolean_schema_counts_and_nonobject_records_are_rejected(self):
        meta, record = fixture()
        with self.assertRaises(ValueError):
            summarize(dict(meta, schema=True), [record])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            for rows in (
                [meta, record, {"kind": "end", "records": True}],
                [meta, [], {"kind": "end", "records": 1}],
            ):
                path.write_text("\n".join(map(json.dumps, rows)))
                with self.assertRaises(ValueError):
                    read_capture(path)

    def test_archived_capture_checksums_and_acceptance_reports(self):
        repo = Path(__file__).resolve().parents[3]
        evidence = repo / "dev/benchmarks/results/dflash-acceptance-m5-pro"
        manifest = json.loads((evidence / "manifest.json").read_text())
        for entry in manifest["captures"]:
            with self.subTest(capture=entry["name"]):
                path = repo / entry["capture_path"]
                data = path.read_bytes()
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), entry["capture_sha256"]
                )
                self.assertEqual(
                    hashlib.sha256(gzip.decompress(data)).hexdigest(),
                    entry["raw_sha256"],
                )
                expected = json.loads(
                    (evidence / (entry["name"] + ".report.json")).read_text()
                )
                actual = summarize(*read_capture(path))
                for key in (
                    "cycles",
                    "verified_proposals",
                    "accepted_proposals",
                    "first_rejections",
                    "selection_miss_unary_rank_counts",
                    "conditional_acceptance",
                    "counterfactual",
                ):
                    self.assertEqual(actual[key], expected[key], key)
                self.assertEqual(
                    actual["mean_retained_tokens"], expected["mean_retained_tokens"]
                )

    def test_incomplete_or_duplicate_batches_are_rejected(self):
        meta, record = fixture()
        with self.assertRaisesRegex(ValueError, "incomplete batch"):
            summarize(dict(meta, width=2), [record])
        with self.assertRaisesRegex(ValueError, "duplicate batch lane"):
            summarize(meta, [record, record])

    def test_greedy_replay_rounds_score_sums_as_fp32(self):
        self.assertEqual(select([[16777216.0, 16777216.0]], [[[0.0, 1.0]]]), [0])

    def test_scale_one_preserves_replay_and_zero_uses_unary(self):
        unary = [[1.0, 0.0]]
        edges = [[[0.0, 2.0]]]
        self.assertEqual(select(unary, edges, edge_scale=1), [1])
        self.assertEqual(select(unary, edges, edge_scale=0), [0])
        meta, record = fixture((10, 20, 30))
        report = summarize(meta, [record], edge_scales=(0, 1, 2))
        self.assertEqual(report["edge_scale_replay"]["complete_reference_cycles"], 1)
        baseline = report["counterfactual"]["mean_tokens_at_original_starts"]["greedy"]
        self.assertTrue(
            all(
                row["mean_tokens_at_original_starts"] == baseline
                for row in report["edge_scale_replay"]["scales"]
            )
        )

    def test_scale_replay_excludes_censored_tails(self):
        meta, record = fixture()
        report = summarize(meta, [record], edge_scales=(0, 1))
        self.assertEqual(report["edge_scale_replay"]["complete_reference_cycles"], 0)
        self.assertIsNone(
            report["edge_scale_replay"]["scales"][0]["mean_tokens_at_original_starts"]
        )

    def test_invalid_edge_scales_are_rejected(self):
        meta, record = fixture()
        for scale in (-1, math.inf, math.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                summarize(meta, [record], edge_scales=(scale,))
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                select([[0]], [[[0]]], edge_scale=scale)

    def test_dynamic_programs_match_exhaustive_enumeration(self):
        rng = random.Random(17)
        for _ in range(30):
            n, k = 3, 3
            unary = [[rng.uniform(-3, 3) for _ in range(k)] for _ in range(n)]
            edges = [
                [
                    [rng.uniform(-3, 3) for _ in range(k)]
                    for _ in range(1 if p == 0 else k)
                ]
                for p in range(n)
            ]

            def objective(path, mode):
                previous, survival, total = 0, 1.0, 0.0
                for p, chosen in enumerate(path):
                    scores = [u + e for u, e in zip(unary[p], edges[p][previous])]
                    q = probabilities(scores)[chosen]
                    survival *= q
                    total += survival
                    previous = chosen
                return total if mode == "expected_length" else survival

            for mode in ("expected_length", "viterbi"):
                selected = select(unary, edges, mode)
                best = max(
                    objective(path, mode)
                    for path in itertools.product(range(k), repeat=n)
                )
                self.assertAlmostEqual(objective(selected, mode), best)

    def test_ties_use_lowest_candidate_and_logits_are_shift_invariant(self):
        unary = [[0.0, 0.0]] * 3
        edges = [[[0.0, 0.0]], [[0.0, 0.0]] * 2, [[0.0, 0.0]] * 2]
        for mode in ("greedy", "expected_length", "viterbi"):
            self.assertEqual(select(unary, edges, mode), [0, 0, 0])
        shifted = copy.deepcopy(edges)
        shifted[1][1] = [10000.0, 10000.0]
        for mode in ("expected_length", "viterbi"):
            self.assertEqual(select(unary, shifted, mode), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
