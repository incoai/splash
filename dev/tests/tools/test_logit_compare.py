import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dev.benchmarks import logit_compare

SEQUENCES = [[5, 1, 2, 3], [7, 7, 0]]


def dump(directory: Path, records, rows) -> Path:
    """A logits dump of SEQUENCES with these records, (sequence, prefix), and
    their logits rows."""
    directory.mkdir()
    (directory / "sequences.txt").write_text(
        "".join(" ".join(map(str, sequence)) + "\n" for sequence in SEQUENCES)
    )
    (directory / "records.txt").write_text(
        "".join(f"{sequence} {prefix}\n" for sequence, prefix in records)
    )
    np.asarray(rows, dtype="<f4").tofile(directory / "logits.f32")
    return directory


class LogitCompareTest(unittest.TestCase):
    RECORDS = [(0, 1), (0, 3), (1, 2)]
    ROWS = [[2.0, 0.0, 1.0, -1.0], [0.5, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]]

    def test_identical_dumps_agree_with_no_divergence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            a = dump(root / "a", self.RECORDS, self.ROWS)
            summary = logit_compare.compare(a, a)
        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["top1_agreement"], 1.0)
        self.assertAlmostEqual(summary["kl_max"], 0.0)
        self.assertAlmostEqual(summary["nll_change_mean"], 0.0)

    def test_records_match_by_prefix_in_any_order(self):
        # The candidate holds the same prefixes reversed, the second with its
        # two most likely tokens swapped; a constant added to a row changes
        # nothing.
        rows = [list(row) for row in self.ROWS]
        swapped = [rows[2], [3.0, 0.5, 0.0, 0.0], [r + 7.0 for r in rows[0]]]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = dump(root / "a", self.RECORDS, rows)
            candidate = dump(root / "b", list(reversed(self.RECORDS)), swapped)
            summary = logit_compare.compare(reference, candidate)
        self.assertAlmostEqual(summary["top1_agreement"], 2 / 3)
        p = logit_compare.log_softmax(np.array(rows[1]))
        q = logit_compare.log_softmax(np.array(swapped[1]))
        self.assertAlmostEqual(summary["kl_max"], float(np.exp(p) @ (p - q)))
        # The second record goes on with token 3, the third with token 0.
        self.assertAlmostEqual(summary["nll_change_mean"], float(p[3] - q[3]) / 3)

    def test_bounds_fail_the_comparison_and_a_missing_prefix_is_an_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = dump(root / "a", self.RECORDS, self.ROWS)
            other = [self.ROWS[0], [0.0, 0.0, 0.0, 9.0], self.ROWS[2]]
            candidate = dump(root / "b", self.RECORDS, other)
            output = root / "summary.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    logit_compare.main(
                        [str(reference), str(candidate), "--min-top1", "0.5"]
                    ),
                    0,
                )
                status = logit_compare.main(
                    [
                        str(reference),
                        str(candidate),
                        "--min-top1",
                        "0.9",
                        "--max-mean-kl",
                        "1e-3",
                        "--json",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertEqual(len(json.loads(output.read_text())["failures"]), 2)
            short = dump(root / "c", self.RECORDS[:2], self.ROWS[:2])
            with self.assertRaisesRegex(ValueError, "lacks 1"):
                logit_compare.compare(reference, short)


if __name__ == "__main__":
    unittest.main()
