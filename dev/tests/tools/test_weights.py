import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dev.benchmarks import weights

MODEL_ROOT = Path("/models/.resolved/assembly")
A, B, C = ("a" * 64, "b" * 64, "c" * 64)
# Options of weight-digests that stream the experts and the PLE table.
STREAMED = ("--expert-memory", "7G", "--ple-memory", "2G")


def digest_tool(build: Path, images: dict, status: int = 0):
    """A weight-digests in build that prints images, {component: sha256},
    and logs the model root and options it was given."""
    tool = build / weights.WEIGHT_DIGESTS
    tool.parent.mkdir(parents=True, exist_ok=True)
    document = [
        {"component": component, "bytes": 16384, "sha256": digest}
        for component, digest in images.items()
    ]
    tool.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"with open({str(build / 'arguments')!r}, 'a') as log:\n"
        "    log.write(' '.join(sys.argv[2:]) + '\\n')\n"
        + (
            f"print('error: no model', file=sys.stderr)\nraise SystemExit({status})\n"
            if status
            else f"print({json.dumps(json.dumps(document))})\n"
        )
    )
    tool.chmod(0o755)


class WeightBytesTests(unittest.TestCase):
    def test_builds_must_load_the_same_images_with_the_same_bytes(self):
        self.assertTrue(weights.compare({"x": A}, {"x": A})["pass"])
        result = weights.compare({"x": A, "y": B}, {"x": A, "y": C, "z": C})
        self.assertFalse(result["pass"])
        self.assertEqual(
            result["failures"],
            [
                f"y: the baseline loaded {B}, the candidate {C}",
                f"z: the baseline loaded nothing, the candidate {C}",
            ],
        )
        self.assertEqual(
            result["images"][0],
            {"component": "x", "baseline_sha256": A, "candidate_sha256": A},
        )
        self.assertIn(
            "y: the baseline loaded " + B + ", the candidate nothing",
            weights.compare({"x": A, "y": B}, {"x": A})["failures"],
        )

    def test_digests_runs_the_build_tool(self):
        with TemporaryDirectory() as directory:
            build = Path(directory)
            self.assertFalse(weights.loads_in_memory(build))
            digest_tool(build, {"target/layer-0.bin": A, "vision/model.bin": B})
            self.assertTrue(weights.loads_in_memory(build))
            self.assertEqual(
                weights.digests(build, MODEL_ROOT),
                {"target/layer-0.bin": A, "vision/model.bin": B},
            )
            self.assertEqual((build / "arguments").read_text(), f"{MODEL_ROOT}\n")
            # Options load the model as they say.
            weights.digests(build, MODEL_ROOT, STREAMED)
            self.assertEqual(
                (build / "arguments").read_text().splitlines()[-1],
                " ".join([str(MODEL_ROOT), *STREAMED]),
            )
            digest_tool(build, {}, status=1)
            with self.assertRaisesRegex(RuntimeError, "exited 1: error: no model"):
                weights.digests(build, MODEL_ROOT)


if __name__ == "__main__":
    unittest.main()
