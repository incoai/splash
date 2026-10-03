import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dev.benchmarks import weights

PACKAGE = Path("/models/.resolved/assembly")
A, B, C = ("a" * 64, "b" * 64, "c" * 64)


def digest_tool(build: Path, images: dict, status: int = 0):
    """A weight-digests in build that prints images, {component: sha256},
    and logs the model root it was given."""
    tool = build / weights.WEIGHT_DIGESTS
    tool.parent.mkdir(parents=True, exist_ok=True)
    document = [
        {"component": component, "bytes": 16384, "sha256": digest}
        for component, digest in images.items()
    ]
    tool.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"with open({str(build / 'roots')!r}, 'a') as log:\n"
        "    log.write(sys.argv[2] + '\\n')\n"
        + (
            f"print('error: no model', file=sys.stderr)\nraise SystemExit({status})\n"
            if status
            else f"print({json.dumps(json.dumps(document))})\n"
        )
    )
    tool.chmod(0o755)


def entry(cache: Path, key: str, digest: str, lines: list[str]):
    """A cache entry of an earlier release whose source file holds lines."""
    directory = cache / key
    directory.mkdir(parents=True)
    (directory / "weights").write_text("")
    (directory / "sha256").write_text(digest)
    (directory / "source").write_text("".join(f"{line}\n" for line in lines))


def provenance(component: str, source=PACKAGE / "target") -> list[str]:
    return [
        weights.PROVENANCE,
        f"component {component}",
        "inputs " + "i" * 64,
        f"source {source}",
    ]


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
                weights.digests(build, PACKAGE),
                {"target/layer-0.bin": A, "vision/model.bin": B},
            )
            self.assertEqual((build / "roots").read_text(), f"{PACKAGE}\n")
            digest_tool(build, {}, status=1)
            with self.assertRaisesRegex(RuntimeError, "exited 1: error: no model"):
                weights.digests(build, PACKAGE)

    def test_an_earlier_release_prepared_the_images_of_the_package(self):
        with TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "release/model"
            environment = weights.baseline_environment(output)
            cache = output / "baseline-weights"
            self.assertEqual(environment, {"SPLASH_WEIGHT_CACHE": str(cache)})
            self.assertTrue(cache.is_dir())
            self.assertEqual(weights.prepared(environment, PACKAGE), {})

            entry(cache, "1" * 64, A, provenance("target/layer-0.bin"))
            entry(
                cache, "2" * 64, B, provenance("vision/model.bin", PACKAGE / "vision")
            )
            # Another model's and another revision's entries, an entry of an
            # earlier format, one without its digest and staging are not.
            entry(cache, "3" * 64, C, provenance("target/head.bin", "/other"))
            entry(cache, "4" * 64, C, provenance("target/head.bin", f"{PACKAGE}-old"))
            entry(cache, "5" * 64, C, [str(PACKAGE / "target"), "head.bin"])
            entry(cache, "6" * 64, "not a digest", provenance("target/head.bin"))
            entry(cache, ".staging-7", C, provenance("target/head.bin"))
            (cache / ("8" * 64)).mkdir()
            self.assertEqual(
                weights.prepared(environment, PACKAGE),
                {"target/layer-0.bin": A, "vision/model.bin": B},
            )

    def test_compare_builds_reads_each_baseline_as_it_loads(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            baseline, candidate = root / "baseline", root / "candidate"
            digest_tool(candidate, {"target/layer-0.bin": A})
            environment = weights.baseline_environment(root / "release")
            # A package an earlier release mapped as it is: nothing to compare.
            self.assertEqual(
                weights.compare_builds(
                    baseline, candidate, PACKAGE, environment, False
                ),
                {"images": [], "failures": [], "pass": True},
            )
            self.assertFalse((candidate / "roots").exists())
            # An assembly it prepared into its cache, which its build names.
            with self.assertRaisesRegex(
                RuntimeError, "has no engine-tests/weight-digests"
            ):
                weights.compare_builds(baseline, candidate, PACKAGE, environment, True)
            (baseline / weights.IDENTITY_HEADER).parent.mkdir(parents=True)
            (baseline / weights.IDENTITY_HEADER).write_text("")
            entry(
                Path(environment["SPLASH_WEIGHT_CACHE"]),
                "1" * 64,
                B,
                provenance("target/layer-0.bin"),
            )
            result = weights.compare_builds(
                baseline, candidate, PACKAGE, environment, True
            )
            self.assertEqual(
                result["failures"],
                [f"target/layer-0.bin: the baseline loaded {B}, the candidate {A}"],
            )
            # A baseline that loads into memory is asked, whatever it was given.
            digest_tool(baseline, {"target/layer-0.bin": A})
            for assembly in (True, False):
                result = weights.compare_builds(
                    baseline, candidate, PACKAGE, None, assembly
                )
                self.assertTrue(result["pass"], result["failures"])
                self.assertEqual(len(result["images"]), 1)


if __name__ == "__main__":
    unittest.main()
