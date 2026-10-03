import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dev.benchmarks import prepared

PACKAGE = Path("/models/.resolved/assembly")


class PreparedBytesTests(unittest.TestCase):
    @staticmethod
    def entry(
        cache: Path,
        key: str,
        digest: str,
        component=None,
        inputs="i" * 64,
        source=PACKAGE / "target",
    ):
        """An entry with provenance when it names its component; with no
        source file otherwise."""
        directory = cache / key
        directory.mkdir(parents=True)
        (directory / "weights").write_text("")
        (directory / "sha256").write_text(digest)
        if component:
            (directory / "source").write_text(
                f"{prepared.PROVENANCE}\ncomponent {component}\n"
                f"inputs {inputs}\nsource {source}\n"
            )

    def compare(self, baseline, candidate, required=True):
        return prepared.compare(baseline, candidate, package=PACKAGE, required=required)

    def test_candidate_must_hold_every_baseline_entry_of_the_package(self):
        a, b, d, e = ("a" * 64, "b" * 64, "d" * 64, "e" * 64)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline", root / "candidate"
            # Another model's and another revision's entries are not compared.
            self.entry(baseline, "0" * 64, e, "target/layer-0.bin", "o" * 64, "/other")
            self.entry(
                baseline, "9" * 64, e, "target/layer-0.bin", "r" * 64, f"{PACKAGE}-old"
            )
            (baseline / ("4" * 64)).mkdir()  # an incomplete entry
            (baseline / "verified-v3").mkdir()
            self.assertEqual(len(prepared.entries(baseline)), 2)
            self.assertEqual(prepared.entries(root / "missing"), [])
            result = self.compare(baseline, candidate)
            self.assertEqual(result["entries"], [])
            self.assertIn("prepared no weights", result["failures"][0])
            self.assertTrue(self.compare(baseline, candidate, required=False)["pass"])

            self.entry(baseline, "1" * 64, a, "target/layer-0.bin")
            self.entry(
                baseline, "2" * 64, b, "vision/model.bin", source=PACKAGE / "vision"
            )
            self.assertFalse(self.compare(baseline, candidate)["pass"])
            # Other keys, same bytes: equal.
            self.entry(candidate, "5" * 64, a, "target/layer-0.bin")
            self.entry(candidate, "6" * 64, b, "vision/model.bin")
            result = self.compare(baseline, candidate)
            self.assertTrue(result["pass"], result["failures"])
            self.assertEqual(
                [(row["key"][0], row["candidate_sha256"]) for row in result["entries"]],
                [("1", a), ("2", b)],
            )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline", root / "candidate"
            self.entry(baseline, "1" * 64, a, "target/layer-0.bin")
            self.entry(baseline, "2" * 64, b, "vision/model.bin")
            # Changed bytes of a component from the same source data.
            self.entry(candidate, "5" * 64, a, "target/layer-0.bin")
            self.entry(candidate, "6" * 64, d, "vision/model.bin")
            failures = self.compare(baseline, candidate)["failures"]
            self.assertEqual(len(failures), 1, failures)
            self.assertIn("vision/model.bin: the candidate prepared", failures[0])
            # An entry without a source record cannot be told apart: compared
            # by its bytes, which no candidate entry holds.
            self.entry(baseline, "8" * 64, e)
            failures = self.compare(baseline, candidate)["failures"]
            self.assertEqual(len(failures), 2, failures)
            self.assertIn("8" * 64, failures[1])
            self.assertIn("lacks the baseline's prepared bytes", failures[1])
            # Any entry may hold them.
            self.entry(candidate, "7" * 64, e, "target/layer-9.bin", "k" * 64)
            self.assertEqual(len(self.compare(baseline, candidate)["failures"]), 1)

    def test_a_baseline_of_another_identity_gets_its_own_cache(self):
        with TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "release/model"
            environment = prepared.baseline_environment(output)
            self.assertEqual(
                environment, {"SPLASH_WEIGHT_CACHE": str(output / "baseline-weights")}
            )
            self.assertTrue((output / "baseline-weights").is_dir())
            self.assertEqual(prepared.baseline_environment(output), environment)

    def test_preparation_identity_and_cache_root(self):
        with TemporaryDirectory() as directory:
            build = Path(directory)
            self.assertIsNone(prepared.preparation_identity(build))
            (build / "engine").mkdir()
            (build / prepared.IDENTITY_HEADER).write_text("#define X 1\n")
            self.assertEqual(prepared.preparation_identity(build), b"#define X 1\n")
        self.assertEqual(
            prepared.cache_root({"SPLASH_WEIGHT_CACHE": "/c", "HOME": "/h"}), Path("/c")
        )
        self.assertEqual(
            prepared.cache_root({"SPLASH_WEIGHT_CACHE": "", "HOME": "/h"}),
            Path("/h/Library/Caches/Splash/weights"),
        )
        with self.assertRaises(ValueError):
            prepared.cache_root({})


if __name__ == "__main__":
    unittest.main()
