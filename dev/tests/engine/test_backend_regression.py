import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dev.benchmarks import backend_regression as regression
from dev.tests import smoke_real as smoke

SCENARIO_NAMES = ("decode", "partial")


def benchmark_document(
    scenarios=SCENARIO_NAMES, build="build", layout="layout", step=10.0
):
    """backend-benchmark output of two samples for these scenarios."""
    samples = [
        {
            "width": width,
            "sample": sample,
            "output_token_hash": f"hash-{width}",
            "accepted_draft_tokens": 40,
            "drafted_tokens": 64,
            "decode_batches": 8,
            "decode_gpu_ms": step * width * 8,
        }
        for sample in range(2)
        for width in regression.WIDTHS
    ]
    measurements = [
        {
            "scenario": scenario,
            "sample": sample,
            "output_tokens": [7],
            "prefill_gpu_ms": 1000.0,
            "ttft_ms": 300.0,
        }
        for sample in range(2)
        for scenario in regression.PARTIAL
    ]
    return {
        "schema_version": 2,
        "build_id": build,
        "identity": {
            "model_root": "/models/m",
            "loaded_model_layout_sha256": layout,
            "device": "Apple M5 Pro",
        },
        "decode_throughput": {"samples": samples if "decode" in scenarios else []},
        "measurements": measurements if "partial" in scenarios else [],
        "performance_pass": True,
        "performance_failures": [],
    }


def rounds(changes=None):
    """Four ABBA round records; changes maps a round index to a function
    that edits that round's benchmark document."""
    changes = changes or {}
    result = []
    for index, version in enumerate(regression.ROUNDS):
        document = benchmark_document(build=version, layout=f"layout-{version}")
        if index in changes:
            changes[index](document)
        result.append(regression.round_record(version, [document]))
    return result


def decode_samples(document, width):
    return [s for s in document["decode_throughput"]["samples"] if s["width"] == width]


class BackendRegressionTests(unittest.TestCase):
    def test_identical_rounds_pass_every_rule(self):
        summary = regression.summarize(rounds(), False)
        self.assertEqual(summary["failures"], [])
        self.assertTrue(summary["pass"])
        self.assertEqual(
            sorted(summary["speed"]),
            sorted(
                [f"decode_B{width}_gpu_ms_per_step" for width in regression.WIDTHS]
                + ["partial_4k_cold_prefill_gpu_ms", "partial_4k_hit_ttft_ms"]
            ),
        )
        # GPU time per decode step, not per request.
        self.assertEqual(summary["speed"]["decode_B2_gpu_ms_per_step"]["baseline"], 20)
        self.assertEqual(summary["acceptance"]["B1"]["candidate"], 40 / 64)

    def test_speed_regressions_fail_per_metric(self):
        def slower(field, factor, scenario=None, width=None):
            def change(document):
                if width:
                    for sample in decode_samples(document, width):
                        sample[field] *= factor
                for measurement in document["measurements"]:
                    if measurement["scenario"] == scenario:
                        measurement[field] *= factor

            return change

        for name, change in {
            "decode_B2_gpu_ms_per_step": slower("decode_gpu_ms", 1.05, width=2),
            "partial_4k_cold_prefill_gpu_ms": slower(
                "prefill_gpu_ms", 1.05, "partial_4k_cold"
            ),
            "partial_4k_hit_ttft_ms": slower("ttft_ms", 1.05, "partial_4k_hit"),
        }.items():
            with self.subTest(metric=name):
                summary = regression.summarize(rounds({1: change, 2: change}), False)
                failed = [f for f in summary["failures"] if f.startswith(name)]
                self.assertEqual(len(failed), 1, summary["failures"])
                self.assertEqual(summary["speed"][name]["verdict"], "fail")
                self.assertFalse(summary["pass"])
        # One slow baseline round: too noisy to decide, which also fails.
        summary = regression.summarize(
            rounds({3: slower("decode_gpu_ms", 1.2, width=1)}), False
        )
        self.assertEqual(
            summary["speed"]["decode_B1_gpu_ms_per_step"]["verdict"], "inconclusive"
        )
        self.assertFalse(summary["pass"])

    def test_outputs_and_acceptance_must_match_unless_a_change_is_expected(self):
        def changed_output(document):
            for sample in decode_samples(document, 3):
                sample["output_token_hash"] = "other"

        def lower_acceptance(accepted):
            def change(document):
                for sample in document["decode_throughput"]["samples"]:
                    sample["accepted_draft_tokens"] = accepted
                for measurement in document["measurements"]:
                    measurement["output_tokens"] = [8]

            return change

        summary = regression.summarize(
            rounds({1: changed_output, 2: changed_output}), False
        )
        self.assertTrue(
            any("outputs or acceptance differ" in f for f in summary["failures"])
        )
        self.assertTrue(
            regression.summarize(rounds({1: changed_output, 2: changed_output}), True)[
                "pass"
            ]
        )
        # Within 0.02 of the baseline's acceptance rate (40/64 = 0.625).
        self.assertTrue(
            regression.summarize(
                rounds({1: lower_acceptance(39), 2: lower_acceptance(39)}), True
            )["pass"]
        )
        summary = regression.summarize(
            rounds({1: lower_acceptance(38), 2: lower_acceptance(38)}), True
        )
        self.assertEqual(
            len([f for f in summary["failures"] if "acceptance fell" in f]), 4
        )
        # A build must repeat itself even when outputs may change.
        for index in (1, 3):
            with self.subTest(round=index):
                summary = regression.summarize(rounds({index: changed_output}), True)
                self.assertTrue(any("did not repeat" in f for f in summary["failures"]))

    def test_identity_is_compared_within_each_build(self):
        # Different preparation identities load different keys: allowed.
        self.assertTrue(regression.summarize(rounds(), False)["pass"])

        def relayout(document):
            document["identity"]["loaded_model_layout_sha256"] = "other"

        def rebuild(document):
            document["build_id"] = "other"

        def other_device(document):
            document["identity"]["device"] = "Apple M3 Max"

        for index, change, message in (
            (2, relayout, "changed between rounds"),
            (3, rebuild, "changed between rounds"),
            (1, other_device, "different models or devices"),
        ):
            with self.subTest(change=change.__name__):
                summary = regression.summarize(rounds({index: change}), False)
                self.assertTrue(
                    any(message in f for f in summary["failures"]), summary["failures"]
                )
        with self.assertRaisesRegex(regression.RegressionError, "ABBA order"):
            first, second, third, fourth = rounds()
            regression.summarize([first, fourth, second, third], False)

    def test_a_baseline_with_the_candidates_build_id_fails(self):
        # The candidate's checkout, or another build of its sources: the ABBA
        # of identical builds passes every other rule.
        def candidate_build(document):
            document["build_id"] = "candidate"

        summary = regression.summarize(
            rounds({0: candidate_build, 3: candidate_build}), False
        )
        self.assertEqual(
            summary["failures"], ["the baseline has the candidate's build_id candidate"]
        )
        self.assertFalse(summary["pass"])

    def test_candidate_invariants_fail_and_baseline_ones_are_recorded(self):
        def fails(document):
            document["performance_pass"] = False
            document["performance_failures"] = [
                "B3 aggregate decode throughput fell below B2"
            ]

        summary = regression.summarize(rounds({2: fails}), False)
        self.assertTrue(any("invariants failed" in f for f in summary["failures"]))
        summary = regression.summarize(rounds({0: fails}), False)
        self.assertTrue(summary["pass"])
        self.assertEqual(len(summary["baseline_performance_failures"]), 1)

    def test_benchmark_output_parsing(self):
        passing = benchmark_document()
        failing = {**passing, "performance_pass": False, "performance_failures": ["x"]}
        self.assertEqual(regression.parse_document(json.dumps(passing), 0), passing)
        self.assertEqual(regression.parse_document(json.dumps(failing), 1), failing)
        for stdout, code in (
            ("", 1),
            ("{", 0),
            (json.dumps({"schema_version": 2}), 0),
            (json.dumps(failing), 0),
            (json.dumps(passing), 1),
            (json.dumps(passing), 2),
            (json.dumps(passing), -9),
        ):
            with self.subTest(stdout=stdout[:20], code=code):
                with self.assertRaises(regression.RegressionError):
                    regression.parse_document(stdout, code)
        # Separate decode and partial loads of an older build form one round.
        record = regression.round_record(
            "baseline",
            [benchmark_document(("decode",)), benchmark_document(("partial",))],
        )
        self.assertEqual(len(record["decode"][4]), 2)
        self.assertEqual(len(record["partial"]["partial_4k_hit"]), 2)
        self.assertEqual(len(record["identities"]), 2)
        with self.assertRaisesRegex(regression.RegressionError, "lacks"):
            regression.round_record("baseline", [benchmark_document(("decode",))])

    def fake_checkout(
        self,
        root: Path,
        name: str,
        identity: str,
        list_support: bool,
        share=None,
        honours_share=True,
    ):
        """A checkout whose backend-benchmark prints canned output and logs
        its invocations. With a share it has the Neural Engine split: it
        reports that share as calibrated, or runs the one --ane-ffn-share
        gives unless honours_share is false, and logs what it was given."""
        checkout = root / name
        (checkout / "build/engine-tests").mkdir(parents=True)
        (checkout / "build/engine").mkdir()
        (checkout / "build/splash.metallib").write_text("")
        (checkout / "build/engine/WeightPreparationIdentity.hpp").write_text(identity)
        usage = (
            "[--scenario NAME[,NAME...]]"
            if list_support
            else "[--scenario decode|partial]"
        ) + (" [--ane-ffn-share SHARE]" if share is not None else "")
        given = (
            "given = sys.argv[sys.argv.index('--ane-ffn-share') + 1] "
            "if '--ane-ffn-share' in sys.argv else None\n"
            f"with open({str(root / 'shares.jsonl')!r}, 'a') as log:\n"
            f"    log.write(json.dumps([{name!r}, given]) + '\\n')\n"
            "document['ane_ffn_share'] = "
            + ("float(given) if given else " if honours_share else "")
            + f"{share!r}\n"
            if share is not None
            else ""
        )
        script = checkout / regression.BENCHMARK
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "if len(sys.argv) < 3:\n"
            f"    print('usage: backend-benchmark {usage}', file=sys.stderr)\n"
            "    raise SystemExit(2)\n"
            "scenarios = sys.argv[sys.argv.index('--scenario') + 1].split(',')\n"
            f"with open({str(root / 'calls.jsonl')!r}, 'a') as log:\n"
            f"    log.write(json.dumps([{name!r}, scenarios, os.environ.get('SPLASH_WEIGHT_CACHE')]) + '\\n')\n"
            f"document = json.loads({json.dumps(json.dumps(benchmark_document(build=name, layout=identity)))})\n"
            "if 'decode' not in scenarios: document['decode_throughput']['samples'] = []\n"
            "if 'partial' not in scenarios: document['measurements'] = []\n"
            + given
            + "print(json.dumps(document))\n"
        )
        script.chmod(0o755)
        return checkout

    @staticmethod
    def run_main(root: Path) -> int:
        """main on the fake checkouts under root and a legacy package."""
        models = root / "models"
        package = models / "incoai/Qwen3.8-27B-Splash"
        package.mkdir(parents=True)
        (package / "manifest.json").write_text("{}")
        arguments = [
            "--baseline",
            str(root / "baseline"),
            "--candidate",
            str(root / "candidate"),
            "--package",
            str(package),
            "--output-dir",
            str(root / "release"),
        ]
        with (
            mock.patch.object(smoke.model_artifacts, "MODELS", models),
            mock.patch.dict(os.environ, {"SPLASH_WEIGHT_CACHE": str(root / "cache")}),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return regression.main(arguments)

    def test_a_failed_benchmark_stops_the_comparison_with_its_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "same", True)
            candidate = self.fake_checkout(root, "candidate", "same", True)
            script = candidate / regression.BENCHMARK
            script.write_text(
                script.read_text().replace(
                    "print(json.dumps(document))\n",
                    "print('backend-benchmark: request failed: boom', file=sys.stderr)\n"
                    "raise SystemExit(1)\n",
                )
            )
            with self.assertRaisesRegex(
                regression.RegressionError,
                r"round-2-candidate-decode-partial: .*without JSON(.|\n)*boom",
            ):
                self.run_main(root)
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertFalse(document["pass"])
            self.assertIn("without JSON", document["error"])
            # The baseline's first round ran; nothing after the failure did.
            calls = (root / "calls.jsonl").read_text().splitlines()
            self.assertEqual(
                [json.loads(line)[0] for line in calls], ["baseline", "candidate"]
            )

    def test_main_runs_abba_rounds_and_isolates_another_preparation_identity(self):
        for identity, list_support in (("same", True), ("new", True), ("new", False)):
            with (
                self.subTest(identity=identity, list_support=list_support),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                self.fake_checkout(root, "baseline", "same", list_support)
                self.fake_checkout(root, "candidate", identity, True)
                output = root / "release"
                self.assertEqual(self.run_main(root), 0)
                calls = [
                    json.loads(line)
                    for line in (root / "calls.jsonl").read_text().splitlines()
                ]
                scenarios = (
                    [["decode", "partial"]]
                    if list_support
                    else [["decode"], ["partial"]]
                )
                self.assertEqual(
                    calls,
                    [
                        [
                            version,
                            names,
                            # The candidate keeps the cache of its other steps.
                            str(root / "cache")
                            if version == "candidate" or identity == "same"
                            else str(output / "baseline-weights"),
                        ]
                        for version in regression.ROUNDS
                        for names in scenarios
                    ],
                )
                document = json.loads((output / "backend-regression.json").read_text())
                self.assertTrue(document["pass"])
                self.assertEqual(
                    document["prepared"]["shared_identity"], identity == "same"
                )
                # A legacy package prepares nothing, which is no failure.
                self.assertEqual(document["prepared"].get("entries", []), [])

    def test_later_rounds_run_the_first_reported_ane_ffn_share(self):
        calibrated = 1 - 24 / 34
        # A baseline without the split leaves the candidate's first round to
        # calibrate; one with it calibrates first and the candidate runs its
        # share.
        for baseline_share, expected in (
            (None, [["candidate", None], ["candidate", repr(0.32)]]),
            (
                calibrated,
                [
                    ["baseline", None],
                    ["candidate", repr(calibrated)],
                    ["candidate", repr(calibrated)],
                    ["baseline", repr(calibrated)],
                ],
            ),
        ):
            with (
                self.subTest(baseline_share=baseline_share),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                self.fake_checkout(root, "baseline", "same", True, baseline_share)
                self.fake_checkout(root, "candidate", "same", True, 0.32)
                self.assertEqual(self.run_main(root), 0)
                shares = [
                    json.loads(line)
                    for line in (root / "shares.jsonl").read_text().splitlines()
                ]
                self.assertEqual(shares, expected)
                document = json.loads(
                    (root / "release/backend-regression.json").read_text()
                )
                self.assertEqual(
                    document["ane_ffn_share"],
                    0.32 if baseline_share is None else calibrated,
                )

    def test_a_round_that_runs_another_ane_ffn_share_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "same", True, 0.3)
            self.fake_checkout(
                root, "candidate", "same", True, 0.0, honours_share=False
            )
            with self.assertRaisesRegex(
                regression.RegressionError,
                r"round-2-candidate-decode-partial: ran the Neural Engine split "
                r"at share 0.0, not the first round's 0.3",
            ):
                self.run_main(root)

    def test_package_slug_names_results_by_selection(self):
        models = Path("/install/models")
        with mock.patch.object(smoke.model_artifacts, "MODELS", models):
            self.assertEqual(
                regression.package_slug(models / "mlx-community/Qwen3.8-27B-4bit"),
                "mlx-community--Qwen3.8-27B-4bit",
            )
            self.assertEqual(
                regression.package_slug(models / ".selections/abc"), ".selections--abc"
            )
            self.assertEqual(regression.package_slug(Path("/elsewhere/pkg")), "pkg")


if __name__ == "__main__":
    unittest.main()
