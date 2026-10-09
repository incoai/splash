import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from dev.benchmarks import backend_regression as regression
from dev.benchmarks import weights
from dev.tests import smoke_real as smoke

CONTEXT = 262144


def benchmark_document(
    scenarios=regression.SCENARIOS, build="build", layout="layout", step=10.0
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
    requests = [
        (scenario, 14096) for scenario in regression.PARTIAL if "partial" in scenarios
    ] + [("short", rows + 1) for rows in regression.SHORT if "short" in scenarios]
    measurements = [
        {
            "scenario": scenario,
            "sample": sample,
            "prompt_tokens": tokens,
            "output_tokens": [7],
            "prefill_gpu_ms": 1000.0,
            "ttft_ms": 300.0,
        }
        for sample in range(2)
        for scenario, tokens in requests
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
        "max_context_tokens": CONTEXT,
        "no_ane_context_tokens": CONTEXT,
        "dynamic_budget_bytes": 35 << 30,
        "ane_ffn_bytes": 0,
        "measurements": measurements,
        "performance_pass": True,
        "performance_failures": [],
    }


def rounds(changes=None, scenarios=regression.SCENARIOS):
    """Four ABBA round records of these scenarios; changes maps a round
    index to a function that edits that round's benchmark document."""
    changes = changes or {}
    result = []
    for index, version in enumerate(regression.ROUNDS):
        document = benchmark_document(
            scenarios, build=version, layout=f"layout-{version}"
        )
        if index in changes:
            changes[index](document)
        result.append(regression.round_record(version, [document], scenarios))
    return result


def decode_samples(document, width):
    return [s for s in document["decode_throughput"]["samples"] if s["width"] == width]


class BackendRegressionTests(unittest.TestCase):
    def test_identical_rounds_pass_every_rule(self):
        summary = regression.summarize(rounds(), False)
        self.assertEqual(summary["failures"], [])
        self.assertTrue(summary["pass"])
        decode_and_partial = [
            f"decode_B{width}_gpu_ms_per_step" for width in regression.WIDTHS
        ] + ["partial_4k_cold_prefill_gpu_ms", "partial_4k_hit_ttft_ms"]
        self.assertEqual(
            sorted(summary["speed"]),
            sorted(
                decode_and_partial
                + [f"short_{rows}_rows_prefill_gpu_ms" for rows in regression.SHORT]
            ),
        )
        # GPU time per decode step, not per request.
        self.assertEqual(summary["speed"]["decode_B2_gpu_ms_per_step"]["baseline"], 20)
        self.assertEqual(summary["acceptance"]["B1"]["candidate"], 40 / 64)
        # Rounds against a build without the short scenario compare the rest.
        summary = regression.summarize(rounds(scenarios=("decode", "partial")), False)
        self.assertTrue(summary["pass"])
        self.assertEqual(sorted(summary["speed"]), sorted(decode_and_partial))

    def test_speed_regressions_fail_per_metric(self):
        def slower(field, factor, scenario=None, width=None, tokens=None):
            def change(document):
                if width:
                    for sample in decode_samples(document, width):
                        sample[field] *= factor
                for measurement in document["measurements"]:
                    if measurement["scenario"] == scenario and tokens in (
                        None,
                        measurement["prompt_tokens"],
                    ):
                        measurement[field] *= factor

            return change

        for name, change in {
            "decode_B2_gpu_ms_per_step": slower("decode_gpu_ms", 1.05, width=2),
            "partial_4k_cold_prefill_gpu_ms": slower(
                "prefill_gpu_ms", 1.05, "partial_4k_cold"
            ),
            "partial_4k_hit_ttft_ms": slower("ttft_ms", 1.05, "partial_4k_hit"),
            # A prompt of a 512-row chunk 2.5% slower: beyond the 2% floor.
            "short_512_rows_prefill_gpu_ms": slower(
                "prefill_gpu_ms", 1.025, "short", tokens=513
            ),
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
        # Two builds may load different layouts of one model: allowed.
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
        # Separate loads, one per scenario, form one round.
        record = regression.round_record(
            "baseline",
            [benchmark_document((name,)) for name in regression.SCENARIOS],
            regression.SCENARIOS,
        )
        self.assertEqual(len(record["decode"][4]), 2)
        self.assertEqual(len(record["partial"]["partial_4k_hit"]), 2)
        self.assertEqual(len(record["short"][2016]), 2)
        self.assertEqual(len(record["identities"]), 3)
        self.assertEqual(
            record["memory_plans"][0],
            {
                "max_context_tokens": CONTEXT,
                "no_ane_context_tokens": CONTEXT,
                "dynamic_budget_bytes": 35 << 30,
                "ane_ffn_bytes": 0,
            },
        )
        for documents, scenarios, lacking in (
            ([("decode",)], ("decode", "partial"), "lacks partial samples"),
            ([("decode", "partial")], regression.SCENARIOS, "lacks short samples"),
        ):
            with self.assertRaisesRegex(regression.RegressionError, lacking):
                regression.round_record(
                    "baseline",
                    [benchmark_document(names) for names in documents],
                    scenarios,
                )

    def fake_checkout(
        self,
        root: Path,
        name: str,
        digest: str,
        list_support: bool,
        share=None,
        honours_share=True,
        short=True,
        context=CONTEXT,
        minimum_rows=832,
        earlier=False,
    ):
        """A checkout whose backend-benchmark prints canned output and logs
        its invocations. Its weight-digests prints one image of digest; a
        build of an earlier release (earlier) has none, and its benchmark
        prepares that image into its weight cache instead. Its benchmark takes a list of scenarios with list_support, the short
        scenario among them with short too, and serves context tokens. With a
        share it has the Neural Engine split: it reports that share and
        minimum_rows as calibrated, or runs the share --ane-ffn-share gives
        and the rows --ane-ffn-minimum-rows gives, else 512, unless
        honours_share is false, and logs what it was given."""
        checkout = root / name
        (checkout / "build/engine-tests").mkdir(parents=True)
        (checkout / "build/splash.metallib").write_text("")
        if earlier:
            (checkout / "build" / weights.IDENTITY_HEADER).parent.mkdir()
            (checkout / "build" / weights.IDENTITY_HEADER).write_text("")
        else:
            tool = checkout / "build" / weights.WEIGHT_DIGESTS
            image = {"component": "target/layer-0.bin", "bytes": 1, "sha256": digest}
            tool.write_text(f"#!/bin/sh\necho '{json.dumps([image])}'\n")
            tool.chmod(0o755)
        usage = (
            "[--scenario NAME[,NAME...]]"
            if list_support
            else "[--scenario decode|partial]"
        ) + (" [--ane-ffn-share SHARE]" if share is not None else "")
        if share is not None:
            usage += " [--ane-ffn-minimum-rows ROWS]"
        if list_support:
            names = "decode, partial, short" if short else "decode, partial"
            usage += f"\n  NAME: {names}, context or exact"
        document = benchmark_document(build=name)
        document["max_context_tokens"] = document["no_ane_context_tokens"] = context
        given = (
            "option = lambda name: sys.argv[sys.argv.index(name) + 1] "
            "if name in sys.argv else None\n"
            "given, rows = option('--ane-ffn-share'), option('--ane-ffn-minimum-rows')\n"
            f"with open({str(root / 'shares.jsonl')!r}, 'a') as log:\n"
            f"    log.write(json.dumps([{name!r}, given, rows]) + '\\n')\n"
            "document['ane_ffn_share'] = "
            + ("float(given) if given else " if honours_share else "")
            + f"{share!r}\n"
            + "document['ane_ffn_minimum_rows'] = 0 if not document['ane_ffn_share'] "
            + ("else int(rows) if rows else 512 if given " if honours_share else "")
            + f"else {minimum_rows!r}\n"
            if share is not None
            else ""
        )
        # The cache entry weights.prepared reads: the image's digest and its
        # provenance, under the model root the benchmark was given.
        source = [weights.PROVENANCE, "component target/layer-0.bin"]
        prepare = (
            "entry = pathlib.Path(os.environ['SPLASH_WEIGHT_CACHE']) / ('1' * 64)\n"
            "entry.mkdir(exist_ok=True)\n"
            f"(entry / 'sha256').write_text({digest!r})\n"
            f"lines = {source!r} + ['source ' + sys.argv[2] + '/target']\n"
            "(entry / 'source').write_text(''.join(line + '\\n' for line in lines))\n"
            if earlier
            else ""
        )
        script = checkout / regression.BENCHMARK
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "if len(sys.argv) < 3:\n"
            f"    print('usage: backend-benchmark ' + {usage!r}, file=sys.stderr)\n"
            "    raise SystemExit(2)\n"
            "scenarios = sys.argv[sys.argv.index('--scenario') + 1].split(',')\n"
            f"with open({str(root / 'calls.jsonl')!r}, 'a') as log:\n"
            f"    log.write(json.dumps([{name!r}, scenarios, os.environ.get('SPLASH_WEIGHT_CACHE')]) + '\\n')\n"
            f"document = json.loads({json.dumps(json.dumps(document))})\n"
            "if 'decode' not in scenarios: document['decode_throughput']['samples'] = []\n"
            "document['measurements'] = [m for m in document['measurements'] if "
            "('short' if m['scenario'] == 'short' else 'partial') in scenarios]\n"
            + given
            + prepare
            + "print(json.dumps(document))\n"
        )
        script.chmod(0o755)
        return checkout

    @staticmethod
    def run_main(root: Path, *options: str) -> int:
        """main with these options on the fake checkouts under root and an
        installed model."""
        models = root / "models"
        assembly = models / ".resolved/assembly"
        assembly.mkdir(parents=True)
        (assembly / "model.json").write_text("{}")
        link = models / "mlx-community/Qwen3.8-27B-4bit"
        link.parent.mkdir()
        link.symlink_to(assembly, target_is_directory=True)
        arguments = [
            "--baseline",
            str(root / "baseline"),
            "--candidate",
            str(root / "candidate"),
            "--model-root",
            str(link),
            "--output-dir",
            str(root / "release"),
            *options,
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
            self.fake_checkout(root, "baseline", "a" * 64, True)
            candidate = self.fake_checkout(root, "candidate", "a" * 64, True)
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
                r"round-2-candidate-decode-partial-short: .*without JSON(.|\n)*boom",
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

    def test_main_runs_abba_rounds_and_compares_the_weights(self):
        a, b = "a" * 64, "b" * 64
        # A baseline without the short scenario runs the others alone, and so
        # does the candidate.
        for earlier, list_support, short, scenarios in (
            (False, True, True, [["decode", "partial", "short"]]),
            (False, True, False, [["decode", "partial"]]),
            (True, True, True, [["decode", "partial", "short"]]),
            (True, False, False, [["decode"], ["partial"]]),
        ):
            with (
                self.subTest(earlier=earlier, list_support=list_support, short=short),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                self.fake_checkout(
                    root, "baseline", a, list_support, short=short, earlier=earlier
                )
                self.fake_checkout(root, "candidate", a, True)
                output = root / "release"
                self.assertEqual(self.run_main(root), 0)
                calls = [
                    json.loads(line)
                    for line in (root / "calls.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    calls,
                    [
                        [
                            version,
                            names,
                            # Only a baseline of an earlier release gets a
                            # cache of its own.
                            str(output / "baseline-weights")
                            if version == "baseline" and earlier
                            else str(root / "cache"),
                        ]
                        for version in regression.ROUNDS
                        for names in scenarios
                    ],
                )
                document = json.loads((output / "backend-regression.json").read_text())
                self.assertTrue(document["pass"])
                # An earlier release's image is read from its cache.
                self.assertEqual(len(document["weights"]["images"]), 1)

        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", a, True)
            self.fake_checkout(root, "candidate", b, True)
            self.assertEqual(self.run_main(root), 1)
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertFalse(document["pass"])
            self.assertEqual(
                document["weights"]["failures"],
                [f"target/layer-0.bin: the baseline loaded {a}, the candidate {b}"],
            )

        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", a, True)
            self.fake_checkout(root, "candidate", a, True, earlier=True)
            with self.assertRaises(SystemExit):
                self.run_main(root)

    def test_later_rounds_run_the_first_reported_ane_ffn_split(self):
        calibrated = 1 - 24 / 34
        # A baseline without the split leaves the candidate's first round to
        # calibrate; one with it calibrates first and the candidate runs its
        # share and least chunk rows.
        for baseline_share, baseline_rows, expected in (
            (None, 832, [["candidate", None, None], ["candidate", repr(0.32), "896"]]),
            (
                calibrated,
                832,
                [
                    ["baseline", None, None],
                    ["candidate", repr(calibrated), "832"],
                    ["candidate", repr(calibrated), "832"],
                    ["baseline", repr(calibrated), "832"],
                ],
            ),
        ):
            with (
                self.subTest(baseline_share=baseline_share),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                self.fake_checkout(
                    root,
                    "baseline",
                    "a" * 64,
                    True,
                    baseline_share,
                    minimum_rows=baseline_rows,
                )
                self.fake_checkout(
                    root, "candidate", "a" * 64, True, 0.32, minimum_rows=896
                )
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
                    (document["ane_ffn_share"], document["ane_ffn_minimum_rows"]),
                    (0.32, 896) if baseline_share is None else (calibrated, 832),
                )
        args = SimpleNamespace(
            ane_ffn_share=0.3,
            ane_ffn_minimum_rows=832,
            ane_ffn_share_given=False,
        )
        regression.pin_ane_ffn(
            args, "candidate", {"ane_ffn_share": 0.3, "ane_ffn_minimum_rows": 832}
        )
        with self.assertRaisesRegex(
            regression.RegressionError,
            "at share 0.3 from 512 rows, not the first round's 0.3 from 832",
        ):
            regression.pin_ane_ffn(
                args, "candidate", {"ane_ffn_share": 0.3, "ane_ffn_minimum_rows": 512}
            )

    def test_a_first_round_without_the_split_pins_no_share(self):
        # A candidate whose first round ran no split runs every later round
        # as it is, not as --ane-ffn-share 0, which would serve another
        # context.
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64, True)
            self.fake_checkout(root, "candidate", "a" * 64, True, 0.0, minimum_rows=0)
            self.assertEqual(self.run_main(root), 0)
            shares = [
                json.loads(line)
                for line in (root / "shares.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                shares, [["candidate", None, None], ["candidate", None, None]]
            )

    def test_a_round_that_runs_another_ane_ffn_share_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64, True, 0.3)
            self.fake_checkout(
                root, "candidate", "a" * 64, True, 0.0, honours_share=False
            )
            with self.assertRaisesRegex(
                regression.RegressionError,
                r"round-2-candidate-decode-partial-short: ran the Neural Engine split "
                r"at share 0.0 from 0 rows, not the first round's 0.3 from 832",
            ):
                self.run_main(root)

    def test_a_given_ane_ffn_share_runs_in_every_round_that_takes_it(self):
        # Share 0, the GPU alone, against a baseline without the split, which
        # runs as it is and reports no share, and against one with it.
        for baseline_share, expected in (
            (None, [["candidate", "0.0", None], ["candidate", "0.0", None]]),
            (1 - 24 / 34, [[version, "0.0", None] for version in regression.ROUNDS]),
        ):
            with (
                self.subTest(baseline_share=baseline_share),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                self.fake_checkout(root, "baseline", "a" * 64, True, baseline_share)
                self.fake_checkout(root, "candidate", "a" * 64, True, 0.32)
                self.assertEqual(self.run_main(root, "--ane-ffn-share", "0"), 0)
                shares = [
                    json.loads(line)
                    for line in (root / "shares.jsonl").read_text().splitlines()
                ]
                self.assertEqual(shares, expected)
                document = json.loads(
                    (root / "release/backend-regression.json").read_text()
                )
                self.assertEqual(document["ane_ffn_share"], 0.0)

    def test_a_reported_ane_ffn_share_must_be_the_given_one(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64, True)
            self.fake_checkout(
                root, "candidate", "a" * 64, True, 0.32, honours_share=False
            )
            with self.assertRaisesRegex(
                regression.RegressionError,
                r"round-2-candidate-decode-partial-short: ran the Neural Engine split "
                r"at share 0.32, not the given 0.0",
            ):
                self.run_main(root, "--ane-ffn-share", "0")
        # A build that takes no --ane-ffn-share runs without the given share
        # and must report none.
        args = SimpleNamespace(
            ane_ffn_share=0.0,
            ane_ffn_share_given=True,
            takes_ane_ffn_share={"baseline": False},
        )
        regression.pin_ane_ffn(args, "baseline", {})
        with self.assertRaisesRegex(
            regression.RegressionError,
            "takes no --ane-ffn-share but reported the share 0.24",
        ):
            regression.pin_ane_ffn(args, "baseline", {"ane_ffn_share": 0.24})

    def test_an_ane_ffn_share_outside_zero_to_one_is_refused(self):
        for value in ("1", "1.5", "-0.1", "nan", "inf", "half"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit),
            ):
                regression.parse_args(
                    ["--baseline", "b", "--model-root", "m", "--ane-ffn-share", value]
                )
            self.assertIn("argument --ane-ffn-share", errors.getvalue())
        self.assertEqual(regression.ane_ffn_share("0"), 0.0)
        self.assertEqual(regression.ane_ffn_share("0.999"), 0.999)

    def test_the_candidate_must_hold_the_baselines_context(self):
        def serves(tokens, budget, without=CONTEXT):
            def change(document):
                document["max_context_tokens"] = tokens
                document["no_ane_context_tokens"] = without
                document["dynamic_budget_bytes"] = budget
                document["ane_ffn_bytes"] = (35 << 30) - budget

            return change

        # A split of 173 MiB that takes 4K tokens of context from the KV
        # cache passes, and is reported.
        split = serves(CONTEXT - 4096, (35 << 30) - (173 << 20))
        summary = regression.summarize(rounds({1: split, 2: split}), False)
        self.assertTrue(summary["pass"], summary["failures"])
        self.assertEqual(
            summary["memory_plan"]["candidate"]["max_context_tokens"], CONTEXT - 4096
        )
        self.assertEqual(
            summary["memory_plan"]["dynamic_budget_change_bytes"], -(173 << 20)
        )
        self.assertEqual(
            summary["memory_plan"]["candidate"]["ane_ffn_bytes"], 173 << 20
        )
        # A plan that holds less context without the split fails.
        smaller = serves(CONTEXT - 4096, 35 << 30, CONTEXT - 4096)
        summary = regression.summarize(rounds({1: smaller, 2: smaller}), False)
        self.assertEqual(
            summary["failures"],
            [
                f"the candidate holds {CONTEXT - 4096} tokens of context without "
                f"the Neural Engine split, less than the baseline's {CONTEXT}"
            ],
        )

        # A baseline that reports no memory plan is not compared, and one
        # before the split is compared by the context it serves.
        def unreported(document):
            for key in regression.MEMORY_PLAN:
                del document[key]

        def before_split(document):
            del document["no_ane_context_tokens"]

        summary = regression.summarize(
            rounds({0: unreported, 3: unreported, 1: smaller, 2: smaller}), False
        )
        self.assertTrue(summary["pass"])
        self.assertIsNone(summary["memory_plan"]["dynamic_budget_change_bytes"])
        summary = regression.summarize(
            rounds({0: before_split, 3: before_split, 1: smaller, 2: smaller}), False
        )
        self.assertEqual(len(summary["failures"]), 1)
        # A build's plan must not change between its rounds.
        summary = regression.summarize(rounds({2: split}), False)
        self.assertEqual(
            summary["failures"],
            ["the candidate build's memory plan changed between rounds"],
        )
        # main fails on the loss too.
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64, True)
            self.fake_checkout(root, "candidate", "a" * 64, True, context=CONTEXT - 1)
            self.assertEqual(self.run_main(root), 1)
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertEqual(
                document["comparison"]["failures"],
                [
                    f"the candidate holds {CONTEXT - 1} tokens of context without "
                    f"the Neural Engine split, less than the baseline's {CONTEXT}"
                ],
            )

    def test_results_are_named_by_selection(self):
        models = Path("/install/models")
        with mock.patch.object(smoke.model_artifacts, "MODELS", models):
            self.assertEqual(
                regression.results_slug(models / "mlx-community/Qwen3.8-27B-4bit"),
                "mlx-community--Qwen3.8-27B-4bit",
            )
            self.assertEqual(
                regression.results_slug(models / ".selections/abc"), ".selections--abc"
            )
            self.assertEqual(regression.results_slug(Path("/elsewhere/model")), "model")


if __name__ == "__main__":
    unittest.main()
