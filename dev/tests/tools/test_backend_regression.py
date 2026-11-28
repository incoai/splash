import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from dev.benchmarks import backend_regression as regression
from dev.benchmarks import chat_prompts, weights
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
            "decode_output_tokens": 64 * width,
            "decode_wall_ms": 1.25 * 64 * width,
        }
        for sample in range(2)
        for width in regression.WIDTHS
    ]
    requests = (
        [(scenario, 14096) for scenario in regression.PARTIAL if "partial" in scenarios]
        # A short prompt holds its chunk's rows, as backend-benchmark sends it.
        + [("short", rows) for rows in regression.SHORT if "short" in scenarios]
        + [(name, 40) for name in regression.TRANSCRIPTS if "transcripts" in scenarios]
    )
    measurements = [
        {
            "scenario": scenario,
            "sample": sample,
            "prompt_tokens": tokens,
            "output_tokens": [7],
            "prefill_gpu_ms": 1000.0,
            "ttft_ms": 300.0,
            "wall_ms": 320.0,
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
        # The share and least chunk rows of a load that runs no split.
        "ane_ffn_share": 0.0,
        "ane_ffn_minimum_rows": 0,
        "max_context_tokens": CONTEXT,
        "no_ane_context_tokens": CONTEXT,
        "dynamic_budget_bytes": 35 << 30,
        "ane_ffn_bytes": 0,
        "measurements": measurements,
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
        result.append(regression.round_record(version, document))
    return result


def decode_samples(document, width):
    return [s for s in document["decode_throughput"]["samples"] if s["width"] == width]


class BackendRegressionTests(unittest.TestCase):
    def test_identical_rounds_pass_every_rule(self):
        summary = regression.summarize(rounds(), False)
        self.assertEqual(summary["failures"], [])
        self.assertTrue(summary["pass"])
        decode_and_partial = [
            f"decode_B{width}_{metric}"
            for width in regression.WIDTHS
            for metric in ("gpu_ms_per_step", "wall_ms_per_token")
        ] + [
            "partial_4k_cold_prefill_gpu_ms",
            "partial_4k_cold_wall_ms",
            "partial_4k_hit_ttft_ms",
        ]
        self.assertEqual(
            sorted(summary["speed"]),
            sorted(
                decode_and_partial
                + [
                    f"short_{rows}_rows_{metric}"
                    for rows in regression.SHORT
                    for metric in ("prefill_gpu_ms", "ttft_ms")
                ]
            ),
        )
        # GPU time per decode step, not per request.
        self.assertEqual(summary["speed"]["decode_B2_gpu_ms_per_step"]["baseline"], 20)
        # Wall time per output token: the decode phase over the lanes' tokens.
        self.assertEqual(
            summary["speed"]["decode_B2_wall_ms_per_token"]["baseline"], 1.25
        )
        self.assertEqual(summary["acceptance"]["B1"]["candidate"], 40 / 64)

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
            "decode_B3_wall_ms_per_token": slower("decode_wall_ms", 1.05, width=3),
            "partial_4k_cold_wall_ms": slower("wall_ms", 1.05, "partial_4k_cold"),
            "short_1024_rows_ttft_ms": slower("ttft_ms", 1.05, "short", tokens=1024),
            # A prompt of a 512-row chunk 2.5% slower: beyond the 2% floor.
            "short_512_rows_prefill_gpu_ms": slower(
                "prefill_gpu_ms", 1.025, "short", tokens=512
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

        def changed_transcript(document):
            for measurement in document["measurements"]:
                if measurement["scenario"] == "transcript_3":
                    measurement["output_tokens"] = [9]

        summary = regression.summarize(
            rounds({1: changed_output, 2: changed_output}), False
        )
        self.assertTrue(
            any("outputs or acceptance differ" in f for f in summary["failures"])
        )
        # A transcript is compared token for token, like the other requests.
        summary = regression.summarize(
            rounds({1: changed_transcript, 2: changed_transcript}), False
        )
        self.assertTrue(
            any(
                "outputs or acceptance differ" in f and "transcript_3 sample 0" in f
                for f in summary["failures"]
            )
        )
        self.assertTrue(
            regression.summarize(
                rounds({1: changed_transcript, 2: changed_transcript}), True
            )["pass"]
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
        # One load runs every scenario of a round.
        record = regression.round_record("baseline", benchmark_document())
        self.assertEqual(len(record["decode"][4]), 2)
        self.assertEqual(len(record["partial"]["partial_4k_hit"]), 2)
        self.assertEqual(len(record["short"][2016]), 2)
        self.assertEqual(len(record["transcripts"]["transcript_7"]), 2)
        self.assertEqual(record["identity"]["build_id"], "build")
        self.assertEqual(
            record["memory_plan"],
            {
                "max_context_tokens": CONTEXT,
                "no_ane_context_tokens": CONTEXT,
                "dynamic_budget_bytes": 35 << 30,
                "ane_ffn_bytes": 0,
            },
        )
        for scenarios, lacking in (
            (("decode",), "lacks partial and short and transcripts samples"),
            (("decode", "partial"), "lacks short and transcripts samples"),
        ):
            with self.assertRaisesRegex(regression.RegressionError, lacking):
                regression.round_record("baseline", benchmark_document(scenarios))

    def fake_checkout(
        self,
        root: Path,
        name: str,
        digest: str,
        share=0.0,
        honours_share=True,
        context=CONTEXT,
        minimum_rows=832,
        units=None,
        two_steps=None,
    ):
        """A checkout whose backend-benchmark prints canned output of every
        scenario and logs the scenarios it was given, and whose
        weight-digests prints one image of digest and logs the arguments
        after its library in digests.log. Its benchmark serves context tokens
        and has the Neural Engine split: it reports share and minimum_rows as
        calibrated, no rows for share 0, or runs the share --ane-ffn-share
        gives, in whole units of `units` where given, and the rows
        --ane-ffn-minimum-rows gives, else 512, unless honours_share is
        false, and logs what it was given. Its start logs its split: a
        calibrated one of `two_steps`, a share in two steps from 1024 rows,
        reports the larger share, as RuntimeResources::aneFfnShare."""
        checkout = root / name
        (checkout / "build/engine-tests").mkdir(parents=True)
        (checkout / "build/splash.metallib").write_text("")
        tool = checkout / "build" / weights.WEIGHT_DIGESTS
        image = {"component": "target/layer-0.bin", "bytes": 1, "sha256": digest}
        tool.write_text(
            "#!/bin/sh\n"
            f"shift; echo {name} \"$@\" >> '{root / 'digests.log'}'\n"
            f"echo '{json.dumps([image])}'\n"
        )
        tool.chmod(0o755)
        document = benchmark_document(build=name)
        document["max_context_tokens"] = document["no_ane_context_tokens"] = context
        script = checkout / regression.BENCHMARK
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, sys\n"
            "scenarios = sys.argv[sys.argv.index('--scenario') + 1].split(',')\n"
            f"with open({str(root / 'calls.jsonl')!r}, 'a') as log:\n"
            f"    log.write(json.dumps([{name!r}, scenarios]) + '\\n')\n"
            "prompts = sys.argv[sys.argv.index('--prompts') + 1] if '--prompts' in sys.argv else None\n"
            f"with open({str(root / 'prompts.jsonl')!r}, 'a') as log:\n"
            f"    log.write(json.dumps([{name!r}, prompts]) + '\\n')\n"
            f"document = json.loads({json.dumps(json.dumps(document))})\n"
            "option = lambda name: sys.argv[sys.argv.index(name) + 1] "
            "if name in sys.argv else None\n"
            "given, rows = option('--ane-ffn-share'), option('--ane-ffn-minimum-rows')\n"
            f"with open({str(root / 'shares.jsonl')!r}, 'a') as log:\n"
            f"    log.write(json.dumps([{name!r}, given, rows]) + '\\n')\n"
            "document['ane_ffn_share'] = "
            + (
                (
                    f"round(float(given) * {units}) / {units} if given else "
                    if units
                    else "float(given) if given else "
                )
                if honours_share
                else ""
            )
            + f"max({share!r}, {two_steps or 0.0!r})\n"
            + "document['ane_ffn_minimum_rows'] = 0 if not document['ane_ffn_share'] "
            + ("else int(rows) if rows else 512 if given " if honours_share else "")
            + f"else {minimum_rows!r}\n"
            + "share, rows = document['ane_ffn_share'], document['ane_ffn_minimum_rows']\n"
            + "two = ' and %.2f in two steps from 1024 rows' % "
            + f"({two_steps!r} or share if not given else share)\n"
            + "print('12:00:00 Setting up the Neural Engine FFN split.', file=sys.stderr)\n"
            + "print('12:00:01 ' + ('The GPU runs the prefill FFN alone.' if not share else "
            "'Neural Engine FFN split at %s%.2f for chunks of %d rows or more%s (0.1 s).' % "
            + f"('the given share ' if given else 'share ', share if given else {share!r}, "
            "rows, two)), file=sys.stderr)\n"
            + "print('12:00:02 Runtime precision: the Neural Engine FFN split multiplies "
            "its share in int8.', file=sys.stderr)\n" + "print(json.dumps(document))\n"
        )
        script.chmod(0o755)
        return checkout

    @staticmethod
    def run_main(root: Path, *options: str, prompts=None) -> int:
        """main with these options on the fake checkouts under root and an
        installed model whose tokenizer and chat template give prompts, by
        default the benchmark's built-in ones."""
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
            mock.patch.object(
                chat_prompts,
                "model_prompts",
                return_value=chat_prompts.builtin() if prompts is None else prompts,
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return regression.main(arguments)

    def test_a_failed_benchmark_stops_the_comparison_with_its_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64)
            candidate = self.fake_checkout(root, "candidate", "a" * 64)
            script = candidate / regression.BENCHMARK
            script.write_text(
                script.read_text().replace(
                    "print(json.dumps(document))\n",
                    "if 'decode' in scenarios:\n"
                    "    print('backend-benchmark: request failed: boom', file=sys.stderr)\n"
                    "    raise SystemExit(1)\n"
                    "print(json.dumps(document))\n",
                )
            )
            with self.assertRaisesRegex(
                regression.RegressionError,
                r"round-2-candidate-decode-partial-short-transcripts: .*without JSON(.|\n)*boom",
            ):
                self.run_main(root)
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertFalse(document["pass"])
            self.assertIn("without JSON", document["error"])
            # The candidate's calibration start and the baseline's first
            # round ran; nothing after the failure did.
            calls = (root / "calls.jsonl").read_text().splitlines()
            self.assertEqual(
                [json.loads(line)[0] for line in calls],
                ["candidate", "baseline", "candidate"],
            )

    def test_main_runs_abba_rounds_and_compares_the_weights(self):
        a, b = "a" * 64, "b" * 64
        # Each round runs every scenario in one load.
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", a)
            self.fake_checkout(root, "candidate", a)
            output = root / "release"
            self.assertEqual(self.run_main(root), 0)
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                calls,
                [["candidate", ["short"]]]
                + [
                    [version, list(regression.SCENARIOS)]
                    for version in regression.ROUNDS
                ],
            )
            document = json.loads((output / "backend-regression.json").read_text())
            self.assertTrue(document["pass"])
            self.assertEqual(len(document["weights"]["images"]), 1)

        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", a)
            self.fake_checkout(root, "candidate", b)
            self.assertEqual(self.run_main(root), 1)
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertFalse(document["pass"])
            self.assertEqual(
                document["weights"]["failures"],
                [f"target/layer-0.bin: the baseline loaded {a}, the candidate {b}"],
            )

        # A change of the image layout reports the images it changed and
        # fails nothing.
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", a)
            self.fake_checkout(root, "candidate", b)
            self.assertEqual(self.run_main(root, "--expect-image-change"), 0)
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertTrue(document["pass"] and document["expect_image_change"])
            self.assertFalse(document["weights"]["pass"])

        # Either build without weight-digests is refused before the rounds.
        for build in ("baseline", "candidate"):
            with self.subTest(build=build), TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                for name in ("baseline", "candidate"):
                    self.fake_checkout(root, name, a)
                (root / build / "build" / weights.WEIGHT_DIGESTS).unlink()
                with (
                    contextlib.redirect_stderr(io.StringIO()) as errors,
                    self.assertRaises(SystemExit),
                ):
                    regression.parse_args(
                        [
                            "--baseline",
                            str(root / "baseline"),
                            "--candidate",
                            str(root / "candidate"),
                            "--model-root",
                            str(root / "model"),
                        ]
                    )
                self.assertIn(
                    f"the {build} build has no {weights.WEIGHT_DIGESTS}",
                    errors.getvalue(),
                )

    def test_every_load_sends_the_models_own_chat_requests(self):
        # A model whose tokenizer and chat template give the built-in tokens,
        # Qwen's, loads without --prompts, as before.
        self.assertEqual(len(chat_prompts.builtin()), len(chat_prompts.TEXTS))
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64)
            self.fake_checkout(root, "candidate", "a" * 64)
            self.assertEqual(self.run_main(root), 0)
            given = [
                json.loads(line)
                for line in (root / "prompts.jsonl").read_text().splitlines()
            ]
            self.assertEqual({prompts for _name, prompts in given}, {None})
            self.assertFalse((root / "release/prompts.txt").exists())
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertEqual(document["prompts"], "built-in")

        # Another family's tokens go to the calibration and every round.
        own = [[1, 2, 3]] + [
            [4 + index, 5] for index in range(len(chat_prompts.TEXTS) - 1)
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64)
            self.fake_checkout(root, "candidate", "a" * 64)
            self.assertEqual(self.run_main(root, prompts=own), 0)
            path = root / "release/prompts.txt"
            given = [
                json.loads(line)
                for line in (root / "prompts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                given,
                [["candidate", str(path)]]
                + [[version, str(path)] for version in regression.ROUNDS],
            )
            self.assertEqual(
                path.read_text().splitlines(),
                [" ".join(map(str, tokens)) for tokens in own],
            )
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertEqual(document["prompts"], str(path))

    def test_both_builds_load_the_compared_images_as_the_options_say(self):
        # The options stream the routed experts and the PLE table in each
        # build's weight-digests, and only where they are given; the result
        # records them and its report names them.
        streamed = ["--expert-memory", "7G", "--ple-memory", "2G"]
        for options, reported in (
            ([], "weight bytes: 1 images PASS"),
            (
                streamed[2:],
                "weight bytes: 1 images (loaded with --ple-memory 2G) PASS",
            ),
            (
                streamed,
                "weight bytes: 1 images "
                "(loaded with --expert-memory 7G --ple-memory 2G) PASS",
            ),
        ):
            with self.subTest(options=options), TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                self.fake_checkout(root, "baseline", "a" * 64)
                self.fake_checkout(root, "candidate", "a" * 64)
                self.assertEqual(self.run_main(root, *options), 0)
                calls = [
                    line.split()
                    for line in (root / "digests.log").read_text().splitlines()
                ]
                self.assertEqual([call[0] for call in calls], ["baseline", "candidate"])
                for call in calls:
                    self.assertEqual(call[2:], options)
                document = json.loads(
                    (root / "release/backend-regression.json").read_text()
                )
                self.assertEqual(document["weight_digest_options"], options)
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    regression.report(document)
                self.assertIn(reported, printed.getvalue().splitlines())

    def test_every_round_runs_the_split_the_candidate_calibrates(self):
        # The baseline would calibrate a split of two shares, 0.26 in one
        # step and 0.44 in two, and report 0.44 from 531 rows; the
        # candidate's start calibrates 10 of 34 units from 832 rows before
        # the rounds, and all four rounds, the baseline's first among them,
        # run that split.
        calibrated = 1 - 24 / 34
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(
                root, "baseline", "a" * 64, 0.26, minimum_rows=531, two_steps=0.44
            )
            self.fake_checkout(root, "candidate", "a" * 64, calibrated)
            self.assertEqual(self.run_main(root), 0)
            shares = [
                json.loads(line)
                for line in (root / "shares.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                shares,
                [["candidate", None, None]]
                + [[version, repr(calibrated), "832"] for version in regression.ROUNDS],
            )
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertEqual(
                (document["ane_ffn_share"], document["ane_ffn_minimum_rows"]),
                (calibrated, 832),
            )
            self.assertEqual(
                [
                    (record["ane_ffn"]["share"], record["ane_ffn"]["minimum_rows"])
                    for record in document["rounds"]
                ],
                [(calibrated, 832)] * 4,
            )
            # The result says the split the calibration chose and each
            # round ran, as their starts logged them.
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                regression.report(document)
            lines = printed.getvalue().splitlines()
            self.assertIn(
                "Neural Engine split calibrated by the candidate: Neural Engine "
                "FFN split at share 0.29 for chunks of 832 rows or more and 0.29 "
                "in two steps from 1024 rows (0.1 s).",
                lines,
            )
            for index, version in enumerate(regression.ROUNDS):
                self.assertIn(
                    f"round {index + 1} {version}: Neural Engine FFN split at the "
                    "given share 0.29 for chunks of 832 rows or more and 0.29 in "
                    "two steps from 1024 rows (0.1 s).",
                    lines,
                )
        args = SimpleNamespace(ane_ffn_share=0.3, ane_ffn_split=None)
        regression.pin_ane_ffn(
            args, {"ane_ffn_share": 0.3, "ane_ffn_minimum_rows": 832}
        )
        with self.assertRaisesRegex(
            regression.RegressionError,
            "at share 0.3 from 512 rows, not the first round's 0.3 from 832",
        ):
            regression.pin_ane_ffn(
                args, {"ane_ffn_share": 0.3, "ane_ffn_minimum_rows": 512}
            )

    def test_rounds_of_different_splits_fail_the_comparison(self):
        rounds = [
            regression.round_record(version, benchmark_document(build=version))
            for version in regression.ROUNDS
        ]
        rounds[0]["ane_ffn"] = {"share": 0.44, "minimum_rows": 531, "log": None}
        summary = regression.summarize(rounds, expect_output_change=False)
        self.assertIn(
            "rounds ran different Neural Engine splits (share, least rows): "
            "[(0.0, 0), (0.44, 531)]",
            summary["failures"],
        )

    def test_a_calibration_without_the_split_pins_no_share(self):
        # After a calibration that ran no split, every round runs as it is,
        # not as --ane-ffn-share 0, which would serve another context.
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64)
            self.fake_checkout(root, "candidate", "a" * 64)
            self.assertEqual(self.run_main(root), 0)
            shares = [
                json.loads(line)
                for line in (root / "shares.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                shares,
                [["candidate", None, None]]
                + [[version, None, None] for version in regression.ROUNDS],
            )

    def test_a_round_that_runs_another_ane_ffn_share_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64, 0.3)
            self.fake_checkout(root, "candidate", "a" * 64, honours_share=False)
            with self.assertRaisesRegex(
                regression.RegressionError,
                r"round-2-candidate-decode-partial-short-transcripts: ran the Neural Engine split "
                r"at share 0.0 from 0 rows, not the first round's 0.3 from 832",
            ):
                self.run_main(root)

    def test_a_given_ane_ffn_share_runs_in_every_round(self):
        # Share 0, the GPU alone, in builds that would calibrate a split.
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64, 1 - 24 / 34)
            self.fake_checkout(root, "candidate", "a" * 64, 0.32)
            self.assertEqual(self.run_main(root, "--ane-ffn-share", "0"), 0)
            shares = [
                json.loads(line)
                for line in (root / "shares.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                shares, [[version, "0.0", None] for version in regression.ROUNDS]
            )
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertEqual(document["ane_ffn_share"], 0.0)

    def test_a_reported_ane_ffn_share_must_be_the_given_one(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64)
            self.fake_checkout(root, "candidate", "a" * 64, 0.32, honours_share=False)
            with self.assertRaisesRegex(
                regression.RegressionError,
                r"round-2-candidate-decode-partial-short-transcripts: ran the Neural Engine split "
                r"at share 0.32, not the given 0.0",
            ):
                self.run_main(root, "--ane-ffn-share", "0")

    def test_a_given_share_runs_at_the_whole_units_it_rounds_to(self):
        # 0.44 of 34 units runs as 15 of them, 0.4411..., in every round.
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64, 0.26, units=34)
            self.fake_checkout(root, "candidate", "a" * 64, 0.29, units=34)
            self.assertEqual(self.run_main(root, "--ane-ffn-share", "0.44"), 0)
            shares = [
                json.loads(line)
                for line in (root / "shares.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                shares, [[version, "0.44", None] for version in regression.ROUNDS]
            )
            document = json.loads(
                (root / "release/backend-regression.json").read_text()
            )
            self.assertEqual(
                (document["ane_ffn_share"], document["ane_ffn_minimum_rows"]),
                (15 / 34, 512),
            )
            self.assertNotIn("ane_ffn_calibration", document)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                regression.report(document)
            self.assertIn(
                "round 1 baseline (given share 0.44): Neural Engine FFN split at "
                "the given share 0.44 for chunks of 512 rows or more and 0.44 in "
                "two steps from 1024 rows (0.1 s).",
                printed.getvalue().splitlines(),
            )
        # A build that runs no split for a given share fails.
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64, units=34)
            self.fake_checkout(root, "candidate", "a" * 64, honours_share=False)
            with self.assertRaisesRegex(
                regression.RegressionError,
                r"round-2-candidate-decode-partial-short-transcripts: ran no Neural "
                r"Engine split, not the share 0.44 it was given",
            ):
                self.run_main(root, "--ane-ffn-share", "0.44")

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
        # A build's plan must not change between its rounds.
        summary = regression.summarize(rounds({2: split}), False)
        self.assertEqual(
            summary["failures"],
            ["the candidate build's memory plan changed between rounds"],
        )
        # main fails on the loss too.
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fake_checkout(root, "baseline", "a" * 64)
            self.fake_checkout(root, "candidate", "a" * 64, context=CONTEXT - 1)
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
