import contextlib
import copy
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dev.benchmarks import http_regression as benchmark
from dev.benchmarks import weights
from dev.tests import smoke_real as smoke


class CharacterTokenizer:
    def encode(self, content, **_kwargs):
        return list(content)

    def decode(self, tokens):
        return "".join(tokens)

    def apply_chat_template(self, messages, **_kwargs):
        return self.encode("<user>" + messages[0]["content"] + "<assistant>")


class HttpRegressionTests(unittest.TestCase):
    def test_any_repository_selects_matching_model_root_and_api_name(self):
        for selected in (
            "mlx-community/Qwen3.8-27B-4bit",
            "mlx-community/Qwen3.6-35B-A3B-4bit",
            "community/custom-splash",
            "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
        ):
            with self.subTest(model=selected):
                arguments = smoke.parse_args(["--model", selected])
                self.assertEqual(arguments.model, selected)
                self.assertEqual(
                    arguments.model_root,
                    smoke.model_artifacts.selection_link(
                        smoke.model_artifacts.MODELS, selected
                    ),
                )
        model = "mlx-community/Qwen3.8-27B-4bit"
        arguments = smoke.parse_args(["--model-root", "custom-root", "--model", model])
        self.assertEqual(str(arguments.model_root), "custom-root")
        self.assertEqual(arguments.model, model)

    def test_real_helpers_require_a_canonical_model(self):
        for parse in (smoke.parse_args, benchmark.parse_args):
            for arguments in ([], ["--model", "qwen3.8-27b"], ["--model", "custom"]):
                with (
                    self.subTest(parse=parse, arguments=arguments),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        parse(arguments)

    def test_benchmark_runs_any_installation_and_holds_its_assembly(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binary = root / "splash"
            binary.touch()
            (root / "splash.metallib").touch()
            models = root / "models"
            # A selection links an assembly that records model.json; a Splash
            # package an earlier release installed records manifest.json.
            package = "incoai/Qwen3.6-35B-A3B-Splash"
            (models / package).mkdir(parents=True)
            (models / package / "manifest.json").write_text("{}")
            upstream = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"
            assembly = models / ".resolved/assembly"
            assembly.mkdir(parents=True)
            (assembly / "model.json").write_text("{}")
            (models / upstream).parent.mkdir(parents=True)
            (models / upstream).symlink_to(assembly, target_is_directory=True)

            def parse(model):
                with mock.patch.object(smoke.model_artifacts, "MODELS", models):
                    return benchmark.parse_args(
                        [
                            "--model",
                            model,
                            "--binary",
                            str(binary),
                            "--baseline-binary",
                            str(binary),
                        ]
                    )

            def hold(arguments):
                with mock.patch.object(smoke.model_artifacts, "MODELS", models):
                    smoke.hold_model_root(arguments)

            # The candidate's weight images are compared by its weight-digests.
            with (
                contextlib.redirect_stderr(io.StringIO()) as error,
                self.assertRaises(SystemExit),
            ):
                parse(upstream)
            self.assertIn("weight-digests", error.getvalue())
            (root / weights.WEIGHT_DIGESTS).parent.mkdir()
            (root / weights.WEIGHT_DIGESTS).touch()
            # Parsing only names the selection link; the servers' run holds it.
            arguments = parse(upstream)
            self.assertEqual(arguments.model_root, models / upstream)
            self.assertFalse(smoke.assembly.is_held(assembly))
            hold(arguments)
            try:
                # Installations collect an unlinked assembly unless it is held.
                self.assertEqual(arguments.model_root, assembly)
                self.assertTrue(smoke.assembly.is_held(assembly))
            finally:
                arguments.held_record.close()
            self.assertFalse(smoke.assembly.is_held(assembly))
            # Neither a package nor a model that is not installed is served.
            for model in (package, "community/not-installed"):
                error = io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                    parse(model)
                self.assertIn("missing installed model", error.getvalue())

    def test_another_checkouts_assembly_is_held_by_its_installation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            # This checkout installed nothing; another one built the assembly.
            checkout = root / "checkout/install/models"
            models = root / "other/install/models"
            assembly = models / ".resolved/assembly"
            assembly.mkdir(parents=True)
            (assembly / "model.json").write_text("{}")
            link = models / "owner/model"
            link.parent.mkdir()
            link.symlink_to(assembly, target_is_directory=True)
            for model_root in (assembly, link):
                with self.subTest(model_root=str(model_root.relative_to(models))):
                    with mock.patch.object(smoke.model_artifacts, "MODELS", checkout):
                        arguments = smoke.parse_args(
                            [
                                "--model-root",
                                str(model_root),
                                "--model",
                                "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
                            ]
                        )
                        smoke.hold_model_root(arguments)
                    try:
                        self.assertEqual(arguments.model_root, assembly)
                        self.assertTrue(smoke.assembly.is_held(assembly))
                    finally:
                        arguments.held_record.close()
            self.assertTrue((models / ".install.lock").exists())
            self.assertFalse(checkout.exists())

    def row(self, version, sample=0, latency=10, round=None):
        return {
            "version": version,
            "round": {"baseline": 0, "candidate": 1}[version]
            if round is None
            else round,
            "sample": sample,
            "context": 2048,
            "scenario": "cold",
            "prompt_sha256": "prompt",
            "response_sha256": "answer",
            "usage": {"completion_tokens": 1},
            "metrics": {"request_latency": {"ttft_ms": latency}},
        }

    def test_exact_prompt_lengths_unique_prefix_and_reproducibility(self):
        tokenizer = CharacterTokenizer()
        prompts = benchmark.prompts(tokenizer, [256, 2048], 2, "nonce")
        self.assertEqual(prompts, benchmark.prompts(tokenizer, [256, 2048], 2, "nonce"))
        for (_sample, context), prompt in prompts.items():
            self.assertEqual(
                len(
                    tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
                ),
                context,
            )
        self.assertNotEqual(prompts[0, 256][:32], prompts[1, 256][:32])

    def abba_rows(self, latencies):
        """Samples 0 and 1 of both versions in rounds baseline, candidate,
        candidate, baseline with these per-round latencies."""
        return [
            self.row(version, sample, latency, round)
            for round, (version, sample, latency) in enumerate(
                zip(benchmark.ROUNDS, (0, 0, 1, 1), latencies)
            )
        ]

    def test_abba_rule_keeps_raw_timing_separate_from_correctness(self):
        for latencies, verdict in (
            ((10, 10.1, 10.1, 10), "pass"),
            ((10, 10.5, 10.5, 10), "fail"),
            ((9.8, 10.5, 10.5, 10.2), "pass"),
            ((9.6, 10, 10, 10.4), "inconclusive"),
        ):
            with self.subTest(latencies=latencies):
                summary = benchmark.summarize(self.abba_rows(latencies))[0]
                self.assertEqual(summary["verdict"], verdict)
                self.assertEqual(summary["pass"], verdict == "pass")
                self.assertEqual(summary["rounds"], list(latencies))
                self.assertEqual(summary["samples_per_round"], [1, 1, 1, 1])
        rows = self.abba_rows((10, 10, 10, 10))
        with self.assertRaisesRegex(ValueError, "no samples"):
            benchmark.summarize(rows[:3] + [{**rows[3], "round": 0}])

    def test_layout_identity_is_compared_within_each_build(self):
        def status(build, layout, kv="int8"):
            return {
                "identity": {
                    "cache": {"build_id": build, "loaded_model_layout_sha256": layout},
                    "kv": {"format": kv},
                }
            }

        baseline = {"version": "baseline", **status("b", "layout-b")}
        # The builds' weights are compared by their bytes instead.
        benchmark.check_identity(status("c", "layout-c"), "candidate", [baseline])
        for current, version in (
            (status("b", "layout-c"), "baseline"),
            (status("b2", "layout-b"), "baseline"),
            (status("c", "layout-c", "bf16"), "candidate"),
        ):
            with (
                self.subTest(current=current, version=version),
                self.assertRaises(smoke.SmokeFailure),
            ):
                benchmark.check_identity(current, version, [baseline])

    def test_missing_duplicate_or_changed_transcript_fails(self):
        baseline, candidate = self.row("baseline"), self.row("candidate")
        for rows in ([baseline], [baseline, baseline, candidate]):
            with self.assertRaises(ValueError):
                benchmark.summarize(rows)
        for field in ("response_sha256", "prompt_sha256"):
            changed = copy.deepcopy(candidate)
            changed[field] = "different"
            with self.assertRaisesRegex(ValueError, "transcript differs"):
                benchmark.summarize([baseline, changed])

    def burst_rows(self, failures, ttft, decode):
        """One burst per round in ABBA order: each version's replay-point
        publication failures, and per round the next turn's time to first
        token (no next turn when ttft is None) and the burst's decode time
        per token."""
        return [
            {
                "version": version,
                "round": round,
                "sample": round // 2,
                "context": 16384,
                "scenario": "burst",
                "replay_state_publication_failures": failures[version],
                "decode_ms_per_token": decode[round],
                "maximum_context_tokens": 100000,
                "follow_ups": []
                if ttft is None
                else [
                    {"ttft_ms": ttft[round], "matched_tokens": 16352, "resumed": True}
                ],
            }
            for round, version in enumerate(benchmark.ROUNDS)
        ]

    def test_a_burst_keeps_the_candidate_only_by_every_rule(self):
        lost = {"baseline": 2, "candidate": 0}
        sooner = (100, 80, 81, 102)
        steady = (10, 10.1, 10.1, 10)
        kept = benchmark.summarize_bursts(self.burst_rows(lost, sooner, steady))[0]
        self.assertTrue(kept["keep"])
        self.assertEqual(kept["baseline"]["failures_per_burst"], 2)
        self.assertEqual(kept["candidate"]["resumed_follow_ups"], 2)
        for name, failures, ttft, decode in (
            ("nothing lost", {"baseline": 0, "candidate": 0}, sooner, steady),
            ("still lost", {"baseline": 2, "candidate": 1}, sooner, steady),
            ("not 15% sooner", lost, (100, 90, 90, 100), steady),
            ("overlapping rounds", lost, (100, 60, 85, 80), steady),
            ("decode slower", lost, sooner, (10, 10.5, 10.5, 10)),
            ("decode inconclusive", lost, sooner, (10, 9, 10, 10)),
            ("no follow-ups", lost, None, steady),
        ):
            with self.subTest(name):
                summary = benchmark.summarize_bursts(
                    self.burst_rows(failures, ttft, decode)
                )[0]
                self.assertFalse(summary["keep"])

    def test_a_follow_up_needs_a_burst_of_two(self):
        model = "mlx-community/Qwen3.8-27B-4bit"
        for extra in (["--follow-up"], ["--burst", "1", "--follow-up"]):
            with (
                self.subTest(extra=extra),
                contextlib.redirect_stderr(io.StringIO()) as error,
                self.assertRaises(SystemExit),
            ):
                benchmark.parse_args(
                    ["--model", model, "--baseline-binary", "baseline", *extra]
                )
            self.assertIn("--burst needs two or more", error.getvalue())

    def test_decode_uses_native_decode_cycle_per_token(self):
        rows = self.abba_rows((10, 10, 10, 10))
        for row in rows:
            row["scenario"] = "decode"
            row["native_delta"] = {
                "decode_wall_ms": 80,
                "decode_cycle_ms": 100,
                "decode_output_tokens": 50,
            }
        summary = benchmark.summarize(rows)[0]
        self.assertEqual(summary["metric"], "decode_cycle_ms_per_token")
        self.assertEqual(summary["baseline_median"], 2)


if __name__ == "__main__":
    unittest.main()
