import contextlib
import importlib.util
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from dev.tests.training.fixtures import tiny_components, tiny_target, write_data
from draft_training.config import Recipe, TrainingError, read_json

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional training dependencies are not installed")
class CaptureTests(unittest.TestCase):
    def setUp(self):
        import torch

        from draft_training.capture import TargetAdapter

        torch.set_num_threads(1)
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "target"
        self.reference, self.tokenizer = tiny_target(self.target)
        self.adapter = TargetAdapter.load(self.target, [0, 2], "cpu")
        self.recipe = replace(
            Recipe(),
            sequence_length=64,
            max_new_tokens=12,
            max_steps=2,
            gradient_accumulation=1,
            validation_interval=1,
            max_validation_blocks=2,
        )

    def test_actual_target_adapter_matches_decoder_outputs_without_embedding_offset_error(
        self,
    ):
        tokens = self.torch.arange(16) % 28
        features = self.adapter.features(tokens)
        with self.torch.no_grad():
            output = self.adapter.decoder(
                input_ids=tokens[None],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        expected = self.torch.cat(
            [output.hidden_states[1], output.hidden_states[3]], dim=-1
        )[0]
        self.torch.testing.assert_close(features, expected, rtol=0, atol=0)
        self.assertFalse(features.requires_grad)
        self.assertFalse(any(p.requires_grad for p in self.adapter.model.parameters()))

    def test_generate_capture_cache_reuse_and_provenance_rejection(self):
        from draft_training.capture import capture_dataset, load_sample

        data = self.root / "data.jsonl"
        write_data(data, "w5 w6", "original assistant response")
        output = self.root / "features"
        with contextlib.redirect_stdout(io.StringIO()):
            first = capture_dataset(
                self.adapter,
                data,
                output,
                self.recipe,
                {"teacher": "one"},
                "target-generated",
            )
            again = capture_dataset(
                self.adapter,
                data,
                output,
                self.recipe,
                {"teacher": "one"},
                "target-generated",
            )
        self.assertEqual(first, again)
        sample = load_sample(output, first["shards"][0])
        self.assertEqual(sample["features"].shape, (len(sample["tokens"]), 32))
        self.assertEqual(int(sample["assistant_mask"].sum()), 12)
        with self.assertRaisesRegex(TrainingError, "provenance"):
            capture_dataset(
                self.adapter,
                data,
                output,
                self.recipe,
                {"teacher": "two"},
                "target-generated",
            )

    def test_supplied_answer_uses_actual_length_instead_of_generation_budget(self):
        from unittest import mock

        from draft_training.data import anchors

        messages = [{"role": "user", "content": " ".join(["w5"] * 54)}]
        with mock.patch.object(self.adapter.model, "generate") as generate:
            sample = self.adapter.make_sample(
                messages, " ".join(["w6"] * 8), self.recipe
            )
            self.assertEqual(len(sample["tokens"]), self.recipe.sequence_length)
            self.assertTrue(list(anchors(sample, 8)))
            with self.assertRaisesRegex(TrainingError, "supplied answer exceeds"):
                self.adapter.make_sample(messages, " ".join(["w6"] * 9), self.recipe)
            with self.assertRaisesRegex(TrainingError, "prompt plus max_new_tokens"):
                self.adapter.make_sample(messages, None, self.recipe)
            generate.assert_not_called()

    def test_thinking_settings_apply_to_prompt_and_supplied_answer(self):
        from unittest import mock

        for enabled in (False, True):
            with self.subTest(enable_thinking=enabled):
                recipe = replace(
                    self.recipe, enable_thinking=enabled, reasoning_effort="low"
                )
                with mock.patch.object(
                    self.adapter.tokenizer,
                    "apply_chat_template",
                    wraps=self.adapter.tokenizer.apply_chat_template,
                ) as apply_template:
                    self.adapter.make_sample(
                        [{"role": "user", "content": "w5 w6"}],
                        " ".join(["w7"] * 8),
                        recipe,
                    )
                self.assertEqual(apply_template.call_count, 2)
                for call in apply_template.call_args_list:
                    self.assertIs(call.kwargs["enable_thinking"], enabled)
                    self.assertEqual(call.kwargs["reasoning_effort"], "low")

    def test_family_template_thinking_modes_preserve_supplied_answer_prefix(self):
        template = (
            Path(__file__).resolve().parents[1] / "fixtures/chat_templates/qwen38.jinja"
        ).read_text()
        messages = [{"role": "user", "content": "w5 w6"}]
        for enabled in (False, True):
            for effort in ("low", "medium", "xhigh"):
                with self.subTest(enable_thinking=enabled, reasoning_effort=effort):
                    recipe = replace(
                        self.recipe, enable_thinking=enabled, reasoning_effort=effort
                    )
                    kwargs = {
                        "chat_template": template,
                        **recipe.chat_template_kwargs(),
                    }
                    prefix = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **kwargs,
                    )
                    complete = self.tokenizer.apply_chat_template(
                        [*messages, {"role": "assistant", "content": "w7"}],
                        tokenize=False,
                        add_generation_prompt=False,
                        **kwargs,
                    )
                    self.assertTrue(complete.startswith(prefix))
                    self.assertTrue(
                        prefix.endswith("<think>\n" if enabled else "</think>\n\n")
                    )
                    if enabled and effort != "medium":
                        self.assertIn(f"Reasoning effort is set to {effort}.", prefix)
                    else:
                        self.assertNotIn("Reasoning effort is set to", prefix)

    def test_capture_rejects_changed_thinking_settings_before_reuse(self):
        from unittest import mock

        from draft_training.capture import capture_dataset

        data, output = self.root / "data.jsonl", self.root / "features"
        write_data(data, "w5 w6")
        manifest = capture_dataset(
            self.adapter, data, output, self.recipe, {}, "target-generated"
        )
        self.assertEqual(
            manifest["binding"]["chat_template"], self.recipe.chat_template_kwargs()
        )
        for changes in ({"enable_thinking": False}, {"reasoning_effort": "low"}):
            with self.subTest(changes=changes):
                with mock.patch.object(self.adapter, "make_sample") as make_sample:
                    with self.assertRaisesRegex(TrainingError, "provenance"):
                        capture_dataset(
                            self.adapter,
                            data,
                            output,
                            replace(self.recipe, **changes),
                            {},
                            "target-generated",
                        )
                    make_sample.assert_not_called()

    def test_capture_train_export_and_final_test_replay(self):
        from draft_training.capture import capture_dataset
        from draft_training.config import checkpoint_identity, write_json
        from draft_training.data import disjoint, split_manifest
        from draft_training.evaluate import (
            evaluate_cached,
            validate_selection,
            write_test_report,
        )
        from draft_training.export import export_model
        from draft_training.trainer import train

        model, _, _ = tiny_components()
        files = [
            self.root / f"{name}.jsonl" for name in ("train", "validation", "test")
        ]
        manifests, feature_dirs = [], []
        for i, file in enumerate(files):
            write_data(file, f"w{i + 5} w{i + 10}")
            manifests.append(split_manifest(file))
            destination = self.root / f"features-{i}"
            feature_dirs.append(destination)
            capture_dataset(
                self.adapter,
                file,
                destination,
                self.recipe,
                {"teacher": "fixture"},
                "target-generated",
            )
        for a, b in ((0, 1), (0, 2), (1, 2)):
            disjoint(manifests[a], manifests[b])
        original = self.root / "original-draft"
        export_model(model, original, {}, family=False)
        provenance = {
            "teacher_identity": {"teacher": "fixture"},
            "initial_draft_identity": checkpoint_identity(original),
            "training_split": manifests[0],
            "validation_split": manifests[1],
            "supervision": "target-generated",
        }
        output = self.root / "run"
        output.mkdir()
        train(
            model,
            self.adapter.embedding,
            self.adapter.head,
            feature_dirs[0],
            feature_dirs[1],
            output,
            self.recipe,
            "identity",
            provenance,
            "cpu",
            family=False,
        )
        selection = validate_selection(
            output / "selection.json",
            output / "export",
            manifests[2],
            {"teacher": "fixture"},
            self.recipe,
        )
        result = evaluate_cached(
            model,
            self.adapter.embedding,
            self.adapter.head,
            feature_dirs[2],
            self.recipe,
            "cpu",
        )
        write_test_report(output / "test-report.json", selection, manifests[2], result)
        self.assertTrue((output / "export/model.safetensors").exists())
        self.assertEqual(read_json(output / "test-report.json")["metrics"]["blocks"], 2)
        with self.assertRaisesRegex(TrainingError, "already exists"):
            write_test_report(
                output / "test-report.json", selection, manifests[2], result
            )
        # Persist all three split fingerprints in the generated example report.
        write_json(output / "test-split.json", manifests[2])

    def test_cli_orchestration_uses_real_target_capture_and_training(self):
        import argparse
        from unittest import mock

        from draft_training.__main__ import run_evaluation, run_training
        from draft_training.config import validate_draft
        from draft_training.data import split_manifest
        from draft_training.export import export_model

        self.recipe = replace(
            self.recipe, enable_thinking=False, reasoning_effort="low"
        )
        initial, _, _ = tiny_components()
        draft = self.root / "draft"
        export_model(initial, draft, {}, family=False)
        training, validation = self.root / "train.jsonl", self.root / "validation.jsonl"
        write_data(training, "w5 w6")
        write_data(validation, "w7 w8")
        output = self.root / "cli-run"
        args = argparse.Namespace(
            target=self.target,
            init_draft=draft,
            data=training,
            validation_data=validation,
            output=output,
            command="train",
            device="cpu",
            supervision="target-generated",
            preflight=False,
            dry_run=False,
            resume=False,
        )
        # Only the production family-size gate is replaced. The real local
        # loader, tokenizer, capture, release, baseline, trainer and export run.
        inspected = (
            initial.raw_config,
            self.recipe,
            split_manifest(training),
            split_manifest(validation),
        )
        with mock.patch(
            "draft_training.__main__.inspect_training", return_value=inspected
        ):
            run_training(args, family=False)
        selection = read_json(output / "selection.json")
        self.assertEqual(selection["native_validation"], "not run")
        self.assertTrue((output / "baseline-validation.json").exists())
        self.assertTrue(selection["implementation"])
        self.assertEqual(selection["recipe"], self.recipe.to_dict())
        for split in ("train", "validation"):
            self.assertEqual(
                read_json(output / "features" / split / "manifest.json")["binding"][
                    "chat_template"
                ],
                self.recipe.chat_template_kwargs(),
            )
        args.resume = True
        for changes in ({"enable_thinking": True}, {"reasoning_effort": "medium"}):
            with self.subTest(resume_changes=changes):
                with (
                    mock.patch(
                        "draft_training.__main__.inspect_training",
                        return_value=(
                            inspected[0],
                            replace(self.recipe, **changes),
                            *inspected[2:],
                        ),
                    ),
                    mock.patch("draft_training.capture.TargetAdapter.load") as load,
                ):
                    with self.assertRaisesRegex(TrainingError, "resume configuration"):
                        run_training(args, family=False)
                    load.assert_not_called()
        test_data = self.root / "test.jsonl"
        write_data(test_data, "w9 w10")
        evaluation = argparse.Namespace(
            target=self.target,
            draft=output / "export",
            baseline_draft=draft,
            test_data=test_data,
            selection_manifest=output / "selection.json",
            output=output / "final-test",
            device="cpu",
        )
        with (
            mock.patch("draft_training.__main__.validate_target"),
            mock.patch(
                "draft_training.__main__.validate_draft",
                side_effect=lambda directory: validate_draft(directory, family=False),
            ),
        ):
            metrics = run_evaluation(evaluation)
        self.assertEqual(set(metrics), {"baseline", "trained"})
        report = read_json(evaluation.output / "test-report.json")
        self.assertEqual(report["metrics"], metrics)
        self.assertEqual(report["test_split"], split_manifest(test_data))
        self.assertEqual(
            read_json(evaluation.output / "features/manifest.json")["binding"][
                "chat_template"
            ],
            self.recipe.chat_template_kwargs(),
        )

    def test_changed_execution_context_is_rejected_before_test_data_is_opened(self):
        import argparse
        from unittest import mock

        from draft_training.__main__ import (
            execution_context,
            implementation_identity,
            run_evaluation,
        )
        from draft_training.config import checkpoint_identity, write_json
        from draft_training.export import export_model

        model, _, _ = tiny_components()
        draft = self.root / "draft"
        export_model(model, draft, {}, family=False)
        selection = {
            "export_identity": checkpoint_identity(draft),
            "teacher_identity": checkpoint_identity(self.target),
            "initial_draft_identity": checkpoint_identity(draft),
            "recipe": self.recipe.to_dict(),
            "implementation": implementation_identity(),
            "execution": {
                **execution_context(self.torch, "cpu"),
                "device": "different",
            },
            "versions": {},
        }
        path = self.root / "selection.json"
        write_json(path, selection)
        args = argparse.Namespace(
            target=self.target,
            draft=draft,
            baseline_draft=draft,
            selection_manifest=path,
            test_data=self.root / "unopened-test.jsonl",
            output=self.root / "test-output",
            device="cpu",
        )
        with (
            mock.patch("draft_training.__main__.validate_target"),
            mock.patch(
                "draft_training.__main__.validate_draft", return_value=model.raw_config
            ),
        ):
            with self.assertRaisesRegex(TrainingError, "device context"):
                run_evaluation(args)
        selection["execution"] = execution_context(self.torch, "cpu")
        selection["implementation"]["evaluate.py"] = "changed"
        write_json(path, selection)
        with (
            mock.patch("draft_training.__main__.validate_target"),
            mock.patch(
                "draft_training.__main__.validate_draft", return_value=model.raw_config
            ),
        ):
            with self.assertRaisesRegex(TrainingError, "implementation"):
                run_evaluation(args)
        self.assertFalse(args.output.exists())

    def test_resume_rejects_changed_identity_before_loading_teacher(self):
        from unittest import mock

        from draft_training.__main__ import parse_args, run_training
        from draft_training.config import write_json
        from draft_training.data import split_manifest
        from draft_training.export import export_model

        model, _, _ = tiny_components()
        draft = self.root / "draft"
        export_model(model, draft, {}, family=False)
        training, validation = self.root / "train.jsonl", self.root / "validation.jsonl"
        write_data(training, "w5 w6")
        write_data(validation, "w7 w8")
        output = self.root / "run"
        output.mkdir()
        write_json(output / "run.json", {"run_identity": "different"})
        args = parse_args(
            [
                "train",
                "--target",
                str(self.target),
                "--init-draft",
                str(draft),
                "--data",
                str(training),
                "--validation-data",
                str(validation),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--resume",
            ]
        )
        with (
            mock.patch(
                "draft_training.__main__.inspect_training",
                return_value=(
                    model.raw_config,
                    self.recipe,
                    split_manifest(training),
                    split_manifest(validation),
                ),
            ),
            mock.patch("draft_training.capture.TargetAdapter.load") as load,
        ):
            with self.assertRaisesRegex(TrainingError, "resume configuration"):
                run_training(args, family=False)
            load.assert_not_called()
        self.assertEqual(read_json(output / "run.json"), {"run_identity": "different"})
