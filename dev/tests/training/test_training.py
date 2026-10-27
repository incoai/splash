import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from dev.tests.training.fixtures import cached_features, tiny_components, tiny_config
from draft_training.config import Recipe, TrainingError, read_json

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional training dependencies are not installed")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(1)
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model, self.embedding, self.head = tiny_components()
        self.training, self.validation = self.root / "train", self.root / "validation"
        cached_features(self.training)
        cached_features(self.validation)
        self.recipe = replace(
            Recipe(),
            max_steps=8,
            max_new_tokens=8,
            sequence_length=64,
            gradient_accumulation=3,
            checkpoint_interval=1,
            validation_interval=2,
            max_validation_blocks=2,
            trainable="all",
            learning_rate=0.001,
        )

    def run_train(self, model, directory, **kwargs):
        from draft_training.trainer import train

        directory.mkdir(exist_ok=True)
        return train(
            model,
            self.embedding,
            self.head,
            self.training,
            self.validation,
            directory,
            self.recipe,
            "identity",
            {
                "teacher_identity": {"teacher": "sha"},
                "initial_draft_identity": {},
                "training_split": {"identities": ["train"]},
                "validation_split": {"identities": ["validation"]},
            },
            "cpu",
            family=False,
            **kwargs,
        )

    def test_resume_matches_uninterrupted_training_across_dataset_wrap(self):
        from safetensors.torch import load_file

        uninterrupted = self.root / "full"
        resumed = self.root / "resumed"
        self.run_train(self.model, uninterrupted)
        restored, _, _ = tiny_components()
        self.run_train(restored, resumed, stop_after=6)
        # Six steps consume 18 blocks, crossing the fixture's first epoch.
        # Recover a committed checkpoint without its pointer and discard an
        # uncheckpointed/partial log tail left by interruption.
        (resumed / "checkpoints/latest.json").unlink()
        with (resumed / "metrics.jsonl").open("a") as stream:
            stream.write('{"step": 999}\n{"step":')
        after_restart, _, _ = tiny_components()
        self.run_train(after_restart, resumed, resume=True)
        for filename in (
            "checkpoints/step-00000008/model.safetensors",
            "export/model.safetensors",
        ):
            a, b = (
                load_file(str(uninterrupted / filename)),
                load_file(str(resumed / filename)),
            )
            for name in a:
                self.torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        self.assertEqual(
            read_json(uninterrupted / "checkpoints/step-00000008/progress.json"),
            read_json(resumed / "checkpoints/step-00000008/progress.json"),
        )

    def test_export_roundtrip_and_finite_guard(self):
        from draft_training.config import checkpoint_identity, validate_draft
        from draft_training.export import export_model
        from draft_training.model import load_draft

        destination = self.root / "export"
        export_model(
            self.model,
            destination,
            {"source": "fixture"},
            family=False,
            shard_bytes=1024,
        )
        self.assertEqual(validate_draft(destination, family=False), tiny_config())
        restored = load_draft(destination, tiny_config())
        for name, value in self.model.state_dict().items():
            self.torch.testing.assert_close(
                value, restored.state_dict()[name], rtol=0, atol=0
            )
        self.assertTrue(checkpoint_identity(destination))
        with self.assertRaisesRegex(TrainingError, "already exists"):
            export_model(self.model, destination, {}, family=False)
        with self.torch.no_grad():
            self.model.fc.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(TrainingError, "nonfinite"):
            export_model(self.model, self.root / "bad-export", {}, family=False)
        self.assertFalse((self.root / "bad-export").exists())

    def test_selection_rejects_test_overlap_and_modified_export(self):
        from draft_training.config import checkpoint_identity, write_json
        from draft_training.evaluate import validate_selection
        from draft_training.export import export_model

        export = self.root / "export"
        export_model(self.model, export, {}, family=False)
        path = self.root / "selection.json"
        selection = {
            "export_identity": checkpoint_identity(export),
            "teacher_identity": {"a": "b"},
            "recipe": self.recipe.to_dict(),
            "training_split": {"identities": ["train"]},
            "validation_split": {"identities": ["validation"]},
        }
        write_json(path, selection)
        validate_selection(
            path, export, {"identities": ["test"]}, {"a": "b"}, self.recipe
        )
        with self.assertRaisesRegex(TrainingError, "overlap"):
            validate_selection(
                path, export, {"identities": ["validation"]}, {"a": "b"}, self.recipe
            )
        write_json(export / "config.json", {"changed": True})
        with self.assertRaisesRegex(TrainingError, "differs"):
            validate_selection(
                path, export, {"identities": ["test"]}, {"a": "b"}, self.recipe
            )

    def test_feature_shard_integrity_is_enforced(self):
        from draft_training.capture import load_sample

        manifest = read_json(self.training / "manifest.json")
        with (self.training / "000000.safetensors").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(TrainingError, "digest"):
            load_sample(self.training, manifest["shards"][0])

    def test_training_improves_diagnostic_unary_loss_with_frozen_head(self):
        from draft_training.capture import load_sample
        from draft_training.evaluate import evaluate_cached

        # A small, repeated supervised block proves real optimization, rather
        # than testing that parameters merely changed due to weight decay.
        from draft_training.losses import block_loss
        from draft_training.trainer import configure_trainable

        configure_trainable(self.model, "all")
        optimizer = self.torch.optim.AdamW(
            self.model.parameters(), lr=0.01, weight_decay=0
        )
        sample = load_sample(
            self.training, read_json(self.training / "manifest.json")["shards"][0]
        )
        before = self.head.weight.clone()
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            with self.torch.autocast("cpu", dtype=self.torch.bfloat16):
                hidden = self.model.block_hidden(
                    sample["tokens"], sample["features"], 5, self.embedding
                )
                loss, _ = block_loss(
                    self.model, hidden, sample["tokens"], 5, self.head, 0
                )
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0] * 0.7)
        self.torch.testing.assert_close(before, self.head.weight, rtol=0, atol=0)
        self.assertIsNone(self.head.weight.grad)
        self.assertTrue(
            evaluate_cached(
                self.model,
                self.embedding,
                self.head,
                self.validation,
                self.recipe,
                "cpu",
            )["blocks"]
        )

    def test_resume_from_completed_capture_without_optimizer_checkpoint(self):
        output = self.root / "capture-only"
        output.mkdir()
        result = self.run_train(self.model, output, resume=True)
        self.assertEqual(result["step"], self.recipe.max_steps)
        self.assertTrue((output / "selection.json").is_file())

    def test_resume_finishes_selection_after_export_was_published(self):
        from unittest import mock

        from draft_training.config import checkpoint_identity, write_json

        output = self.root / "interrupted-export"

        def interrupt_selection(path, value):
            if Path(path).name == "selection.json":
                raise KeyboardInterrupt
            write_json(path, value)

        with mock.patch(
            "draft_training.trainer.write_json", side_effect=interrupt_selection
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.run_train(self.model, output)
        self.assertTrue((output / "export").is_dir())
        self.assertFalse((output / "selection.json").exists())
        published = checkpoint_identity(output / "export")
        restored, _, _ = tiny_components()
        with mock.patch(
            "torch.optim.AdamW.step", side_effect=AssertionError("training repeated")
        ):
            self.run_train(restored, output, resume=True)
        self.assertEqual(
            read_json(output / "selection.json")["export_identity"], published
        )
        self.assertEqual(checkpoint_identity(output / "export"), published)
        with mock.patch(
            "draft_training.trainer.write_json",
            side_effect=AssertionError("completed selection overwritten"),
        ):
            self.run_train(restored, output, resume=True)

    def test_resume_rejects_changed_published_export(self):
        from safetensors.torch import load_file, save_file

        output = self.root / "changed-export"
        self.run_train(self.model, output)
        (output / "selection.json").unlink()
        weights = output / "export/model.safetensors"
        state = load_file(str(weights))
        state["fc.weight"][0, 0] += 1
        save_file(state, str(weights))
        restored, _, _ = tiny_components()
        with self.assertRaisesRegex(TrainingError, "export differs"):
            self.run_train(restored, output, resume=True)
        self.assertFalse((output / "selection.json").exists())

    def test_validation_scores_exported_bf16_weights_and_restores_fp32_masters(self):
        from draft_training.evaluate import evaluate_cached
        from draft_training.export import export_model
        from draft_training.model import load_draft
        from draft_training.trainer import configure_trainable

        configure_trainable(self.model, "all")
        with self.torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(0.0013)
        masters = {name: p.clone() for name, p in self.model.named_parameters()}
        export = self.root / "bf16-validation"
        export_model(self.model, export, {}, family=False)
        restored = load_draft(export, tiny_config())
        expected = evaluate_cached(
            restored, self.embedding, self.head, self.validation, self.recipe, "cpu"
        )
        actual = evaluate_cached(
            self.model, self.embedding, self.head, self.validation, self.recipe, "cpu"
        )
        self.assertEqual(actual, expected)
        for name, parameter in self.model.named_parameters():
            self.assertEqual(parameter.dtype, self.torch.float32)
            self.torch.testing.assert_close(parameter, masters[name], rtol=0, atol=0)

        # A failed validation must also leave the optimizer's master values intact.
        with self.assertRaises(OSError):
            evaluate_cached(
                self.model,
                self.embedding,
                self.head,
                self.root / "missing",
                self.recipe,
                "cpu",
            )
        for name, parameter in self.model.named_parameters():
            self.torch.testing.assert_close(parameter, masters[name], rtol=0, atol=0)

    def test_evaluation_budget_is_spread_across_continuations(self):
        import shutil

        from draft_training.config import file_digest, write_json
        from draft_training.evaluate import evaluate_cached

        manifest = read_json(self.validation / "manifest.json")
        for i in (1, 2):
            name = f"{i:06d}.safetensors"
            shutil.copyfile(
                self.validation / "000000.safetensors", self.validation / name
            )
            manifest["shards"].append(
                {
                    "file": name,
                    "sha256": file_digest(self.validation / name),
                    "tokens": 24,
                }
            )
        write_json(self.validation / "manifest.json", manifest)
        recipe = replace(self.recipe, max_validation_blocks=6)
        metrics = evaluate_cached(
            self.model, self.embedding, self.head, self.validation, recipe, "cpu"
        )
        self.assertEqual(metrics["blocks"], 6)
        self.assertEqual(metrics["continuations_evaluated"], 3)

    def validation_continuations(self, assistant_lengths):
        from safetensors.torch import load_file, save_file

        from draft_training.config import file_digest, write_json

        sample = load_file(str(self.validation / "000000.safetensors"))
        manifest = read_json(self.validation / "manifest.json")
        manifest["shards"] = []
        for i, length in enumerate(assistant_lengths):
            sample["assistant_mask"] = self.torch.arange(24) >= 24 - length
            path = self.validation / f"{i:06d}.safetensors"
            save_file(sample, str(path))
            manifest["shards"].append(
                {"file": path.name, "sha256": file_digest(path), "tokens": 24}
            )
        write_json(self.validation / "manifest.json", manifest)

    def test_evaluation_skips_short_continuations_before_spending_budget(self):
        from draft_training.evaluate import evaluate_cached

        # With seed 42, the old one-shard sample chose only the short answer.
        self.validation_continuations([1, 12])
        recipe = replace(self.recipe, max_validation_blocks=1)
        first = evaluate_cached(
            self.model, self.embedding, self.head, self.validation, recipe, "cpu"
        )
        self.assertEqual(first["blocks"], 1)
        self.assertEqual(first["continuations_evaluated"], 1)
        self.assertEqual(first["total_continuations"], 2)
        self.assertEqual(
            first,
            evaluate_cached(
                self.model, self.embedding, self.head, self.validation, recipe, "cpu"
            ),
        )

    def test_evaluation_redistributes_unused_continuation_quota(self):
        from draft_training.evaluate import evaluate_cached

        # One answer supplies one block; the other supplies six. Both orders
        # must fill the six-block budget without duplicating any anchors.
        for lengths in ([7, 12], [12, 7]):
            with self.subTest(lengths=lengths):
                self.validation_continuations(lengths)
                recipe = replace(self.recipe, max_validation_blocks=6)
                metrics = evaluate_cached(
                    self.model,
                    self.embedding,
                    self.head,
                    self.validation,
                    recipe,
                    "cpu",
                )
                self.assertEqual(metrics["blocks"], 6)
                self.assertEqual(metrics["continuations_evaluated"], 2)
