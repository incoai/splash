"""Apple GPU smoke tests with local tiny checkpoints and no model downloads."""

import importlib.util
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from dev.tests.training.fixtures import (
    cached_features,
    tiny_components,
    tiny_config,
    tiny_target,
    write_data,
)
from draft_training.__main__ import parse_args
from draft_training.config import Recipe, TrainingError, read_json

HAS_TORCH = importlib.util.find_spec("torch") is not None


def has_mps():
    if not HAS_TORCH:
        return False
    import torch

    return torch.backends.mps.is_available()


HAS_MPS = has_mps()


class MPSAvailabilityTests(unittest.TestCase):
    def test_commands_accept_explicit_mps_device(self):
        for command in ("train", "capture"):
            with self.subTest(command=command):
                args = parse_args(
                    [command, "--device", "mps"]
                    + [
                        value
                        for name in (
                            "target",
                            "init-draft",
                            "data",
                            "validation-data",
                            "output",
                        )
                        for value in ("--" + name, name)
                    ]
                )
                self.assertEqual(args.device, "mps")
        args = parse_args(
            ["evaluate", "--device", "mps"]
            + [
                value
                for name in (
                    "target",
                    "draft",
                    "baseline-draft",
                    "test-data",
                    "selection-manifest",
                    "output",
                )
                for value in ("--" + name, name)
            ]
        )
        self.assertEqual(args.device, "mps")

    @unittest.skipUnless(HAS_TORCH, "optional training dependencies are not installed")
    def test_unavailable_mps_is_rejected_without_cpu_fallback(self):
        from draft_training.__main__ import require_torch

        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(TrainingError, "MPS is unavailable"):
                require_torch("mps")

    @unittest.skipUnless(
        os.environ.get("DFLASH_REQUIRE_MPS") == "1",
        "set DFLASH_REQUIRE_MPS=1 to require actual Apple GPU execution",
    )
    def test_requested_mps_backend_is_available(self):
        from draft_training.__main__ import require_torch

        self.assertTrue(require_torch("mps").backends.mps.is_available())


@unittest.skipUnless(HAS_MPS, "Apple GPU MPS and optional dependencies are required")
class MPSTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(1)
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model, embedding, head = tiny_components()
        self.model, self.embedding, self.head = (
            value.to("mps") for value in (model, embedding, head)
        )
        self.training, self.validation = self.root / "train", self.root / "validation"
        cached_features(self.training)
        cached_features(self.validation)
        self.recipe = replace(
            Recipe(),
            max_steps=3,
            sequence_length=64,
            max_new_tokens=12,
            gradient_accumulation=2,
            checkpoint_interval=1,
            validation_interval=1,
            max_validation_blocks=2,
            learning_rate=0.01,
        )

    def test_selector_training_preserves_frozen_backbone_and_target_tables(self):
        from draft_training.trainer import train

        frozen = {
            name: p.detach().cpu().clone()
            for name, p in self.model.named_parameters()
            if not name.startswith("candidate_selector.")
        }
        embedding, head = (
            self.embedding.weight.cpu().clone(),
            self.head.weight.cpu().clone(),
        )
        output = self.root / "run"
        output.mkdir()
        progress = train(
            self.model,
            self.embedding,
            self.head,
            self.training,
            self.validation,
            output,
            self.recipe,
            "mps-fixture",
            {},
            "mps",
            family=False,
        )
        self.assertGreater(progress["updates"], 0)
        self.assertTrue((output / "selection.json").is_file())
        for name, p in self.model.named_parameters():
            if name in frozen:
                self.torch.testing.assert_close(p.cpu(), frozen[name], rtol=0, atol=0)
                self.assertIsNone(p.grad)
        self.torch.testing.assert_close(
            self.embedding.weight.cpu(), embedding, rtol=0, atol=0
        )
        self.torch.testing.assert_close(self.head.weight.cpu(), head, rtol=0, atol=0)
        self.assertIsNone(self.head.weight.grad)

    def test_checkpoint_restores_cpu_and_mps_rng_and_rejects_backend_change(self):
        from draft_training.export import restore_checkpoint, save_checkpoint
        from draft_training.trainer import configure_trainable

        configure_trainable(self.model, "selector")
        optimizer = self.torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=0.001
        )
        checkpoint = save_checkpoint(
            self.model,
            optimizer,
            self.root / "checkpoints",
            {"step": 1, "run_identity": "mps-fixture"},
        )
        expected_cpu = self.torch.rand(16)
        expected_mps = self.torch.rand(16, device="mps").cpu()
        self.torch.manual_seed(999)
        restore_checkpoint(self.model, optimizer, checkpoint, "mps-fixture")
        self.torch.testing.assert_close(
            self.torch.rand(16), expected_cpu, rtol=0, atol=0
        )
        self.torch.testing.assert_close(
            self.torch.rand(16, device="mps").cpu(), expected_mps, rtol=0, atol=0
        )
        cpu_model, _, _ = tiny_components()
        configure_trainable(cpu_model, "selector")
        cpu_optimizer = self.torch.optim.AdamW(
            [p for p in cpu_model.parameters() if p.requires_grad]
        )
        with self.assertRaisesRegex(TrainingError, "original MPS device backend"):
            restore_checkpoint(cpu_model, cpu_optimizer, checkpoint, "mps-fixture")
        state = self.torch.load(
            checkpoint / "state.pt", weights_only=True, map_location="cpu"
        )
        del state["mps_rng"]
        self.torch.save(state, checkpoint / "state.pt")
        with self.assertRaisesRegex(TrainingError, "MPS RNG state"):
            restore_checkpoint(self.model, optimizer, checkpoint, "mps-fixture")

    def test_validation_matches_bf16_export_and_preserves_fp32_masters(self):
        from safetensors.torch import load_file

        from draft_training.evaluate import evaluate_cached
        from draft_training.export import export_model
        from draft_training.model import load_draft
        from draft_training.trainer import configure_trainable

        configure_trainable(self.model, "all")
        with self.torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(0.0013)
        masters = {
            name: p.detach().cpu().clone() for name, p in self.model.named_parameters()
        }
        parameters = dict(self.model.named_parameters())
        export = self.root / "export"
        export_model(self.model, export, {}, family=False)
        exported = load_file(str(export / "model.safetensors"))
        for name, value in exported.items():
            self.torch.testing.assert_close(
                value, masters[name].bfloat16(), rtol=0, atol=0
            )
        restored = load_draft(export, tiny_config(), "mps")
        expected = evaluate_cached(
            restored, self.embedding, self.head, self.validation, self.recipe, "mps"
        )
        actual = evaluate_cached(
            self.model, self.embedding, self.head, self.validation, self.recipe, "mps"
        )
        self.assertEqual(actual, expected)
        for name, parameter in self.model.named_parameters():
            self.assertIs(parameter, parameters[name])
            self.assertEqual(parameter.dtype, self.torch.float32)
            self.torch.testing.assert_close(
                parameter.cpu(), masters[name], rtol=0, atol=0
            )
        with self.assertRaises(OSError):
            evaluate_cached(
                self.model,
                self.embedding,
                self.head,
                self.root / "missing",
                self.recipe,
                "mps",
            )
        for name, parameter in self.model.named_parameters():
            self.assertIs(parameter, parameters[name])
            self.assertEqual(parameter.dtype, self.torch.float32)
            self.torch.testing.assert_close(
                parameter.cpu(), masters[name], rtol=0, atol=0
            )

    def test_backbone_training_reduces_exported_bf16_unary_loss(self):
        from draft_training.capture import load_sample
        from draft_training.export import export_model
        from draft_training.losses import block_loss
        from draft_training.model import load_draft
        from draft_training.trainer import configure_trainable

        sample = load_sample(
            self.training,
            read_json(self.training / "manifest.json")["shards"][0],
            "mps",
        )

        def unary_loss(model):
            with self.torch.autocast("mps", dtype=self.torch.bfloat16):
                hidden = model.block_hidden(
                    sample["tokens"], sample["features"], 5, self.embedding
                )
                return block_loss(model, hidden, sample["tokens"], 5, self.head, 0)[0]

        before = float(unary_loss(self.model).detach().cpu())
        configure_trainable(self.model, "all")
        optimizer = self.torch.optim.AdamW(
            self.model.parameters(), lr=0.01, weight_decay=0
        )
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = unary_loss(self.model)
            loss.backward()
            optimizer.step()
        destination = self.root / "optimized-export"
        export_model(self.model, destination, {}, family=False)
        exported = load_draft(destination, tiny_config(), "mps")
        after = float(unary_loss(exported).detach().cpu())
        self.assertLess(after, before * 0.7)


@unittest.skipUnless(HAS_MPS, "Apple GPU MPS and optional dependencies are required")
class MPSPipelineTests(unittest.TestCase):
    def setUp(self):
        import torch

        from draft_training.data import split_manifest
        from draft_training.export import export_model

        torch.set_num_threads(1)
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target, self.draft = self.root / "target", self.root / "draft"
        tiny_target(self.target)
        model, _, _ = tiny_components()
        export_model(model, self.draft, {}, family=False)
        self.training, self.validation, self.test = (
            self.root / f"{name}.jsonl" for name in ("train", "validation", "test")
        )
        for i, path in enumerate((self.training, self.validation, self.test)):
            write_data(path, f"w{i + 5} w{i + 10}")
        self.recipe = replace(
            Recipe(),
            max_steps=3,
            sequence_length=64,
            max_new_tokens=12,
            gradient_accumulation=2,
            checkpoint_interval=1,
            validation_interval=1,
            max_validation_blocks=2,
            trainable="all",
            learning_rate=0.001,
        )
        # Replace only production dimensions. Tokenization, teacher loading,
        # generation, capture, optimization, serialization and replay are real.
        self.inspected = (
            model.raw_config,
            self.recipe,
            split_manifest(self.training),
            split_manifest(self.validation),
        )

    def arguments(self, output, command="train", *flags):
        return parse_args(
            [
                command,
                "--target",
                str(self.target),
                "--init-draft",
                str(self.draft),
                "--data",
                str(self.training),
                "--validation-data",
                str(self.validation),
                "--output",
                str(output),
                "--device",
                "mps",
                *flags,
            ]
        )

    def run_training(self, args):
        from draft_training.__main__ import run_training

        with mock.patch(
            "draft_training.__main__.inspect_training", return_value=self.inspected
        ):
            return run_training(args, family=False)

    def test_preflight_reports_memory_snapshots_without_creating_a_run(self):
        output = self.root / "preflight"
        result = self.run_training(self.arguments(output, "train", "--preflight"))
        self.assertEqual(result["mode"], "device-preflight")
        self.assertFalse(output.exists())
        self.assertIsNone(result["capture_peak_cuda_bytes"])
        self.assertIsNone(result["training_peak_cuda_bytes"])
        for phase in ("capture_memory", "training_memory"):
            self.assertEqual(result[phase]["device"], "mps")
            self.assertIn("not peak", result[phase]["measurement"])
            self.assertGreater(result[phase]["allocated_bytes"], 0)
            self.assertGreater(result[phase]["recommended_max_bytes"], 0)

    def test_capture_train_resume_export_and_final_evaluation(self):
        from safetensors.torch import load_file

        from draft_training.__main__ import run_evaluation
        from draft_training.config import validate_draft
        from draft_training.trainer import train

        resumed, uninterrupted = self.root / "resumed", self.root / "full"
        self.run_training(self.arguments(resumed, "capture"))
        self.assertFalse((resumed / "checkpoints").exists())
        with mock.patch(
            "draft_training.trainer.train",
            side_effect=lambda *args, **kwargs: train(*args, **kwargs, stop_after=1),
        ):
            self.run_training(self.arguments(resumed, "train", "--resume"))
        self.torch.manual_seed(999)
        self.run_training(self.arguments(resumed, "train", "--resume"))
        self.run_training(self.arguments(uninterrupted))
        for filename in (
            "checkpoints/step-00000003/model.safetensors",
            "export/model.safetensors",
        ):
            expected = load_file(str(uninterrupted / filename))
            actual = load_file(str(resumed / filename))
            for name in expected:
                self.torch.testing.assert_close(
                    actual[name], expected[name], rtol=0, atol=0
                )
        selection = read_json(resumed / "selection.json")
        self.assertEqual(selection["execution"]["device"], "mps")
        self.assertTrue(selection["execution"]["macos_version"])
        self.assertTrue(selection["execution"]["device_name"])
        evaluation = parse_args(
            [
                "evaluate",
                "--target",
                str(self.target),
                "--draft",
                str(resumed / "export"),
                "--baseline-draft",
                str(self.draft),
                "--test-data",
                str(self.test),
                "--selection-manifest",
                str(resumed / "selection.json"),
                "--output",
                str(resumed / "final-test"),
                "--device",
                "mps",
            ]
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
        self.assertEqual(
            read_json(evaluation.output / "test-report.json")["metrics"], metrics
        )
