import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional PyTorch dependencies")
class SelectorExperimentTests(unittest.TestCase):
    def setUp(self):
        import torch

        from dev.tests.training.fixtures import tiny_components

        self.t = torch
        torch.set_num_threads(1)
        model, _, _ = tiny_components()
        self.baseline = model.candidate_selector.float()
        self.block = torch.randn(2, 8, 16)
        self.logits = torch.randn(2, 7, 32)
        self.labels = self.logits.argmax(-1)
        self.anchor = torch.tensor([3, 9])

    def test_zero_initialized_candidates_match_baseline_scores_and_walk(self):
        from draft_training.model import top_candidates
        from draft_training.selector_experiments import ExperimentalSelector

        t = self.t
        unary, candidates = top_candidates(self.logits, 4)
        prev = t.cat((self.anchor[:, None], self.labels[:, :-1]), 1)
        expected = self.baseline.scores(self.block[:, 1:], unary, candidates, prev)
        for variant in ("bilinear", "pair_context", "nonlinear"):
            experiment = ExperimentalSelector(
                copy.deepcopy(self.baseline), variant=variant
            )
            t.testing.assert_close(
                experiment.scores(self.block, unary, candidates, prev),
                expected,
                rtol=0,
                atol=0,
            )
            t.testing.assert_close(
                experiment.path(self.block, self.logits, self.anchor),
                self.baseline.path(self.block[:, 1:], self.logits, self.anchor),
            )

    def test_anchor_and_lane_alignment_do_not_wrap(self):
        from draft_training.model import top_candidates
        from draft_training.selector_experiments import ExperimentalSelector

        t = self.t
        experiment = ExperimentalSelector(self.baseline, variant="pair_context")
        with t.no_grad():
            experiment.previous_projection.weight.fill_(0.2)
            experiment.base.successor_codebook.fill_(1)
        unary, candidates = top_candidates(self.logits, 4)
        prev = t.cat((self.anchor[:, None], self.labels[:, :-1]), 1)
        block = self.block.clone().requires_grad_()
        score = experiment.scores(block, unary, candidates, prev)[0, 0].sum()
        score.backward()
        self.assertGreater(float(block.grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(block.grad[1].abs().sum()), 0)
        self.assertEqual(float(block.grad[0, 2:].abs().sum()), 0)

    def test_training_loss_decreases_cpu_and_mps_for_each_variant(self):
        from draft_training.selector_experiments import ExperimentalSelector

        t = self.t
        devices = ["cpu"] + (["mps"] if t.backends.mps.is_available() else [])
        for device in devices:
            for variant in ("bilinear", "pair_context", "nonlinear"):
                with self.subTest(device=device, variant=variant):
                    exp = ExperimentalSelector(
                        copy.deepcopy(self.baseline), variant=variant
                    ).to(device)
                    opt = t.optim.AdamW(exp.parameters(), lr=0.025)
                    losses = []
                    for _ in range(20):
                        opt.zero_grad()
                        loss, metrics = exp.loss(
                            self.block.to(device),
                            self.logits.to(device),
                            self.labels.to(device),
                            self.anchor.to(device),
                        )
                        losses.append(float(loss.detach()))
                        loss.backward()
                        opt.step()
                    self.assertLess(losses[-1], losses[0])
                    self.assertEqual(metrics["covered"], 14)

    def test_roundtrip_explicit_capability_and_bf16_forward(self):
        from draft_training.config import TrainingError
        from draft_training.selector_experiments import ExperimentalSelector

        t = self.t
        for variant in ("pair_context", "nonlinear"):
            exp = ExperimentalSelector(copy.deepcopy(self.baseline), variant=variant)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "candidate"
                exp.save_research(output, "sha256-baseline")
                restored = ExperimentalSelector(
                    copy.deepcopy(self.baseline), variant=variant
                )
                restored.load_research(output, "sha256-baseline")
                self.assertFalse((output / "config.json").exists())
                for name, value in exp.state_dict().items():
                    t.testing.assert_close(
                        value, restored.state_dict()[name], rtol=0, atol=0
                    )
                with self.assertRaises(TrainingError):
                    restored.load_research(output, "different-baseline")
                metadata = json.loads((output / "experiment.json").read_text())
                metadata["capability"] = "unrecognized"
                (output / "experiment.json").write_text(json.dumps(metadata))
                with self.assertRaises(TrainingError):
                    restored.load_research(output, "sha256-baseline")
            with t.autocast("cpu", dtype=t.bfloat16):
                loss, _ = exp.loss(
                    self.block.bfloat16(), self.logits, self.labels, self.anchor
                )
            self.assertTrue(t.isfinite(loss))
            loss.backward()

    def test_natural_support_misses_have_zero_loss(self):
        from draft_training.selector_experiments import ExperimentalSelector

        t = self.t
        exp = ExperimentalSelector(self.baseline, variant="pair_context")
        logits = t.zeros(2, 7, 32)
        logits[..., :4] = 10
        loss, metrics = exp.loss(self.block, logits, t.full((2, 7), 31), self.anchor)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(metrics["covered"], 0)

    def test_failed_save_is_atomic_and_can_be_retried(self):
        from unittest import mock

        from draft_training.config import TrainingError
        from draft_training.selector_experiments import ExperimentalSelector

        exp = ExperimentalSelector(self.baseline, variant="pair_context")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            with mock.patch(
                "safetensors.torch.save_file", side_effect=OSError("disk full")
            ):
                with self.assertRaises(OSError):
                    exp.save_research(output, "baseline")
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(directory).glob(".research-selector-*")))
            exp.save_research(output, "baseline")
            with self.assertRaises(TrainingError):
                exp.save_research(output, "baseline")

    def test_shape_failure_does_not_partially_replace_weights(self):
        from safetensors.torch import load_file, save_file

        from draft_training.config import TrainingError
        from draft_training.selector_experiments import ExperimentalSelector

        t = self.t
        exp = ExperimentalSelector(self.baseline, variant="pair_context")
        before = {k: v.clone() for k, v in exp.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            exp.save_research(output, "baseline")
            state = load_file(str(output / "research-selector.safetensors"))
            names = list(state)
            state[names[0]] = state[names[0]] + 1
            state[names[-1]] = state[names[-1]].flatten()[:1]
            save_file(state, str(output / "research-selector.safetensors"))
            with self.assertRaises(TrainingError):
                exp.load_research(output, "baseline")
        for name, value in exp.state_dict().items():
            t.testing.assert_close(value, before[name], rtol=0, atol=0)
