import importlib.util
import itertools
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional PyTorch dependencies")
class PrefixLossTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.t = torch
        torch.set_num_threads(1)

    def test_prefix_sum_matches_exhaustive_independent_survival(self):
        from draft_training.losses import prefix_survival

        t = self.t
        ps = [0.2, 0.7, 0.4]
        brute = 0.0
        for bits in itertools.product((0, 1), repeat=3):
            probability = 1.0
            length = 0
            reached = True
            for bit, p in zip(bits, ps):
                probability *= p if bit else 1 - p
                reached = reached and bool(bit)
                length += reached
            brute += probability * length
        p = t.tensor(ps, dtype=t.float64, requires_grad=True)
        result = prefix_survival(p, t.ones(3, dtype=t.bool))
        self.assertAlmostEqual(float(result.detach()), brute)
        self.assertTrue(
            t.autograd.gradcheck(
                lambda x: prefix_survival(x, t.ones(3, dtype=t.bool)), (p,)
            )
        )

    def test_support_misses_and_censoring_stop_prefix(self):
        from draft_training.losses import prefix_survival

        t = self.t
        self.assertEqual(
            float(prefix_survival(t.tensor([0.5, 0.0, 1.0]), t.ones(3, dtype=t.bool))),
            0.5,
        )
        self.assertEqual(
            float(
                prefix_survival(
                    t.tensor([0.5, 1.0, 1.0]), t.tensor([True, False, True])
                )
            ),
            0.5,
        )
        p = t.tensor([0.5, 0.7, 0.9], requires_grad=True)
        prefix_survival(p, t.tensor([True, False, False])).backward()
        t.testing.assert_close(p.grad, t.tensor([1.0, 0.0, 0.0]))

    def test_chunking_defaults_and_masked_sentinels(self):
        from dev.tests.training.fixtures import tiny_components
        from draft_training.losses import block_loss

        t = self.t
        model, _, head = tiny_components()
        hidden = t.randn(1, 7, 16, dtype=t.bfloat16)
        tokens = t.arange(8)
        a, ma = block_loss(model, hidden, tokens, 0, head, 1, 1)
        b, mb = block_loss(model, hidden, tokens, 0, head, 1, 7)
        t.testing.assert_close(a, b)
        self.assertEqual(ma["tokens"], mb["tokens"])
        tokens[-2:] = -999
        mask = t.tensor([True] * 5 + [False] * 2)
        loss, metrics = block_loss(
            model,
            hidden,
            tokens,
            0,
            head,
            1,
            3,
            prefix_weight=0.5,
            position_decay=0.8,
            label_mask=mask,
        )
        self.assertTrue(t.isfinite(loss))
        self.assertEqual(metrics["tokens"], 5)
        self.assertLessEqual(metrics["prefix_survival_surrogate"], 5)

    def test_auxiliary_objective_has_gradients_and_changes_training(self):
        from dev.tests.training.fixtures import tiny_components
        from draft_training.losses import block_loss
        from draft_training.trainer import configure_trainable

        t = self.t
        model, _, head = tiny_components()
        configure_trainable(model, "selector")
        hidden = t.randn(1, 7, 16, dtype=t.bfloat16)
        # Natural top candidates, without injecting a target into support.
        tokens = t.cat((t.zeros(1, dtype=t.long), head(hidden)[0].argmax(-1)))
        with t.autocast("cpu", dtype=t.bfloat16):
            base, _ = block_loss(model, hidden, tokens, 0, head, 1)
            prefix, metrics = block_loss(
                model, hidden, tokens, 0, head, 1, prefix_weight=1
            )
        self.assertLess(float(prefix.detach()), float(base.detach()))
        (prefix - base).backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
            )
        )
        self.assertGreater(metrics["prefix_survival_surrogate"], 0)

    def test_nondefault_recipe_resume_and_export(self):
        from dataclasses import replace

        from safetensors.torch import load_file

        from dev.tests.training.fixtures import tiny_components
        from dev.tests.training.test_training import TrainingTests

        fixture = TrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.recipe = replace(
            fixture.recipe,
            max_steps=4,
            gradient_accumulation=1,
            prefix_loss_weight=0.3,
            position_decay=0.8,
        )
        components = tiny_components
        full, resumed = fixture.root / "new-full", fixture.root / "new-resumed"
        fixture.run_train(fixture.model, full)
        model, _, _ = components()
        fixture.run_train(model, resumed, stop_after=2)
        model, _, _ = components()
        fixture.run_train(model, resumed, resume=True)
        for name, value in load_file(str(full / "export/model.safetensors")).items():
            fixture.torch.testing.assert_close(
                value,
                load_file(str(resumed / "export/model.safetensors"))[name],
                rtol=0,
                atol=0,
            )

    def test_auxiliary_gradient_optimization_on_cpu_and_mps(self):
        from dev.tests.training.fixtures import tiny_components
        from draft_training.losses import block_loss
        from draft_training.trainer import configure_trainable

        t = self.t
        for device in ["cpu"] + (["mps"] if t.backends.mps.is_available() else []):
            with self.subTest(device=device):
                model, _, head = (value.to(device) for value in tiny_components())
                configure_trainable(model, "selector")
                hidden = t.randn(1, 7, 16, device=device, dtype=t.bfloat16)
                tokens = t.cat(
                    (
                        t.zeros(1, device=device, dtype=t.long),
                        head(hidden)[0].argmax(-1),
                    )
                )
                optimizer = t.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=0.03
                )
                rewards = []
                for _ in range(20):
                    optimizer.zero_grad()
                    with t.autocast(device, dtype=t.bfloat16):
                        loss, metrics = block_loss(
                            model, hidden, tokens, 0, head, 0, prefix_weight=1
                        )
                    rewards.append(metrics["prefix_survival_surrogate"])
                    loss.backward()
                    optimizer.step()
                self.assertGreater(rewards[-1], rewards[0])
