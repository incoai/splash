import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional PyTorch dependencies")
class RecallAdaptationTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.t = torch
        torch.set_num_threads(1)

    def test_parameter_scopes_are_separate_and_depth_is_checked(self):
        from dev.tests.training.fixtures import tiny_components
        from draft_training.config import TrainingError
        from draft_training.trainer import configure_trainable

        t = self.t
        for mode in ("context", "late_backbone", "recall"):
            model, _, _ = tiny_components()
            configure_trainable(model, mode, 1)
            names = {name for name, p in model.named_parameters() if p.requires_grad}
            self.assertFalse(
                any(name.startswith("candidate_selector.") for name in names)
            )
            self.assertFalse(any(name.startswith("layers.0.") for name in names))
            self.assertEqual("fc.weight" in names, mode in ("context", "recall"))
            self.assertEqual(
                "layers.1.mlp.up_proj.weight" in names,
                mode in ("late_backbone", "recall"),
            )
            self.assertTrue(
                all(p.dtype == t.float32 for p in model.parameters() if p.requires_grad)
            )
        with self.assertRaises(TrainingError):
            configure_trainable(model, "recall", 3)
        with self.assertRaises(TrainingError):
            configure_trainable(model, "unknown")

    def test_reached_denominators_stop_at_first_rejection(self):
        from draft_training.evaluate import conditional_recall

        t = self.t
        labels = t.tensor([[1, 2, 3], [1, 2, 3]])
        paths = t.tensor([[1, 9, 3], [1, 2, 8]])
        candidates = t.tensor([[[1, 0], [2, 9], [3, 0]], [[1, 0], [2, 0], [8, 0]]])
        self.assertEqual(
            conditional_recall(paths, labels, candidates),
            dict(reached=[2, 2, 1], covered=[2, 2, 0], selected=[2, 1, 0]),
        )

    def test_actual_recall_adaptation_reduces_unary_loss_and_freezes_selector(self):
        from dev.tests.training.fixtures import tiny_components
        from draft_training.losses import block_loss
        from draft_training.trainer import configure_trainable

        t = self.t
        devices = ["cpu"] + (["mps"] if t.backends.mps.is_available() else [])
        for device in devices:
            for mode in ("context", "late_backbone", "recall"):
                with self.subTest(device=device, mode=mode):
                    model, embedding, head = (m.to(device) for m in tiny_components())
                    configure_trainable(model, mode, 1)
                    before = {
                        name: p.detach().clone()
                        for name, p in model.named_parameters()
                        if not p.requires_grad
                    }
                    frozen_head = head.weight.detach().clone()
                    tokens = t.arange(24, device=device)
                    features = t.randn(24, 32, device=device, dtype=t.bfloat16)
                    opt = t.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad],
                        lr=0.01,
                        weight_decay=0,
                    )
                    losses = []
                    for _ in range(15):
                        opt.zero_grad()
                        with t.autocast(device_type=device, dtype=t.bfloat16):
                            hidden = model.block_hidden(tokens, features, 5, embedding)
                            loss, _ = block_loss(model, hidden, tokens, 5, head, 0)
                        losses.append(float(loss.detach()))
                        loss.backward()
                        opt.step()
                    self.assertLess(losses[-1], losses[0])
                    for name, p in model.named_parameters():
                        if name in before:
                            t.testing.assert_close(p, before[name], rtol=0, atol=0)
                    t.testing.assert_close(head.weight, frozen_head, rtol=0, atol=0)

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
            trainable="recall",
            recall_last_layers=1,
            selector_loss_weight=0,
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

    def test_combined_affine_prefix_and_recall_updates_cpu_and_mps(self):
        from dataclasses import replace

        from torch import nn

        from dev.tests.training.fixtures import tiny_config
        from draft_training.config import Recipe
        from draft_training.losses import block_loss
        from draft_training.model import DraftModel
        from draft_training.quantization import enable_affine4
        from draft_training.trainer import configure_trainable

        t = self.t
        config = tiny_config()
        config.update(hidden_size=64, intermediate_size=128, num_attention_heads=8)
        for device in ["cpu"] + (["mps"] if t.backends.mps.is_available() else []):
            for scope in ("selector_context", "all"):
                with self.subTest(device=device, scope=scope):
                    t.manual_seed(4)
                    model = DraftModel(config).to(device=device, dtype=t.bfloat16)
                    embedding = (
                        nn.Embedding(32, 64)
                        .to(device=device, dtype=t.bfloat16)
                        .requires_grad_(False)
                    )
                    head = (
                        nn.Linear(64, 32, bias=False)
                        .to(device=device, dtype=t.bfloat16)
                        .requires_grad_(False)
                    )
                    recipe = replace(
                        Recipe(),
                        trainable="recall",
                        affine4_scope=scope,
                        prefix_loss_weight=0.3,
                        position_decay=0.8,
                    )
                    enable_affine4(model, recipe.affine4_scope)
                    configure_trainable(
                        model, recipe.trainable, recipe.recall_last_layers
                    )
                    frozen = {
                        name: p.detach().cpu().clone()
                        for name, p in model.named_parameters()
                        if not p.requires_grad
                    }
                    parameters = [p for p in model.parameters() if p.requires_grad]
                    optimizer = t.optim.AdamW(parameters, lr=0.01)
                    tokens = t.arange(24, device=device)
                    features = t.randn(24, 128, device=device, dtype=t.bfloat16)
                    before = model.fc.weight.detach().cpu().clone()
                    for _ in range(3):
                        optimizer.zero_grad()
                        with t.autocast(device, dtype=t.bfloat16):
                            hidden = model.block_hidden(tokens, features, 5, embedding)
                            loss, _ = block_loss(
                                model,
                                hidden,
                                tokens,
                                5,
                                head,
                                1,
                                prefix_weight=recipe.prefix_loss_weight,
                                position_decay=recipe.position_decay,
                            )
                        self.assertTrue(t.isfinite(loss))
                        loss.backward()
                        self.assertTrue(
                            all(
                                p.grad is None or t.isfinite(p.grad).all()
                                for p in parameters
                            )
                        )
                        optimizer.step()
                    self.assertFalse(t.equal(before, model.fc.weight.detach().cpu()))
                    for name, p in model.named_parameters():
                        if name in frozen:
                            t.testing.assert_close(
                                p.cpu(), frozen[name], rtol=0, atol=0
                            )

    def test_combined_recipe_roundtrip_resume_and_bf16_export(self):
        import json
        from dataclasses import replace

        from safetensors.torch import load_file, save_file
        from torch import nn

        from dev.tests.training.fixtures import tiny_config
        from dev.tests.training.test_training import TrainingTests
        from draft_training.config import Recipe, file_digest, read_json, write_json
        from draft_training.model import DraftModel

        t = self.t
        fixture = TrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)

        def components():
            t.manual_seed(7)
            config = tiny_config()
            config.update(hidden_size=64, intermediate_size=128, num_attention_heads=8)
            return (
                DraftModel(config).bfloat16(),
                nn.Embedding(32, 64).bfloat16().requires_grad_(False),
                nn.Linear(64, 32, bias=False).bfloat16().requires_grad_(False),
            )

        fixture.model, fixture.embedding, fixture.head = components()
        for directory in (fixture.training, fixture.validation):
            path = directory / "000000.safetensors"
            sample = load_file(str(path))
            sample["features"] = t.randn(24, 128).bfloat16()
            save_file(sample, str(path))
            manifest = read_json(directory / "manifest.json")
            manifest["shards"][0]["sha256"] = file_digest(path)
            write_json(directory / "manifest.json", manifest)
        recipe = replace(
            fixture.recipe,
            max_steps=4,
            gradient_accumulation=1,
            trainable="recall",
            recall_last_layers=1,
            affine4_scope="all",
            prefix_loss_weight=0.3,
            position_decay=0.8,
        )
        recipe_path = fixture.root / "combined.yaml"
        recipe_path.write_text(json.dumps(recipe.to_dict()))
        fixture.recipe = Recipe.load(recipe_path)
        self.assertEqual(fixture.recipe, recipe)
        full, resumed = (
            fixture.root / "combined-full",
            fixture.root / "combined-resumed",
        )
        fixture.run_train(fixture.model, full)
        model, _, _ = components()
        fixture.run_train(model, resumed, stop_after=2)
        model, _, _ = components()
        fixture.run_train(model, resumed, resume=True)
        actual = load_file(str(resumed / "export/model.safetensors"))
        for name, value in load_file(str(full / "export/model.safetensors")).items():
            t.testing.assert_close(value, actual[name], rtol=0, atol=0)
