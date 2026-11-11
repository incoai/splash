import importlib.util
import struct
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional PyTorch dependencies")
class QuantizationTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        torch.set_num_threads(1)

    def test_packed_groups_match_independent_native_preparation_oracle(self):
        from dev.tests.engine.run_draft_preparation import quantized_group
        from draft_training.quantization import affine4_groups

        t = self.torch
        t.manual_seed(15)
        groups = [
            [0.0] * 64,
            [0.75] * 64,
            [-0.75] * 64,
            [2.5, 8.5, *[float(i % 16) for i in range(62)]],
            [-2.5, -8.5, *[-float(i % 15 + 1) for i in range(62)]],
            [1e-8 * (i - 32) for i in range(64)],
        ] + t.randn(80, 64).tolist()
        w = t.tensor(groups).bfloat16()
        if t.backends.mps.is_available():
            native = affine4_groups(w.to("mps"))
            for cpu, gpu in zip(affine4_groups(w), native):
                t.testing.assert_close(cpu, gpu.cpu(), rtol=0, atol=0)
        codes, scales, biases = affine4_groups(w)
        for i, values in enumerate(w.float().tolist()):
            c = codes[i, 0].tolist()
            packed = struct.pack(
                "<HH",
                int(scales[i, 0, 0].view(t.int16)) & 65535,
                int(biases[i, 0, 0].view(t.int16)) & 65535,
            )
            packed += bytes(c[j] | c[j + 1] << 4 for j in range(0, 64, 2))
            self.assertEqual(packed, quantized_group(values))

    def test_forward_exact_and_straight_through_preserves_small_updates(self):
        from draft_training.quantization import affine4_groups, fake_affine4

        t = self.torch
        weight = t.nn.Parameter(t.linspace(-1, 1, 128).reshape(2, 64))
        codes, scales, biases = affine4_groups(weight)
        expected = (codes.float() * scales.float() + biases.float()).reshape_as(weight)
        t.testing.assert_close(fake_affine4(weight), expected, rtol=0, atol=0)
        fake_affine4(weight).sum().backward()
        t.testing.assert_close(weight.grad, t.ones_like(weight))
        before = weight.detach().clone()
        t.optim.SGD([weight], lr=1e-6).step()
        self.assertFalse(t.equal(before, weight))
        self.assertEqual(weight.dtype, t.float32)

    def test_scope_and_state_contract(self):
        from draft_training.config import TrainingError
        from draft_training.quantization import Affine4Linear, enable_affine4

        t = self.torch
        model = t.nn.Module()
        model.fc = t.nn.Linear(64, 64, bias=False)
        model.candidate_selector = t.nn.Module()
        model.candidate_selector.hidden_projection = t.nn.Linear(64, 4, bias=False)
        model.other = t.nn.Linear(64, 64, bias=False)
        model.codebook = t.nn.Parameter(t.ones(32, 4))
        before = {k: id(p) for k, p in model.named_parameters()}
        rng = t.get_rng_state().clone()
        self.assertEqual(len(enable_affine4(model, "selector_context")), 2)
        self.assertTrue(t.equal(rng, t.get_rng_state()))
        self.assertIsInstance(model.fc, Affine4Linear)
        self.assertNotIsInstance(model.other, Affine4Linear)
        self.assertEqual(before, {k: id(p) for k, p in model.named_parameters()})
        self.assertEqual(set(before), set(model.state_dict()))
        enable_affine4(model, "off")
        self.assertNotIsInstance(model.fc, Affine4Linear)
        self.assertNotIsInstance(
            model.candidate_selector.hidden_projection, Affine4Linear
        )
        self.assertEqual(before, {k: id(p) for k, p in model.named_parameters()})
        with self.assertRaises(TrainingError):
            enable_affine4(model, "unknown")
        with self.assertRaises(TrainingError):
            from draft_training.quantization import fake_affine4

            fake_affine4(t.ones(1, 63))
        with self.assertRaises(TrainingError):
            fake_affine4(t.full((1, 64), float("nan")))

    def test_actual_optimization_cpu_and_mps(self):
        from draft_training.quantization import Affine4Linear

        t = self.torch
        devices = ["cpu"] + (["mps"] if t.backends.mps.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                t.manual_seed(3)
                model = Affine4Linear(64, 4, bias=False).to(device)
                x = t.randn(8, 64, device=device).bfloat16()
                y = t.zeros(8, 4, device=device)
                opt = t.optim.AdamW(model.parameters(), lr=0.01)
                losses = []
                for _ in range(12):
                    opt.zero_grad()
                    loss = (model(x).float() - y).square().mean()
                    losses.append(float(loss.detach()))
                    loss.backward()
                    opt.step()
                self.assertLess(losses[-1], losses[0])
                self.assertTrue(all(p.dtype == t.float32 for p in model.parameters()))

    def test_nondefault_recipe_resume_and_export(self):
        from dataclasses import replace

        from safetensors.torch import load_file

        from dev.tests.training.test_training import TrainingTests

        fixture = TrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from safetensors.torch import save_file
        from torch import nn

        from dev.tests.training.fixtures import tiny_config
        from draft_training.config import file_digest, read_json, write_json
        from draft_training.model import DraftModel

        t = self.torch

        def components():
            t.manual_seed(7)
            config = tiny_config()
            config["hidden_size"] = 64
            model = DraftModel(config).bfloat16()
            embedding = nn.Embedding(32, 64).bfloat16().requires_grad_(False)
            head = nn.Linear(64, 32, bias=False).bfloat16().requires_grad_(False)
            return model, embedding, head

        fixture.model, fixture.embedding, fixture.head = components()
        for directory in (fixture.training, fixture.validation):
            shard = directory / "000000.safetensors"
            sample = load_file(str(shard))
            sample["features"] = t.randn(24, 128).bfloat16()
            save_file(sample, str(shard))
            manifest = read_json(directory / "manifest.json")
            manifest["shards"][0]["sha256"] = file_digest(shard)
            write_json(directory / "manifest.json", manifest)
        fixture.recipe = replace(
            fixture.recipe,
            max_steps=4,
            gradient_accumulation=1,
            affine4_scope="selector_context",
        )
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

    def test_half_away_preserves_large_integral_float32_values(self):
        from draft_training.quantization import half_away

        t = self.torch
        values = t.tensor([-0.5, 0.5, -2.5, 2.5, 8388609.0, -8388609.0])
        expected = t.tensor([-1.0, 1.0, -3.0, 3.0, 8388609.0, -8388609.0])
        t.testing.assert_close(half_away(values), expected, rtol=0, atol=0)

    def test_evaluation_scope_switch_restores_optimizer_parameters_after_failure(self):
        import tempfile
        from dataclasses import replace
        from pathlib import Path

        from draft_training.config import Recipe
        from draft_training.evaluate import evaluate_cached
        from draft_training.quantization import Affine4Linear

        t = self.torch
        devices = ["cpu"] + (["mps"] if t.backends.mps.is_available() else [])
        with tempfile.TemporaryDirectory() as directory:
            for device in devices:
                with self.subTest(device=device):
                    model = t.nn.Module()
                    model.fc = t.nn.Linear(64, 64, bias=False).to(device)
                    model.candidate_selector = t.nn.Module()
                    model.candidate_selector.hidden_projection = t.nn.Linear(
                        64, 4, bias=False
                    ).to(device)
                    parameters = dict(model.named_parameters())
                    optimizer = t.optim.AdamW(model.parameters(), lr=0.001)
                    for scope in ("selector_context", "off", "all", "off"):
                        values = {
                            k: p.detach().cpu().clone()
                            for k, p in model.named_parameters()
                        }
                        with self.assertRaises(FileNotFoundError):
                            evaluate_cached(
                                model,
                                None,
                                None,
                                Path(directory) / "absent",
                                replace(Recipe(), affine4_scope=scope),
                                device,
                            )
                        self.assertEqual(
                            isinstance(model.fc, Affine4Linear), scope != "off"
                        )
                        for name, parameter in model.named_parameters():
                            self.assertIs(parameter, parameters[name])
                            self.assertEqual(parameter.dtype, t.float32)
                            t.testing.assert_close(
                                parameter.cpu(), values[name], rtol=0, atol=0
                            )
                        optimizer.zero_grad()
                        with t.autocast(device, dtype=t.bfloat16):
                            loss = (
                                model.fc(t.ones(1, 64, device=device))
                                .float()
                                .square()
                                .sum()
                            )
                        loss.backward()
                        optimizer.step()
                        self.assertFalse(
                            t.equal(model.fc.weight.detach().cpu(), values["fc.weight"])
                        )
