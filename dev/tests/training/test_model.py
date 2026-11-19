import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from dev.tests.training.fixtures import cached_features, tiny_components, tiny_config

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional training dependencies are not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(1)
        self.torch = torch
        self.model, self.embedding, self.head = tiny_components()
        self.tokens = torch.arange(32) % 30
        self.features = torch.randn(32, 32, dtype=torch.bfloat16)

    def test_future_tokens_features_and_later_blocks_cannot_affect_logits(self):
        for anchor in (2, 20):
            with self.subTest(anchor=anchor), self.torch.no_grad():
                before = self.model.block_hidden(
                    self.tokens, self.features, anchor, self.embedding
                )
                tokens = self.tokens.clone()
                tokens[anchor + 1 :] = 29
                features = self.features.clone()
                features[anchor:] = self.torch.randn_like(features[anchor:]) * 100
                after = self.model.block_hidden(
                    tokens, features, anchor, self.embedding
                )
                self.torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.torch.testing.assert_close(
                    self.head(before), self.head(after), rtol=0, atol=0
                )
                self.torch.testing.assert_close(
                    self.model.candidate_selector.hidden_projection(before),
                    self.model.candidate_selector.hidden_projection(after),
                    rtol=0,
                    atol=0,
                )
        # Negative control: exposing future target states changes predictions.
        positions = self.torch.arange(40)[None]
        visible = self.torch.ones(1, 1, 8, 40, dtype=self.torch.bool)
        noise = self.embedding(self.tokens.new_full((1, 8), 31))
        before = self.model(self.features[None], noise, positions, visible)
        corrupted = self.features.clone()
        corrupted[20:] *= -30
        after = self.model(corrupted[None], noise, positions, visible)
        self.assertGreater(float((before - after).detach().abs().max()), 0.001)

    def test_future_features_have_no_gradient_through_loss(self):
        from draft_training.losses import block_loss

        features = self.features.clone().requires_grad_(True)
        hidden = self.model.block_hidden(self.tokens, features, 10, self.embedding)
        loss, _ = block_loss(self.model, hidden, self.tokens, 10, self.head, 1)
        loss.backward()
        self.assertEqual(int(self.torch.count_nonzero(features.grad[10:])), 0)
        self.assertGreater(float(features.grad[:10].abs().sum()), 0)
        self.assertIsNone(self.head.weight.grad)
        self.assertGreater(
            float(self.model.layers[0].self_attn.q_proj.weight.grad.abs().sum()), 0
        )

    def test_bf16_conversion_preserves_fp32_rotary_frequencies_at_long_positions(self):
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        expected = Qwen3RotaryEmbedding(Qwen3Config(**tiny_config()))
        self.assertEqual(self.model.rotary_emb.inv_freq.dtype, self.torch.float32)
        self.torch.testing.assert_close(
            self.model.rotary_emb.inv_freq, expected.inv_freq, rtol=0, atol=0
        )
        positions = self.torch.arange(10000, 10008)[None]
        hidden = self.torch.zeros(1, 8, 16, dtype=self.torch.bfloat16)
        want = expected(hidden, positions)
        got = self.model.rotary_emb(hidden, positions)
        for a, b in zip(got, want):
            self.torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_top_candidates_match_native_tie_order(self):
        from draft_training.model import top_candidates

        scores, ids = top_candidates(self.torch.tensor([[1.0, 3.0, 3.0, 3.0, 0.0]]), 2)
        self.assertEqual(ids.tolist(), [[1, 2]])
        self.assertEqual(scores.tolist(), [[3.0, 3.0]])

    def test_strict_selector_zero_coverage_has_finite_loss(self):
        from draft_training.losses import block_loss

        with self.torch.no_grad():
            self.head.weight.zero_()
        # All logits tie, so top four are IDs 0..3 and label 29 is absent.
        tokens = self.tokens.new_full((32,), 29)
        hidden = self.model.block_hidden(tokens, self.features, 10, self.embedding)
        loss, metrics = block_loss(self.model, hidden, tokens, 10, self.head, 1)
        self.assertTrue(self.torch.isfinite(loss))
        self.assertEqual(metrics["covered"], 0)
        self.assertEqual(metrics["selector_ce"], 0)
        loss.backward()

    def test_loader_and_loss_ignore_future_cached_features(self):
        from draft_training.capture import load_sample
        from draft_training.losses import block_loss

        with tempfile.TemporaryDirectory() as temporary:
            sample, manifest = cached_features(Path(temporary) / "features")
            loaded = load_sample(Path(temporary) / "features", manifest["shards"][0])
            h1 = self.model.block_hidden(
                loaded["tokens"], loaded["features"], 5, self.embedding
            )
            loss1, _ = block_loss(self.model, h1, loaded["tokens"], 5, self.head, 1)
            loaded["features"][5:] *= -100
            h2 = self.model.block_hidden(
                loaded["tokens"], loaded["features"], 5, self.embedding
            )
            loss2, _ = block_loss(self.model, h2, loaded["tokens"], 5, self.head, 1)
            self.torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)
            self.assertEqual(sample["tokens"].tolist(), loaded["tokens"].tolist())

    @unittest.skipUnless(
        os.environ.get("DFLASH_REFERENCE"),
        "set DFLASH_REFERENCE to pinned dflash/model.py",
    )
    def test_pinned_upstream_reference_forward_parity(self):
        from transformers import Qwen3Config

        from draft_training.model import DraftModel

        path = Path(os.environ["DFLASH_REFERENCE"])
        spec = importlib.util.spec_from_file_location("dflash_reference", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        config = tiny_config()
        self.torch.manual_seed(21)
        upstream = reference.DFlash2DraftModel(Qwen3Config(**config)).float().eval()
        # The public module leaves convolution bases empty for checkpoint load.
        with self.torch.no_grad():
            for layer in upstream.layers:
                for conv in (layer.attention_conv, layer.mlp_conv):
                    conv.base_kernel.normal_(0, 0.02)
                    conv.base_kernel[:, 0] += 1
        state = {
            k.removesuffix(".weight") if "codebook.weight" in k else k: v
            for k, v in upstream.state_dict().items()
        }
        model = DraftModel(config).float().eval()
        model.load_state_dict(state, strict=True)
        for context_length in (2, 20):
            with self.subTest(context_length=context_length), self.torch.no_grad():
                features = self.torch.randn(1, context_length, 32)
                noise = self.torch.randn(1, 8, 16)
                positions = self.torch.arange(context_length + 8)[None]
                queries = self.torch.arange(context_length, context_length + 8)
                visible = (queries[:, None] - positions[0, None]) < 16
                want = upstream(
                    position_ids=positions,
                    target_hidden=features,
                    noise_embedding=noise,
                    attention_mask=visible[None, None],
                )
                got = model(features, noise, positions, visible[None, None])
                self.torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
                head = self.torch.nn.Linear(16, 32, bias=False)
                anchor = self.torch.tensor([3])
                want_path = upstream.propose(want[:, 1:], anchor, head, 0)[0]
                got_path = model.candidate_selector.path(
                    got[:, 1:], head(got[:, 1:]), anchor
                )
                self.torch.testing.assert_close(got_path, want_path, rtol=0, atol=0)
