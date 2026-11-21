import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from draft_training.config import (
    TrainingError,
    expected_tensors,
    read_json,
    validate_draft,
    validate_pair,
    validate_target,
    write_json,
)

from . import test_capture as capture_tests
from .fixtures import tiny_target
from .test_metadata import sparse_checkpoint

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "dev/tests/fixtures/model-configs"


def moe_shapes(config, *, packed):
    text = config["text_config"]
    h, v = text["hidden_size"], text["vocab_size"]
    n, m, s = (
        text["num_experts"],
        text["moe_intermediate_size"],
        text["shared_expert_intermediate_size"],
    )
    shapes = {
        "model.language_model.embed_tokens.weight": [v, h],
        "lm_head.weight": [v, h],
    }
    for layer in range(text["num_hidden_layers"]):
        p = f"model.language_model.layers.{layer}."
        shapes.update(
            {
                p + "input_layernorm.weight": [h],
                p + "post_attention_layernorm.weight": [h],
                p + "mlp.gate.weight": [n, h],
                p + "mlp.shared_expert_gate.weight": [1, h],
                p + "mlp.shared_expert.gate_proj.weight": [s, h],
                p + "mlp.shared_expert.up_proj.weight": [s, h],
                p + "mlp.shared_expert.down_proj.weight": [h, s],
            }
        )
        if packed:
            shapes.update(
                {
                    p + "mlp.experts.gate_up_proj": [n, 2 * m, h],
                    p + "mlp.experts.down_proj": [n, h, m],
                }
            )
        else:
            for expert in range(n):
                e = p + f"mlp.experts.{expert}."
                shapes.update(
                    {
                        e + "gate_proj.weight": [m, h],
                        e + "up_proj.weight": [m, h],
                        e + "down_proj.weight": [h, m],
                    }
                )
    return shapes


class FamilyMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = read_json(FIXTURES / "qwen3.6-35b-a3b/config.json")
        self.target.pop("quantization", None)
        self.target.pop("quantization_config", None)
        write_json(self.root / "config.json", self.target)
        write_json(self.root / "tokenizer.json", {})
        write_json(self.root / "tokenizer_config.json", {"chat_template": "template"})

    def test_both_released_draft_contracts_and_moe_layer_id_types(self):
        for name in ("qwen3.8-27b", "qwen3.6-35b-a3b"):
            with self.subTest(family=name):
                draft = read_json(FIXTURES / name / "draft/config.json")
                directory = self.root / name
                sparse_checkpoint(directory, expected_tensors(draft))
                write_json(directory / "config.json", draft)
                self.assertEqual(validate_draft(directory), draft)
        draft["dflash_config"]["target_layer_ids"][0] = True
        write_json(directory / "config.json", draft)
        with self.assertRaisesRegex(TrainingError, "dflash_config"):
            validate_draft(directory)

    def test_moe_target_accepts_both_expert_storage_layouts(self):
        for packed in (True, False):
            with self.subTest(packed=packed):
                sparse_checkpoint(self.root, moe_shapes(self.target, packed=packed))
                self.assertEqual(validate_target(self.root), self.target)

    def test_missing_expert_and_incompatible_router_are_rejected(self):
        shapes = moe_shapes(self.target, packed=False)
        del shapes["model.language_model.layers.39.mlp.experts.255.down_proj.weight"]
        sparse_checkpoint(self.root, shapes)
        with self.assertRaisesRegex(TrainingError, "experts.255.down_proj"):
            validate_target(self.root)
        self.target["text_config"]["num_experts_per_tok"] = 2
        write_json(self.root / "config.json", self.target)
        with self.assertRaisesRegex(TrainingError, "num_experts_per_tok"):
            validate_target(self.root)

    def test_cross_family_pair_is_rejected(self):
        draft = read_json(FIXTURES / "qwen3.8-27b/draft/config.json")
        with self.assertRaisesRegex(TrainingError, "incompatible model families"):
            validate_pair(self.root, draft)
        draft = read_json(FIXTURES / "qwen3.6-35b-a3b/draft/config.json")
        validate_pair(self.root, draft)
        draft["dflash_config"]["target_layer_ids"][-1] = 40
        with self.assertRaisesRegex(TrainingError, "incompatible model families"):
            validate_pair(self.root, draft)

    def test_quantized_or_unknown_moe_teacher_is_rejected(self):
        self.target["quantization_config"] = {"bits": 4}
        write_json(self.root / "config.json", self.target)
        with self.assertRaisesRegex(TrainingError, "unquantized BF16"):
            validate_target(self.root)
        self.target["model_type"] = "unregistered_moe"
        write_json(self.root / "config.json", self.target)
        with self.assertRaisesRegex(TrainingError, "supported Transformers"):
            validate_target(self.root)


class MoECaptureTests(capture_tests.CaptureTests):
    """Run the complete capture/training/export/evaluation contract on a real tiny MoE."""

    def setUp(self):
        super().setUp()
        from draft_training.capture import TargetAdapter

        self.target = self.root / "moe-target"
        self.reference, self.tokenizer = tiny_target(self.target, moe=True)
        self.adapter = TargetAdapter.load(self.target, [0, 2], "cpu")
        self.recipe = replace(self.recipe, trainable="all")

    def test_moe_loader_uses_sparse_teacher_and_freezes_routers(self):
        self.assertEqual(
            type(self.adapter.model).__name__, "Qwen3_5MoeForConditionalGeneration"
        )
        router = self.adapter.decoder.layers[0].mlp.gate
        self.assertEqual(tuple(router.weight.shape), (4, 16))
        self.assertFalse(router.weight.requires_grad)
        self.assertFalse(any(p.requires_grad for p in self.adapter.model.parameters()))

    def test_moe_loader_accepts_packed_and_individual_experts(self):
        from safetensors.torch import save_file

        from draft_training.capture import TargetAdapter

        tokens = self.torch.arange(16) % 28
        expected = self.adapter.features(tokens)
        path = self.target / "model.safetensors"
        state = {
            name: value.detach().clone().contiguous()
            for name, value in self.reference.state_dict().items()
        }
        save_file(state, str(path))
        packed = TargetAdapter.load(self.target, [0, 2], "cpu")
        self.torch.testing.assert_close(
            packed.features(tokens), expected, rtol=0, atol=0
        )
        for layer in range(4):
            prefix = f"model.language_model.layers.{layer}.mlp.experts."
            gates = state.pop(prefix + "gate_up_proj")
            downs = state.pop(prefix + "down_proj")
            for expert in range(4):
                gate, up = gates[expert].chunk(2, dim=0)
                state[prefix + f"{expert}.gate_proj.weight"] = gate.contiguous()
                state[prefix + f"{expert}.up_proj.weight"] = up.contiguous()
                state[prefix + f"{expert}.down_proj.weight"] = downs[
                    expert
                ].contiguous()
        save_file(state, str(path))
        adapter = TargetAdapter.load(self.target, [0, 2], "cpu")
        self.torch.testing.assert_close(
            adapter.features(tokens), expected, rtol=0, atol=0
        )
