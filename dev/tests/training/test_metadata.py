import contextlib
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.training.fixtures import tiny_config, write_data
from draft_training import config
from draft_training.__main__ import main, parse_args
from draft_training.data import conversations, disjoint, examples, split_manifest

ROOT = Path(__file__).resolve().parents[3]


def sparse_checkpoint(directory, shapes, dtype="BF16"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    offset, header = 0, {}
    for name, shape in shapes.items():
        import math

        size = math.prod(shape) * {"BF16": 2, "F32": 4}[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    with (directory / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.truncate(8 + len(raw) + offset)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_valid_native_family_contract_and_shape_rejection(self):
        raw = config.read_json(
            ROOT / "dev/tests/fixtures/model-configs/qwen3.8-27b/draft/config.json"
        )
        shapes = config.expected_tensors(raw)
        sparse_checkpoint(self.root, shapes)
        config.write_json(self.root / "config.json", raw)
        self.assertEqual(config.validate_draft(self.root), raw)
        raw["dflash_config"]["target_layer_ids"][0] += 1
        config.write_json(self.root / "config.json", raw)
        with self.assertRaisesRegex(config.TrainingError, "dflash_config"):
            config.validate_draft(self.root)

    def test_dtype_and_missing_tensor_rejected(self):
        raw = tiny_config()
        sparse_checkpoint(self.root, config.expected_tensors(raw), dtype="F32")
        config.write_json(self.root / "config.json", raw)
        with self.assertRaisesRegex(config.TrainingError, "BF16"):
            config.validate_draft(self.root, family=False)
        shapes = config.expected_tensors(raw)
        del shapes["fc.weight"]
        sparse_checkpoint(self.root, shapes)
        with self.assertRaisesRegex(config.TrainingError, "tensor names"):
            config.validate_draft(self.root, family=False)

    def test_unsupported_target_rejected_before_any_tensor_loading(self):
        config.write_json(
            self.root / "config.json",
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "quantization_config": {"quant_method": "modelopt"},
            },
        )
        with self.assertRaisesRegex(config.TrainingError, "unquantized BF16"):
            config.validate_target(self.root)
        config.write_json(self.root / "config.json", {"model_type": "qwen3"})
        with self.assertRaisesRegex(config.TrainingError, "Transformers"):
            config.validate_target(self.root)

    def test_shard_path_and_index_mismatch_rejected(self):
        config.write_json(
            self.root / "model.safetensors.index.json",
            {"weight_map": {"x": "../outside.safetensors"}},
        )
        with self.assertRaisesRegex(config.TrainingError, "checkpoint directory"):
            config.checkpoint_index(self.root)
        sparse_checkpoint(self.root, {"x": [2]})
        config.write_json(
            self.root / "model.safetensors.index.json",
            {"weight_map": {"y": "model.safetensors"}},
        )
        with self.assertRaisesRegex(config.TrainingError, "index mismatch"):
            config.checkpoint_index(self.root)

    def test_recipe_rejects_unknown_negative_boolean_and_nonfinite_settings(self):
        for values in (
            {"unknown": 1},
            {"max_steps": 0},
            {"batch_size": True},
            {"learning_rate": float("nan")},
            {"trainable": "magic"},
            {"selector_loss_weight": 0},
        ):
            with self.subTest(values=values):
                # JSON is valid YAML, including NaN for this parser check.
                path = self.root / "recipe.yaml"
                path.write_text(json.dumps(values))
                try:
                    import yaml  # noqa: F401
                except ImportError:
                    self.skipTest("optional PyYAML is not installed")
                with self.assertRaises(config.TrainingError):
                    config.Recipe.load(path)

    def test_normalized_prompt_and_source_group_overlap_rejected(self):
        a, b = self.root / "a.jsonl", self.root / "b.jsonl"
        write_data(a, "same  prompt", "answer one")
        write_data(b, "same\nprompt", "different answer")
        with self.assertRaisesRegex(config.TrainingError, "overlap"):
            disjoint(split_manifest(a), split_manifest(b))
        write_data(a, "first", source_id="conversation-1")
        write_data(b, "second", source_id="conversation-1")
        with self.assertRaisesRegex(config.TrainingError, "overlap"):
            disjoint(split_manifest(a), split_manifest(b))
        write_data(b, "second", source_id="conversation-2")
        disjoint(split_manifest(a), split_manifest(b))

    def test_generated_supervision_never_uses_original_future_turns(self):
        path = self.root / "data.jsonl"
        write_data(path, "question", "untrusted original answer")
        records = conversations(path)
        generated = list(examples(records, "target-generated"))
        self.assertEqual(generated, [([{"role": "user", "content": "question"}], None)])
        supplied = list(examples(records, "supplied-answers"))
        self.assertEqual(supplied[0][1], "untrusted original answer")

    def test_cli_refuses_final_test_data_during_training(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(
                [
                    "train",
                    "--target",
                    ".",
                    "--init-draft",
                    ".",
                    "--data",
                    "train",
                    "--validation-data",
                    "val",
                    "--output",
                    "out",
                    "--test-data",
                    "test",
                ]
            )

    def test_help_and_metadata_modules_do_not_import_torch(self):
        code = 'import sys; from draft_training.__main__ import parse_args; assert "torch" not in sys.modules; parse_args(["--help"])'
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validate", result.stdout)

    def test_dry_run_rejects_splits_without_creating_output(self):
        a, b, output = (
            self.root / "a.jsonl",
            self.root / "b.jsonl",
            self.root / "output",
        )
        write_data(a)
        write_data(b)
        with (
            mock.patch("draft_training.__main__.validate_target"),
            mock.patch(
                "draft_training.__main__.validate_draft", return_value=tiny_config()
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = main(
                [
                    "train",
                    "--target",
                    str(self.root / "target"),
                    "--init-draft",
                    str(self.root / "draft"),
                    "--data",
                    str(a),
                    "--validation-data",
                    str(b),
                    "--output",
                    str(output),
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 1)
        self.assertFalse(output.exists())

    def test_target_accepts_serialized_layer_types_without_legacy_interval_field(self):
        target = config.read_json(
            ROOT / "dev/tests/fixtures/model-configs/qwen3.8-27b/config.json"
        )
        target.pop("quantization", None)
        target.pop("quantization_config", None)
        target["text_config"].pop("full_attention_interval", None)
        config.write_json(self.root / "config.json", target)
        config.write_json(self.root / "tokenizer.json", {})
        config.write_json(
            self.root / "tokenizer_config.json", {"chat_template": "template"}
        )
        shapes = {
            "model.language_model.embed_tokens.weight": [248320, 5120],
            "lm_head.weight": [248320, 5120],
        }
        for layer in range(64):
            prefix = f"model.language_model.layers.{layer}."
            for name, shape in {
                "input_layernorm.weight": [5120],
                "post_attention_layernorm.weight": [5120],
                "mlp.gate_proj.weight": [17408, 5120],
                "mlp.up_proj.weight": [17408, 5120],
                "mlp.down_proj.weight": [5120, 17408],
            }.items():
                shapes[prefix + name] = shape
        sparse_checkpoint(self.root, shapes)
        self.assertEqual(config.validate_target(self.root), target)
        target["text_config"]["layer_types"][0] = "full_attention"
        config.write_json(self.root / "config.json", target)
        with self.assertRaisesRegex(config.TrainingError, "layer_types"):
            config.validate_target(self.root)

    def test_same_user_words_in_different_prompt_contexts_are_not_duplicate(self):
        a, b = self.root / "a.jsonl", self.root / "b.jsonl"
        for path, context in ((a, "Python context"), (b, "Rust context")):
            config.write_json(
                path,
                {
                    "messages": [
                        {"role": "system", "content": context},
                        {"role": "user", "content": "continue"},
                    ]
                },
            )
        # write_json pretty-prints; the dataset itself is one JSON object per line.
        for path in (a, b):
            path.write_text(json.dumps(config.read_json(path)) + "\n")
        disjoint(split_manifest(a), split_manifest(b))
