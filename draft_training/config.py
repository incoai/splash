"""Metadata checks usable without PyTorch, CUDA, or Splash's native engine."""

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, fields
from pathlib import Path


class TrainingError(ValueError):
    pass


def read_json(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TrainingError(f"expected a JSON object: {path}")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Replace one metadata file atomically; never expose a partial manifest."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class Recipe:
    seed: int = 42
    max_steps: int = 100
    sequence_length: int = 2048
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    selector_loss_weight: float = 1.0
    selector_warmup_steps: int = 10
    checkpoint_interval: int = 10
    validation_interval: int = 10
    max_new_tokens: int = 128
    enable_thinking: bool = True
    reasoning_effort: str = "xhigh"
    max_validation_blocks: int = 64
    vocabulary_chunk_rows: int = 1
    trainable: str = "selector"

    @classmethod
    def load(cls, path=None):
        values = {}
        if path:
            import yaml

            values = yaml.safe_load(Path(path).read_text())
            if not isinstance(values, dict):
                raise TrainingError("recipe must be a YAML mapping")
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise TrainingError(f"unknown recipe settings: {sorted(unknown)}")
        recipe = cls(**values)
        if type(recipe.enable_thinking) is not bool:
            raise TrainingError("enable_thinking must be a boolean")
        if recipe.reasoning_effort not in ("low", "medium", "xhigh"):
            raise TrainingError("reasoning_effort must be low, medium, or xhigh")
        positive = (
            "max_steps",
            "sequence_length",
            "batch_size",
            "gradient_accumulation",
            "checkpoint_interval",
            "validation_interval",
            "max_new_tokens",
            "max_validation_blocks",
            "vocabulary_chunk_rows",
        )
        for name in positive:
            value = getattr(recipe, name)
            if type(value) is not int or value <= 0:
                raise TrainingError(f"{name} must be a positive integer")
        for name in ("seed", "selector_warmup_steps"):
            value = getattr(recipe, name)
            if type(value) is not int or value < 0:
                raise TrainingError(f"{name} must be a nonnegative integer")
        for name in (
            "learning_rate",
            "max_grad_norm",
            "weight_decay",
            "selector_loss_weight",
        ):
            value = getattr(recipe, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise TrainingError(f"{name} must be finite")
            if value < 0 or (name in ("learning_rate", "max_grad_norm") and value == 0):
                raise TrainingError(f"invalid {name}")
        if recipe.trainable not in ("selector", "all"):
            raise TrainingError("trainable must be selector or all")
        if recipe.sequence_length < 16:
            raise TrainingError("sequence_length must be at least 16")
        if recipe.max_new_tokens >= recipe.sequence_length - 1:
            raise TrainingError("max_new_tokens must leave room for the prompt")
        if recipe.trainable == "selector" and recipe.selector_loss_weight == 0:
            raise TrainingError(
                "selector-only training needs a positive selector loss weight"
            )
        return recipe

    def to_dict(self):
        return asdict(self)

    def chat_template_kwargs(self):
        return {
            "enable_thinking": self.enable_thinking,
            "reasoning_effort": self.reasoning_effort,
        }


def checkpoint_index(directory):
    """Validate safetensors headers and index paths without allocating tensors."""
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    weight_map = (
        read_json(index_path).get("weight_map") if index_path.exists() else None
    )
    if index_path.exists() and (not isinstance(weight_map, dict) or not weight_map):
        raise TrainingError("invalid safetensors weight_map")
    names = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    tensors = {}
    for name in names:
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise TrainingError(
                "weight shards must be safetensors files in the checkpoint directory"
            )
        path = directory / name
        with path.open("rb") as stream:
            size = path.stat().st_size
            raw = stream.read(8)
            if len(raw) != 8:
                raise TrainingError(f"truncated safetensors file: {name}")
            length = struct.unpack("<Q", raw)[0]
            if not 2 <= length <= min(size - 8, 100_000_000):
                raise TrainingError(f"invalid safetensors header length: {name}")
            header = json.loads(stream.read(length))
        end = 0
        ranges = []
        for key, item in header.items():
            if key == "__metadata__":
                continue
            if key in tensors or not isinstance(item, dict):
                raise TrainingError(f"invalid or duplicate tensor: {key}")
            shape, dtype, offsets = (
                item.get("shape"),
                item.get("dtype"),
                item.get("data_offsets"),
            )
            if not isinstance(shape, list) or any(
                type(n) is not int or n < 0 for n in shape
            ):
                raise TrainingError(f"invalid shape: {key}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
            ):
                raise TrainingError(f"invalid offsets: {key}")
            width = {"BF16": 2, "F32": 4, "F16": 2, "I64": 8}.get(dtype)
            start, stop = offsets
            if (
                width is None
                or start < 0
                or stop - start != math.prod(shape) * width
                or stop > size - 8 - length
            ):
                raise TrainingError(f"invalid dtype or tensor extent: {key}")
            if weight_map is not None and weight_map.get(key) != name:
                raise TrainingError(f"shard index mismatch: {key}")
            ranges.append((start, stop))
            tensors[key] = {"shape": shape, "dtype": dtype, "file": name}
        for start, stop in sorted(ranges):
            if start != end:
                raise TrainingError(f"overlapping or noncontiguous tensors: {name}")
            end = stop
        if end != size - 8 - length:
            raise TrainingError(f"unindexed tensor bytes: {name}")
    if not tensors or (weight_map is not None and tensors.keys() != weight_map.keys()):
        raise TrainingError("incomplete safetensors index")
    return tensors


def expected_tensors(config):
    h, m = config["hidden_size"], config["intermediate_size"]
    head = config["head_dim"]
    q, kv = config["num_attention_heads"] * head, config["num_key_value_heads"] * head
    flash = config["dflash_config"]
    rank, vocab = flash["selector_rank"], config["vocab_size"]
    shapes = {
        "fc.weight": [h, h * len(flash["target_layer_ids"])],
        "hidden_norm.weight": [h],
        "norm.weight": [h],
        "candidate_selector.hidden_projection.weight": [rank, h],
        "candidate_selector.predecessor_codebook": [vocab, rank],
        "candidate_selector.successor_codebook": [vocab, rank],
    }
    for layer in range(config["num_hidden_layers"]):
        p = f"layers.{layer}."
        shapes.update(
            {
                p + "input_layernorm.weight": [h],
                p + "post_attention_layernorm.weight": [h],
            }
        )
        for name, shape in {
            "q_proj": [q, h],
            "k_proj": [kv, h],
            "v_proj": [kv, h],
            "o_proj": [h, q],
            "q_norm": [head],
            "k_norm": [head],
        }.items():
            shapes[p + "self_attn." + name + ".weight"] = shape
        for name, shape in {
            "gate_proj": [m, h],
            "up_proj": [m, h],
            "down_proj": [h, m],
        }.items():
            shapes[p + "mlp." + name + ".weight"] = shape
        for name in ("attention_conv", "mlp_conv"):
            shapes[p + name + ".base_kernel"] = [2, flash["conv_kernel_size"], h]
            shapes[p + name + ".kernel_projection.weight"] = [
                2 * flash["conv_kernel_size"] * (h // flash["conv_group_size"]),
                h,
            ]
    return shapes


def validate_draft(directory, *, family=True):
    config = read_json(Path(directory) / "config.json")
    if family:
        required = {
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "num_hidden_layers": 5,
            "hidden_size": 5120,
            "vocab_size": 248320,
            "intermediate_size": 17408,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "sliding_window": 2048,
            "rms_norm_eps": 1e-6,
            "is_causal": False,
            "attention_bias": False,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "layer_types": ["sliding_attention"] * 5,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        }
        for key, value in required.items():
            if config.get(key) != value or isinstance(
                config.get(key), bool
            ) != isinstance(value, bool):
                raise TrainingError(f"Qwen3.8-27B draft config mismatch: {key}")
        flash = {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61],
        }
        if config.get("dflash_config") != flash or any(
            isinstance(v, bool) for v in config.get("dflash_config", {}).values()
        ):
            raise TrainingError("Qwen3.8-27B draft dflash_config mismatch")
        for name, default in [
            ("output_multiplier", 1),
            ("input_embedding_scale", 1),
            ("final_logit_softcapping", None),
        ]:
            if config.get(name, default) != default:
                raise TrainingError(f"unsupported draft {name}")
    tensors = checkpoint_index(directory)
    expected = expected_tensors(config)
    if tensors.keys() != expected.keys():
        raise TrainingError(
            f"draft tensor names mismatch; missing={sorted(expected.keys() - tensors.keys())}, extra={sorted(tensors.keys() - expected.keys())}"
        )
    for name, shape in expected.items():
        if tensors[name]["shape"] != shape or tensors[name]["dtype"] != "BF16":
            raise TrainingError(f"draft tensor must be BF16 with shape {shape}: {name}")
    return config


def validate_target(directory):
    directory = Path(directory)
    config = read_json(directory / "config.json")
    text = config.get("text_config", {})
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != [
        "Qwen3_5ForConditionalGeneration"
    ]:
        raise TrainingError(
            "target must be a Qwen3.8-27B Transformers Qwen3_5ForConditionalGeneration checkpoint"
        )
    for section in (config, text):
        if any(
            key in section
            for key in ("quantization", "quantization_config", "compression_config")
        ):
            raise TrainingError(
                "training requires unquantized BF16 Transformers weights; GGUF/MLX/FP8/NVFP4 are unsupported"
            )
    for name, value in {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "vocab_size": 248320,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "rms_norm_eps": 1e-6,
    }.items():
        if text.get(name) != value:
            raise TrainingError(f"Qwen3.8-27B target config mismatch: {name}")
    expected_layers = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
    ]
    if (
        text.get("layer_types") != expected_layers
        or text.get("full_attention_interval", 4) != 4
    ):
        raise TrainingError("Qwen3.8-27B target layer_types mismatch")
    rope = text.get("rope_parameters", {})
    if (
        rope.get("rope_type") != "default"
        or rope.get("rope_theta") != 10000000
        or rope.get("partial_rotary_factor") != 0.25
    ):
        raise TrainingError("Qwen3.8-27B target RoPE mismatch")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (directory / name).is_file():
            raise TrainingError(f"target needs {name}")
    tokenizer = read_json(directory / "tokenizer_config.json")
    if (
        not tokenizer.get("chat_template")
        and not (directory / "chat_template.jinja").is_file()
    ):
        raise TrainingError("target needs a chat template")
    tensors = checkpoint_index(directory)
    if any(t["dtype"] != "BF16" for t in tensors.values()):
        raise TrainingError("training target weights must all be BF16 safetensors")
    for name, shape in {
        "model.language_model.embed_tokens.weight": [248320, 5120],
        "lm_head.weight": [248320, 5120],
    }.items():
        if name not in tensors or tensors[name]["shape"] != shape:
            raise TrainingError(f"target missing required tensor: {name}")
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}."
        required = {
            "input_layernorm.weight": [5120],
            "post_attention_layernorm.weight": [5120],
            "mlp.gate_proj.weight": [17408, 5120],
            "mlp.up_proj.weight": [17408, 5120],
            "mlp.down_proj.weight": [5120, 17408],
        }
        for suffix, shape in required.items():
            name = prefix + suffix
            if name not in tensors or tensors[name]["shape"] != shape:
                raise TrainingError(f"target missing required decoder tensor: {name}")
    return config


def checkpoint_identity(directory):
    directory = Path(directory)
    names = {item["file"] for item in checkpoint_index(directory).values()}
    names.update(
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix in (".json", ".jinja")
    )
    return {name: file_digest(directory / name) for name in sorted(names)}
