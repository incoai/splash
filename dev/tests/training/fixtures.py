"""Small independent checkpoints for optional tensor tests; no downloads."""

import json
from pathlib import Path


def tiny_config():
    return {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "num_target_layers": 4,
        "vocab_size": 32,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "max_position_embeddings": 128,
        "sliding_window": 16,
        "use_sliding_window": True,
        "layer_types": ["sliding_attention"] * 2,
        "is_causal": False,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "dflash_config": {
            "block_size": 8,
            "conv_kernel_size": 2,
            "conv_group_size": 4,
            "selector_rank": 4,
            "selector_top_k": 4,
            "target_layer_ids": [0, 2],
            "mask_token_id": 31,
        },
    }


def tiny_components():
    import torch
    from torch import nn

    from draft_training.model import DraftModel

    torch.manual_seed(7)
    config = tiny_config()
    model = DraftModel(config).to(dtype=torch.bfloat16)
    embedding = nn.Embedding(32, 16).to(dtype=torch.bfloat16).requires_grad_(False)
    head = nn.Linear(16, 32, bias=False).to(dtype=torch.bfloat16).requires_grad_(False)
    return model, embedding, head


def cached_features(directory, tokens=None):
    import torch
    from safetensors.torch import save_file

    from draft_training.config import file_digest, write_json

    directory = Path(directory)
    directory.mkdir(parents=True)
    torch.manual_seed(13)
    sample = {
        "tokens": torch.arange(24) % 30 if tokens is None else tokens,
        "features": torch.randn(24, 32, dtype=torch.bfloat16),
        "assistant_mask": torch.arange(24) >= 4,
    }
    path = directory / "000000.safetensors"
    save_file(sample, str(path))
    manifest = {
        "binding": {"supervision": "target-generated"},
        "complete": True,
        "shards": [{"file": path.name, "sha256": file_digest(path), "tokens": 24}],
    }
    write_json(directory / "manifest.json", manifest)
    return sample, manifest


def write_data(path, prompt="hello", answer="response", **extra):
    Path(path).write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                **extra,
            }
        )
        + "\n"
    )


def tiny_target(directory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
    )

    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_conv_kernel_dim=2,
            full_attention_interval=2,
            max_position_embeddings=128,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
            eos_token_id=None,
            pad_token_id=0,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            num_position_embeddings=16,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
        ),
        image_token_id=28,
        video_token_id=29,
    )
    torch.manual_seed(19)
    model = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16).eval()
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "user": 2,
                "assistant": 3,
                "system": 4,
                **{f"w{i}": i for i in range(5, 32)},
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    )
    fast.chat_template = (
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    model.save_pretrained(directory)
    fast.save_pretrained(directory)
    return model, fast
