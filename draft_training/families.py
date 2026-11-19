"""Explicit training adapters matching Splash's released target/draft pairs.

Adding a family requires checkpoint validation, a teacher loader and pipeline
tests. A shared DFlash2 architecture alone does not qualify another target.
"""

_DRAFT = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "vocab_size": 248320,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "sliding_window": 2048,
    "rms_norm_eps": 1e-6,
    "is_causal": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
}
_FLASH = {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "selector_rank": 256,
    "selector_top_k": 16,
}
_TARGET = {
    "vocab_size": 248320,
    "head_dim": 256,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "rms_norm_eps": 1e-6,
}

TRAINING_FAMILIES = (
    {
        "name": "Qwen3.8-27B",
        "model_type": "qwen3_5",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "layout": "runtime/model/Qwen3_8.hpp",
        "target": {
            **_TARGET,
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "intermediate_size": 17408,
            "linear_num_value_heads": 48,
        },
        "draft": {
            **_DRAFT,
            "num_hidden_layers": 5,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": ["sliding_attention"] * 5,
        },
        "flash": {
            **_FLASH,
            "mask_token_id": 248070,
            "target_layer_ids": [5, 19, 33, 47, 61],
        },
    },
    {
        "name": "Qwen3.6-35B-A3B",
        "model_type": "qwen3_5_moe",
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "layout": "runtime/model/Qwen3_6Moe.hpp",
        "target": {
            **_TARGET,
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "linear_num_value_heads": 32,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
        },
        "draft": {
            **_DRAFT,
            "num_hidden_layers": 6,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "layer_types": ["sliding_attention"] * 6,
        },
        "flash": {
            **_FLASH,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
        },
    },
)
