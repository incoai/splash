"""The model families Splash serves: each architecture's signature and the
DFlash2 draft trained for it, if one exists.

A target is identified by its own configuration, never by its repository's
name: an MLX config.json states it, and gguf.model_config derives the same
fields from a GGUF header. Legacy Splash packages pack these same layouts.
"""

from __future__ import annotations

from dataclasses import dataclass

if __package__:
    from . import models
else:
    import models


@dataclass(frozen=True)
class Draft:
    # The repository of the DFlash2 checkpoint trained for the family, as its
    # release publishes it: config.json and BF16 safetensors. Installations
    # follow its default branch as they follow the target's.
    repo: str
    # The config.json fields, dotted into its objects, that the native draft
    # inspection requires, with the values it requires: a commit stating
    # others is not installed, so it never replaces a draft that loads.
    signature: tuple[tuple[str, object], ...]

    @property
    def layers(self):
        return dict(self.signature)["num_hidden_layers"]


# The fields every DFlash2 draft the runtime loads states alike.
DFLASH2 = (
    ("architectures", ("DFlash2DraftModel",)),
    ("sliding_window", 2048),
    ("is_causal", False),
    ("attention_bias", False),
    ("tie_word_embeddings", False),
    ("rms_norm_eps", 1e-6),
    ("hidden_act", "silu"),
    ("rope_parameters.rope_type", "default"),
    ("rope_parameters.rope_theta", 10000000),
    ("dflash_config.block_size", 8),
    ("dflash_config.conv_group_size", 16),
    ("dflash_config.conv_kernel_size", 2),
    ("dflash_config.selector_top_k", 16),
)


@dataclass(frozen=True)
class ModelFamily:
    name: str
    # The text_config fields that identify the architecture, as an MLX config
    # states them and as gguf.model_config derives them from a GGUF header,
    # including every one the native source model inspection requires.
    signature: tuple[tuple[str, object], ...]
    # None for a family no DFlash2 draft was trained for: it decodes without
    # one, and --draft-model is refused for it.
    draft: Draft | None


FAMILIES = (
    ModelFamily(
        "Qwen3.8-27B",
        (
            ("model_type", "qwen3_5_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 5120),
            ("num_hidden_layers", 64),
            ("vocab_size", 248320),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 4),
            ("head_dim", 256),
        ),
        Draft(
            "incoai/Qwen3.8-27B-DFlash2",
            DFLASH2
            + (
                ("num_hidden_layers", 5),
                ("hidden_size", 5120),
                ("vocab_size", 248320),
                ("intermediate_size", 17408),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 128),
                ("dflash_config.selector_rank", 256),
                ("dflash_config.mask_token_id", 248070),
                ("dflash_config.target_layer_ids", (5, 19, 33, 47, 61)),
            ),
        ),
    ),
    ModelFamily(
        "Qwen3.6-35B-A3B",
        (
            ("model_type", "qwen3_5_moe_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 2048),
            ("num_hidden_layers", 40),
            ("vocab_size", 248320),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 2),
            ("head_dim", 256),
            ("num_experts", 256),
            ("num_experts_per_tok", 8),
        ),
        Draft(
            "incoai/Qwen3.6-35B-A3B-DFlash2",
            DFLASH2
            + (
                ("num_hidden_layers", 6),
                ("hidden_size", 2048),
                ("vocab_size", 248320),
                ("intermediate_size", 6144),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 128),
                ("dflash_config.selector_rank", 256),
                ("dflash_config.mask_token_id", 248077),
                ("dflash_config.target_layer_ids", (1, 6, 11, 16, 22, 27, 32, 37)),
            ),
        ),
    ),
    ModelFamily(
        "Qwen3.8-Flash-Next",
        (
            ("model_type", "qwen4_exp_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 2560),
            ("num_hidden_layers", 48),
            ("vocab_size", 248320),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 2),
            ("head_dim", 256),
            ("num_experts", 512),
            ("num_experts_per_tok", 10),
            ("hc_count", 4),
            ("hc_lowrank", 320),
        ),
        None,
    ),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)


def family_for(config):
    """The one family whose architecture the target's config states."""
    text = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(text, dict):
        raise models.ModelError("upstream configuration has no text_config")
    matches = [f for f in FAMILIES if all(text.get(k) == v for k, v in f.signature)]
    if len(matches) != 1:
        keys = sorted({key for family in FAMILIES for key, _ in family.signature})
        found = ", ".join(f"{key}={text.get(key)}" for key in keys if key in text)
        raise models.ModelError(
            f"no supported model has this architecture ({found}); "
            f"supported: {', '.join(f.name for f in FAMILIES)}"
        )
    return matches[0]
