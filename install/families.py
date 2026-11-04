"""The model families Splash serves, the DFlash2 draft released for each and
the chat format the server reads and writes each one's output in.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.
"""

from __future__ import annotations

from dataclasses import dataclass

from server.chat_format import FLINT, QWEN, ChatFormat


@dataclass(frozen=True)
class SamplingDefaults:
    """The sampling a family's publisher recommends for its models, which a
    request gets for each of these fields it leaves out."""

    temperature: float
    top_p: float
    top_k: int


# What Qwen publishes with its models: temperature 1.0, top_p 0.95, top_k 20.
QWEN_SAMPLING = SamplingDefaults(temperature=1.0, top_p=0.95, top_k=20)
# What Flint's GGUF release states (general.sampling.temp, top_p, top_k),
# which are Qwen's numbers.
FLINT_SAMPLING = SamplingDefaults(temperature=1.0, top_p=0.95, top_k=20)


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp and Flint.hpp).
    name: str
    # The repository of the DFlash2 draft released for the family, as its
    # release publishes it (config.json and BF16 safetensors). Installations
    # follow its default branch as they follow the target's.
    draft_repo: str
    # How the family's model reasons and calls tools, and the tokens it
    # shares with the engine (server/chat_format.py).
    chat_format: ChatFormat
    # The sampling a request gets unless it names its own.
    sampling: SamplingDefaults
    # Whether the family's models have a vision tower, which an installation
    # loads from the target repository unless --language-only. The models of
    # a family without one serve text only, whatever their repository holds.
    vision: bool = True


FAMILIES = (
    ModelFamily("Qwen3.8-27B", "incoai/Qwen3.8-27B-DFlash2", QWEN, QWEN_SAMPLING),
    ModelFamily(
        "Qwen3.6-35B-A3B", "incoai/Qwen3.6-35B-A3B-DFlash2", QWEN, QWEN_SAMPLING
    ),
    # Its fine-tune Ornith-1.5-122B is the family too, and the draft
    # released for it the family's.
    ModelFamily(
        "Qwen3.5-122B-A10B", "incoai/Ornith-1.5-122B-DFlash2", QWEN, QWEN_SAMPLING
    ),
    ModelFamily(
        "Qwen3.8-Flash-Next",
        "incoai/Qwen3.8-Flash-Next-DFlash2",
        QWEN,
        QWEN_SAMPLING,
    ),
    # It has no vision tower.
    ModelFamily("Flint", "incoai/Flint-DFlash2", FLINT, FLINT_SAMPLING, vision=False),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)
